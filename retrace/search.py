"""Variant selection strategies (they only use the output scores of the TARGET MODEL: black-box).

- random     : keep every random candidate (measures average robustness).
- best_of_n  : generate N candidates, keep the one with the LOWEST p(true group) (strongest attack).
- greedy_k   : for item-based operators, each step adds the item that lowers p(true group) the
               most, for at most k steps (prefixes k=1..k_max).
- targeted   : for E8, keep the candidate with the LARGEST p(decoy B) - p(true group).

score_fn(variant) -> (p_true, p_target|None) is supplied by the runner (it builds the graph and
scores the target model at the current timeline cut-off).
"""
import random


def pick_random(cands: list, rng: random.Random, keep_n: int = 1) -> list:
    if not cands:
        return []
    if keep_n >= len(cands):
        return cands
    return rng.sample(cands, keep_n)


def pick_best_of_n(cands: list, score_fn) -> list:
    """score_fn(v) -> p_true. Keep the candidate with the lowest p_true."""
    if not cands:
        return []
    scored = [(score_fn(v), v) for v in cands]
    scored.sort(key=lambda t: t[0])
    best = scored[0][1]
    best.meta = dict(best.meta or {}); best.meta["p_true_search"] = round(float(scored[0][0]), 5)
    best.strategy = "best_of_n"
    return [best]


def pick_targeted(cands: list, score_fn_pair) -> list:
    """score_fn_pair(v) -> (p_true, p_target). Keep the candidate with the largest (p_target - p_true)."""
    if not cands:
        return []
    scored = []
    for v in cands:
        pt, pg = score_fn_pair(v)
        scored.append((pg - pt, pt, pg, v))
    scored.sort(key=lambda t: -t[0])
    _d, pt, pg, best = scored[0]
    best.meta = dict(best.meta or {}); best.meta.update({"p_true_search": round(pt, 5),
                                                        "p_target_search": round(pg, 5)})
    best.strategy = "targeted" if len(cands) > 1 else (best.strategy or "random")
    return [best]


def greedy_k(op, doc, k_max: int, score_fn, rng: random.Random, max_items: int = 10,
             base_params: dict | None = None) -> list:
    """Item-based operators only. Returns one variant per prefix k=1..k_max (each is the greedy
    result after k steps)."""
    items = op.items(doc)
    if not items:
        return []
    if len(items) > max_items:
        items = items[:max_items]
    chosen: list = []
    remaining = list(range(len(items)))
    out = []
    base_params = {k: v for k, v in (base_params or {}).items() if k != "k"}
    for step in range(min(k_max, len(items))):
        best = None
        for idx in remaining:
            v = op.apply_items(doc, [items[i] for i in chosen + [idx]], base_params, rng)
            if v is None:
                continue
            s = score_fn(v)
            if best is None or s < best[0]:
                best = (s, idx, v)
        if best is None:
            break
        _s, idx, v = best
        chosen.append(idx); remaining.remove(idx)
        v.params = {**base_params, "k": len(chosen), "greedy": True}
        v.strategy = "greedy"
        v.meta = dict(v.meta or {}); v.meta["p_true_search"] = round(float(best[0]), 5)
        out.append(v)
    return out
