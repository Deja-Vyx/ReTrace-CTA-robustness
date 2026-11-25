"""Metric rows for the comparison report.

build_rows() uses EXACTLY the definitions of retrace/metrics.build_table (flip, confident_flip,
target_hit, violation by INV/DIR/TGT, delta_p, acc_drop...); the only difference is that it takes the
variants and scores directly (to merge ThreatMAMBA scores from the ReTrace runs with the scores of the
case-study systems). tests/test_cta_systems.py compares both functions on the same data.
"""
import numpy as np

from retrace.metrics import _orig_index, _pk, k_of

from ..common import ADV_RUNS, RUNS_DIR, confident_tau, groups, iter_jsonl_any

_VCACHE: dict = {}


def load_variants(run_dir) -> dict:
    key = str(run_dir)
    if key not in _VCACHE:
        _VCACHE[key] = {v["vid"]: v for v in iter_jsonl_any(run_dir / "variants.jsonl.gz")}
    return _VCACHE[key]


def load_scores(path, models=None) -> list[dict]:
    return [s for s in iter_jsonl_any(path) if models is None or s["model"] in models]


def build_rows(variants: dict, scores: list[dict], tau: float | None = None, G: list[str] | None = None) -> list[dict]:
    tau = confident_tau() if tau is None else tau
    G = G or groups()
    g2i = {g: i for i, g in enumerate(G)}
    orig = _orig_index(scores)
    rows = []
    for s in scores:
        vid = s["vid"]
        if "|ORIG|" in vid:
            continue
        v = variants.get(vid)
        if v is None:
            continue
        did = v["doc_id"]
        o = orig.get((did, s["model"], s["keep"]))
        if o is None:
            continue
        expect = v.get("expect") or "INV"
        flip = int(s["top1"] != o["top1"])
        conf_flip = int(flip and s["p_top1"] > tau)
        tgt = (v.get("meta") or {}).get("target")
        target_hit = int(tgt is not None and s["top1"] == g2i.get(tgt, -1))
        viol = target_hit if expect == "TGT" else (conf_flip if expect == "DIR" else flip)
        params = v.get("params", {})
        rows.append({
            "vid": vid, "doc_id": did, "group": v["group"], "op": v["op"], "level": v.get("level"),
            "expect": expect, "params": _pk(params), "k": k_of(params),
            "strategy": v.get("strategy") or "random",
            "model": s["model"], "keep": s["keep"],
            "valid": int((v.get("valid") or {}).get("passed", True)),
            "flip": flip, "confident_flip": conf_flip, "target_hit": target_hit, "violation": viol,
            "delta_p": round(o["p_true"] - s["p_true"], 5),
            "p_true": s["p_true"], "p_true_orig": o["p_true"], "rank_true": s["rank_true"],
            "top1_correct": int(s["rank_true"] == 1), "orig_correct": int(o["rank_true"] == 1),
            "acc_drop": int(o["rank_true"] == 1) - int(s["rank_true"] == 1),
            "p_target": s.get("p_target"), "target": tgt,
            "p_target_orig": (round(float(o["probs"][g2i[tgt]]), 5) if tgt in g2i and o.get("probs") else None),
            "runner_up": v.get("runner_up"),
            "top1": s["top1"],
        })
    return rows


def orig_rows(scores: list[dict], variants: dict, G: list[str] | None = None) -> list[dict]:
    """ORIG rows of every document: clean accuracy on the benchmark set per model / cut-off."""
    out = []
    for s in scores:
        if "|ORIG|" not in s["vid"]:
            continue
        did = s["vid"].split("|")[0]
        v = variants.get(s["vid"]) or {}
        out.append({"doc_id": did, "group": v.get("group"), "model": s["model"], "keep": s["keep"],
                    "correct": int(s["rank_true"] == 1), "top1": s["top1"], "p_true": s["p_true"]})
    return out


# ------------------------------------------------------------------ data sources for the report
def transfer_source(run: str, tm_models: list[str], cmp_models: list[str] | None = None):
    """ReTrace run variants + ThreatMAMBA scores (original run) + case-study scores (cta_systems)."""
    vd = ADV_RUNS / run
    if not (vd / "variants.jsonl.gz").exists():
        return None
    variants = load_variants(vd)
    sc = load_scores(vd / "scores.jsonl.gz", set(tm_models))
    cp = RUNS_DIR / run / "scores.jsonl.gz"
    if cp.exists():
        sc += load_scores(cp, None if cmp_models is None else set(cmp_models))
    return variants, sc


def adaptive_source(tag: str):
    d = RUNS_DIR / f"adapt_{tag}"
    if not (d / "scores.jsonl.gz").exists():
        return None
    return load_variants(d), load_scores(d / "scores.jsonl.gz")


def mean_ci(values_by_doc: dict):
    from retrace.metrics import bootstrap_ci
    return bootstrap_ci(values_by_doc)


def per_doc(rows, metric):
    d = {}
    for r in rows:
        v = r.get(metric)
        if v is not None:
            d.setdefault(r["doc_id"], []).append(v)
    return d


def ci(rows, metric) -> dict:
    pdv = per_doc(rows, metric)
    pt, lo, hi = mean_ci(pdv)
    return {"mean": pt, "lo": lo, "hi": hi, "n_docs": len(pdv), "n": sum(len(v) for v in pdv.values())}


def paired_diff(rows_a, rows_b, metric, docs=None) -> dict:
    """Paired difference (A - B) by document (documents present in both), document-level bootstrap CI."""
    a, b = per_doc(rows_a, metric), per_doc(rows_b, metric)
    common = sorted(set(a) & set(b) & (set(docs) if docs is not None else set(a)))
    diff = {d: [float(np.mean(a[d]) - np.mean(b[d]))] for d in common}
    pt, lo, hi = mean_ci(diff)
    return {"mean": pt, "lo": lo, "hi": hi, "n_docs": len(common)}
