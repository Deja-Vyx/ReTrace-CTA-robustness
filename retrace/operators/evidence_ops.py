"""Level 2 - evidence perturbations (E1-E8).

E1 mask IOC           : replace every occurrence of k IOCs (including defanged forms) with a generic phrase   DIR
E2 mask TTP           : drop (mode=drop) or LLM-neutralise (mode=neutral) the sentences that trigger
                        k techniques                                                            DIR
E3 tool alias         : rename k tools to another alias of the same ATT&CK software              INV
E4 group-name mask    : replace the aliases of the true group (scope=true) / of all 20 groups (scope=all)
                        with "the threat actor"                                                DIR
E5 temporal           : mode=markers removes temporal connectives; mode=swap swaps k pairs of adjacent
                        evidence sentences                                                     DIR
E6 blur (LLM)         : generalise tool names / commands in k evidence sentences                 DIR
E7 random IOC         : insert k valid random IOCs via validated carrier sentences              INV
E8 false flag         : plant traces of group B (kind=ioc|ttp|name), B = runner_up | random      TGT
"""
import ipaddress
import random
import re

from src.utils import sent_split
from ..lexicons import (GENERIC_IOC, dataset_groups, find_group_mentions, find_tool_mentions,
                        group_aliases, mask_spans, strip_temporal_marker)
from ..llm import P_BLUR, P_NEUTRALIZE
from ..pipeline import Doc
from .base import Op, find_value_spans, register
from .text_ops import _LLMRewrite

SKIP_IOC_TYPES = {"Protocol", "Port", "Account"}   # not real attack infrastructure


def _insert_positions(n_sents: int, k: int, window: int, rng: random.Random) -> list[int]:
    limit = max(1, min(n_sents, window - k - 1))
    return sorted(rng.randint(0, limit) for _ in range(k))


def _insert(sents: list[str], new: list[str], positions: list[int]) -> list[str]:
    out = list(sents)
    for off, (pos, s) in enumerate(sorted(zip(positions, new))):
        out.insert(pos + off, s)
    return out


# ---------------------------------------------------------------------- E1
@register
class E1MaskIOC(Op):
    code, level, expect = "E1", 2, "DIR"
    item_based = True
    default_grid = {"k": [1, 2, 3, 5]}

    def items(self, doc: Doc) -> list:
        seen, out = set(), []
        # URLs before Domains so that no half-replaced strings are produced
        order = {"URL": 0, "Email": 1}
        for i in sorted(doc.iocs, key=lambda x: order.get(x["type"], 2)):
            if i["type"] in SKIP_IOC_TYPES:
                continue
            key = (i["value"].lower(), i["type"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"value": i["value"], "type": i["type"], "sent_idx": i["sent_idx"]})
        return out

    def apply_items(self, doc, items, params, rng):
        new = list(doc.sents)
        n_rep = 0
        for it in items:
            phrase = GENERIC_IOC.get(it["type"], "an indicator")
            for i, s in enumerate(new):
                spans = find_value_spans(s, it["value"])
                if not spans:
                    continue
                # extend to the whole token if the IOC sits inside a URL, avoiding half-replaced strings
                ext = []
                for a, b in spans:
                    lo = a
                    while lo > 0 and not s[lo - 1].isspace():
                        lo -= 1
                    hi = b
                    while hi < len(s) and not s[hi].isspace():
                        hi += 1
                    tok = s[lo:hi]
                    if "://" in tok or "hxxp" in tok.lower():
                        ext.append((lo, hi, GENERIC_IOC["URL"]))
                    else:
                        ext.append((a, b, phrase))
                for a, b, ph in sorted(set(ext), key=lambda t: -t[0]):
                    s = s[:a] + ph + s[b:]
                    n_rep += 1
                new[i] = s
        if n_rep == 0:
            return None
        return self.make_text_variant(doc, new, params, meta={
            "items": [f"{it['type']}:{it['value']}" for it in items], "n_replaced": n_rep})


# ---------------------------------------------------------------------- E2
@register
class E2MaskTTP(Op):
    code, level, expect = "E2", 2, "DIR"
    item_based = True
    default_grid = {"k": [1, 2, 3], "mode": ["drop"]}

    def items(self, doc: Doc) -> list:
        sets = self.ctx.pipeline.doc_ttp_sets(doc)
        techs = [t for t in doc.trace.get("tech", [])]
        out = []
        for t in techs:
            sents = [i for i, labs in enumerate(sets) if t in labs]
            if sents:
                out.append({"tech": t, "sents": sents})
        return out

    def apply_items(self, doc, items, params, rng):
        mode = params.get("mode", "drop")
        target_sents = sorted({i for it in items for i in it["sents"]})
        if not target_sents:
            return None
        techs = [it["tech"] for it in items]
        if mode == "drop":
            keep = [s for i, s in enumerate(doc.sents) if i not in set(target_sents)]
            if not keep:
                return None
            return self.make_text_variant(doc, keep, params, aligned=False, meta={
                "items": techs, "dropped_sents": target_sents,
                "ioc_in_dropped": sum(1 for i in doc.iocs if i["sent_idx"] in set(target_sents))})
        # mode == neutral: the LLM neutralises each sentence
        llm = self.ctx.get_llm()
        rewriter = _Neutralizer(self.ctx)
        new = list(doc.sents)
        fallback = 0
        for i in target_sents:
            tech = next(it["tech"] for it in items if i in it["sents"])
            rw = rewriter.rewrite(doc, i, tech)
            if rw:
                new[i] = rw
            else:
                fallback += 1
        if new == doc.sents or llm is None:
            return None
        return self.make_text_variant(doc, new, params, aligned=True, meta={
            "items": techs, "rewritten_sents": target_sents, "llm_fallback": fallback})


class _Neutralizer(_LLMRewrite):
    code, level, expect = "E2", 2, "DIR"
    prompt = P_NEUTRALIZE

    def rewrite(self, doc: Doc, idx: int, tech: str) -> str | None:
        self._tech = tech
        rws = self.rewrites_for(doc, idx, 3, {})
        return rws[0] if rws else None

    def prompt_for(self, masked, n, doc, idx, params):
        name = self.ctx.pipeline.technique_name(self._tech) or self._tech
        return self.prompt.format(n=n, sentence=masked, tech_name=name, tech_id=self._tech)


# ---------------------------------------------------------------------- E3
@register
class E3ToolAlias(Op):
    code, level, expect = "E3", 2, "INV"
    item_based = True
    default_grid = {"k": [1, "all"]}

    def items(self, doc: Doc) -> list:
        return sorted(find_tool_mentions(doc.text))

    def apply_items(self, doc, items, params, rng):
        from ..lexicons import tool_aliases
        ta = tool_aliases()
        new = list(doc.sents)
        swaps = {}
        n_rep = 0
        for name in items:
            als = ta.get(name, [])
            for i, s in enumerate(new):
                spans = find_tool_mentions(s).get(name, [])
                if not spans:
                    continue
                for a, b, used in sorted(spans, key=lambda t: -t[0]):
                    choices = [x for x in als if x != used]
                    if not choices:
                        continue
                    alt = swaps.setdefault((name, used), rng.choice(choices))
                    s = s[:a] + alt + s[b:]
                    n_rep += 1
                new[i] = s
        if n_rep == 0:
            return None
        return self.make_text_variant(doc, new, params, meta={
            "items": list(items), "swaps": {f"{k[0]}|{k[1]}": v for k, v in swaps.items()},
            "n_replaced": n_rep})


# ---------------------------------------------------------------------- E4
@register
class E4GroupMask(Op):
    code, level, expect = "E4", 2, "DIR"
    default_grid = {"scope": ["true", "all"]}
    REPL = "the threat actor"

    def candidates(self, doc, params, n, rng):
        # YAML reads `true` as bool True -> normalise to a string
        scope = str(params.get("scope", "true")).strip().lower()
        groups = [doc.group] if scope in ("true", "1") else list(dataset_groups())
        new, n_rep, n_active = [], 0, 0
        act = set(doc.active)
        for i, s in enumerate(doc.sents):
            spans = []
            for g in groups:
                spans += find_group_mentions(s, g)
            # drop overlapping spans (longer spans win)
            spans = sorted(spans, key=lambda t: (t[0], -(t[1] - t[0])))
            clean, last_end = [], -1
            for sp in spans:
                if sp[0] >= last_end:
                    clean.append(sp); last_end = sp[1]
            if clean:
                s2 = mask_spans(s, clean, self.REPL)
                n_rep += len(clean)
                n_active += len(clean) if i in act else 0
                new.append(s2)
            else:
                new.append(s)
        if n_rep == 0:
            return []
        v = self.make_text_variant(doc, new, params, meta={"n_replaced": n_rep, "n_in_active": n_active})
        if v is None or len(sent_split(v.text)) != len(doc.sents):
            return []
        return [v]


# ---------------------------------------------------------------------- E5
@register
class E5Temporal(Op):
    code, level, expect = "E5", 2, "DIR"
    default_grid = {"mode": ["markers", "swap"], "k": [1, 3]}

    def candidates(self, doc, params, n, rng):
        mode = params.get("mode", "markers")
        if mode == "markers":
            if params.get("k", 1) not in (1, "all") and params.get("k") is not None:
                # k only applies to mode swap; avoid duplicates
                return []
            new, cnt = [], 0
            for s in doc.sents:
                t = strip_temporal_marker(s)
                if t:
                    new.append(t); cnt += 1
                else:
                    new.append(s)
            if cnt == 0:
                return []
            v = self.make_text_variant(doc, new, params, meta={"n_markers": cnt})
            return [v] if v and len(sent_split(v.text)) == len(doc.sents) else []
        # swap
        win = int(self.ctx.cfg.get("insert_window", 128))
        act = [i for i in doc.active if i < win]
        pairs = list(zip(act, act[1:]))
        k = int(params.get("k", 1))
        if not pairs:
            return []
        out = []
        for c in range(max(1, n)):
            rng.shuffle(pairs)
            used, chosen = set(), []
            for a, b in pairs:
                if a in used or b in used:
                    continue
                chosen.append((a, b)); used |= {a, b}
                if len(chosen) >= k:
                    break
            new = list(doc.sents)
            for a, b in chosen:
                new[a], new[b] = new[b], new[a]
            v = self.make_text_variant(doc, new, params, meta={"swapped": sorted(chosen)}, aligned=False)
            if v and len(sent_split(v.text)) == len(doc.sents):
                v.cand = c
                out.append(v)
        return out


# ---------------------------------------------------------------------- E6
@register
class E6Blur(_LLMRewrite):
    code, level, expect = "E6", 2, "DIR"
    prompt = P_BLUR
    default_grid = {"k": [1, 3]}

    def targets(self, doc, params, rng):
        win = int(self.ctx.cfg.get("insert_window", 128))
        act = [i for i in doc.active if i < win]
        with_tool = [i for i in act if find_tool_mentions(doc.sents[i])] or act
        k = params.get("k", 1)
        if not with_tool:
            return []
        if k == "all":
            return with_tool
        return sorted(rng.sample(with_tool, min(int(k), len(with_tool))))


# ---------------------------------------------------------------------- E7
TLDS = ["com", "net", "org", "info", "biz", "xyz", "top", "online", "site", "club"]
SYL = ["ka", "lo", "mi", "ter", "zon", "vex", "dra", "pli", "nor", "sat", "qui", "bel", "cor",
       "fin", "gal", "hup", "jor", "lux", "mev", "nix", "ost", "pra", "rud", "sol", "tav", "urb"]


def random_ioc(typ: str, rng: random.Random) -> str:
    from src.ioc_hunter.filters import is_benign
    if typ == "IP":
        while True:
            ip = ipaddress.ip_address(rng.getrandbits(32))
            if ip.is_global and not ip.is_multicast and not ip.is_reserved:
                return str(ip)
    if typ in ("Domain", "URL"):
        while True:
            name = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 4)))
            dom = f"{name}-{rng.choice(SYL)}{rng.randint(1, 99)}.{rng.choice(TLDS)}"
            if not is_benign(dom, "Domain"):
                break
        if typ == "Domain":
            return dom
        return f"http://{dom}/{rng.choice(['upd', 'gate', 'img', 'news', 'api'])}/{rng.choice(SYL)}.php"
    if typ == "Hash":
        return f"{rng.getrandbits(256):064x}"
    raise ValueError(typ)


@register
class E7RandomIOC(Op):
    code, level, expect = "E7", 2, "INV"
    aligned = False
    default_grid = {"k": [1, 3, 5], "type": ["mix"]}

    def candidates(self, doc, params, n, rng):
        pools = self.ctx.pools
        k = int(params.get("k", 1))
        typ = params.get("type", "mix")
        win = int(self.ctx.cfg.get("insert_window", 128))
        out = []
        for c in range(max(1, n)):
            types = [rng.choice(["IP", "Domain", "Hash", "URL"]) if typ == "mix" else typ for _ in range(k)]
            vals, sents, used_tmpl = [], [], []
            for t in types:
                tmpl = pools.carriers(t) if pools else []
                if not tmpl:
                    continue
                v = random_ioc(t, rng)
                vals.append(f"{t}:{v}")
                tp = rng.choice(tmpl)
                used_tmpl.append(tp)
                sents.append(tp.format(v=v))
            if not sents:
                return []
            pos = _insert_positions(len(doc.sents), len(sents), win, rng)
            car_ttp = (getattr(pools, "_car", None) or {}).get("_ttp", {}) if pools else {}
            n_ttp = sum(1 for t in used_tmpl if car_ttp.get(t))
            v = self.make_text_variant(doc, _insert(doc.sents, sents, pos), params, aligned=False,
                                       meta={"injected": vals, "positions": pos,
                                             "n_carrier_with_ttp": n_ttp})
            if v:
                v.cand = c
                out.append(v)
        return out


# ---------------------------------------------------------------------- E8
@register
class E8FalseFlag(Op):
    code, level, expect = "E8", 2, "TGT"
    aligned = False
    default_grid = {"kind": ["ioc", "ttp", "name"], "k": [1, 3], "target": ["runner_up"]}

    def _target(self, doc: Doc, params: dict, rng: random.Random) -> str | None:
        groups = [g for g in dataset_groups() if g != doc.group]
        if params.get("target", "runner_up") == "runner_up":
            b = doc.meta.get("runner_up")
            return b if b in groups else None
        return rng.choice(groups)

    def candidates(self, doc, params, n, rng):
        kind = params.get("kind", "ioc")
        k = int(params.get("k", 1))
        win = int(self.ctx.cfg.get("insert_window", 128))
        pools = self.ctx.pools
        out = []
        for c in range(max(1, n)):
            b = self._target(doc, params, rng)
            if not b:
                return []
            ff = pools.false_flag(b) if pools else {}
            if kind == "ioc":
                cand = ff.get("iocs", [])
                if not cand:
                    return []
                picks = rng.sample(cand, min(k, len(cand)))
                sents = []
                for p in picks:
                    tmpl = pools.carriers(p["type"]) or pools.carriers("Domain")
                    if tmpl:
                        sents.append(rng.choice(tmpl).format(v=p["value"]))
                if not sents:
                    return []
                pos = _insert_positions(len(doc.sents), len(sents), win, rng)
                new = _insert(doc.sents, sents, pos)
                meta = {"target": b, "injected": [f"{p['type']}:{p['value']}" for p in picks]}
            elif kind == "ttp":
                cand = ff.get("ttp_sents", [])
                if not cand:
                    return []
                picks = rng.sample(cand, min(k, len(cand)))
                pos = _insert_positions(len(doc.sents), len(picks), win, rng)
                new = _insert(doc.sents, [p["sent"] for p in picks], pos)
                meta = {"target": b, "injected_tech": [p.get("tech") for p in picks]}
            else:  # name: attach B's name to k EVIDENCE sentences (non-evidence sentences do not enter the graph)
                act = [i for i in doc.active if i < win]
                if not act:
                    return []
                aliases = [a for a in group_aliases().get(b, [b]) if a.lower() != "the threat actor"]
                alias = b if b in aliases else aliases[0]
                tg = sorted(rng.sample(act, min(k, len(act))))
                new = list(doc.sents)
                for i in tg:
                    s = new[i].rstrip()
                    end = s[-1] if s and s[-1] in ".!?" else ""
                    body = s[:-1] if end else s
                    new[i] = f"{body}, consistent with {alias} tradecraft{end or '.'}"
                meta = {"target": b, "alias": alias, "sents": tg}
            v = self.make_text_variant(doc, new, params, meta=meta,
                                       aligned=(kind == "name"))
            if v:
                v.cand = c
                out.append(v)
        return out
