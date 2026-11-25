"""Level 1 - text perturbations (T1-T6) + the identity operator (ID).

T1 paraphrase (LLM)       : rewrite k evidence sentences, keeping 1 sentence -> 1 sentence   INV
T2 restructure (LLM)      : split/merge sentences, active <-> passive voice                  INV
T3 abbreviation           : abbreviation <-> full form (lexicon), with a "touch_regex" flag  INV
T4 block reorder          : move the trailing IOC appendix to the top                        INV
T5 distractor insertion   : insert k unrelated sentences (validated pool)                    INV
T6 date format            : change the date format (negative control, expected ~0)           INV
"""
import random
import re

from src.utils import sent_split
from ..lexicons import ABBREV, reformat_dates
from ..llm import P_PARAPHRASE, P_RESTRUCTURE, check_rewrites
from ..pipeline import Doc, join_sents
from .base import Op, Variant, protect, register, restore, sentence_has_ioc, single_sentence

EXPAND_EXCLUDE = {"PS", "AD", "OS", "IOC", "IOCs", "APT"}   # abbreviations too ambiguous to expand


def _active_in_window(doc: Doc, window: int) -> list[int]:
    return [i for i in doc.active if i < window]


@register
class Identity(Op):
    code, level, expect = "ID", 0, "INV"

    def candidates(self, doc, params, n, rng):
        return [Variant(doc_id=doc.doc_id, group=doc.group, op="ID", level=0, expect="INV",
                        params={}, text=join_sents(doc.sents), edits=[], aligned=True)]


class _LLMRewrite(Op):
    """Common frame for T1/T2/E2(neutral)/E6: rewrite some sentences with the LLM using placeholders."""
    needs_llm = True
    prompt = P_PARAPHRASE
    allow_multi = False

    def targets(self, doc: Doc, params: dict, rng: random.Random) -> list[int]:
        win = int(self.ctx.cfg.get("insert_window", 128))
        act = _active_in_window(doc, win)
        k = params.get("k", 1)
        if not act:
            return []
        if k == "all":
            return act
        return sorted(rng.sample(act, min(int(k), len(act))))

    def prompt_for(self, masked: str, n: int, doc: Doc, idx: int, params: dict) -> str:
        return self.prompt.format(n=n, sentence=masked)

    def rewrites_for(self, doc: Doc, idx: int, n: int, params: dict) -> list[str]:
        sent = doc.sents[idx]
        masked, mapping = protect(sent)
        obj = self.ctx.get_llm().json(self.prompt_for(masked, n, doc, idx, params),
                                check=lambda o: check_rewrites(o, 1), tag=self.code)
        if not obj:
            return []
        good = []
        lo, hi = self.ctx.cfg.get("validate", {}).get("len_ratio", [0.5, 1.8])
        for r in obj:
            t = restore(r, mapping)
            if not t or t == sent:
                continue
            ratio = len(t) / max(1, len(sent))
            if not (float(lo) * 0.7 <= ratio <= float(hi) * 1.4):
                continue
            nsent = len(sent_split(t))
            if not self.allow_multi and nsent != 1:
                continue
            if self.allow_multi and nsent > 2:
                continue
            good.append(t)
        return good

    def candidates(self, doc, params, n, rng):
        tg = self.targets(doc, params, rng)
        if not tg:
            return []
        per = {i: self.rewrites_for(doc, i, max(n, 1), params) for i in tg}
        per = {i: r for i, r in per.items() if r}
        if not per:
            return []
        out = []
        for c in range(max(1, n)):
            new = []
            for i, s in enumerate(doc.sents):
                if i in per:
                    new.append(per[i][c % len(per[i])])
                else:
                    new.append(s)
            v = self.make_text_variant(doc, new, params,
                                       meta={"targets": sorted(per), "n_targets": len(per)},
                                       aligned=not self.allow_multi)
            if v is not None:
                v.cand = c
                out.append(v)
        return out


@register
class T1Paraphrase(_LLMRewrite):
    code, level, expect = "T1", 1, "INV"
    prompt = P_PARAPHRASE
    default_grid = {"k": [1, 3, "all"]}


@register
class T2Restructure(_LLMRewrite):
    code, level, expect = "T2", 1, "INV"
    prompt = P_RESTRUCTURE
    allow_multi = True
    aligned = False
    default_grid = {"k": [1, 3]}


@register
class T3Abbrev(Op):
    code, level, expect = "T3", 1, "INV"
    default_grid = {"direction": ["contract", "expand"], "touch": ["no_regex", "regex"], "k": ["all"]}

    def _occurrences(self, doc: Doc, direction: str, touch: str) -> list[tuple]:
        occ = []
        for full, abbr, rx in ABBREV:
            if touch == "no_regex" and rx:
                continue
            if touch == "regex" and not rx:
                continue
            if direction == "contract":
                pat = re.compile(rf"(?<![\w\-]){re.escape(full)}(?![\w\-])", re.I)
                repl = abbr
            else:
                if abbr in EXPAND_EXCLUDE:
                    continue
                pat = re.compile(rf"(?<![\w\-]){re.escape(abbr)}(?![\w\-])")
                repl = full
            for i, s in enumerate(doc.sents):
                for m in pat.finditer(s):
                    occ.append((i, m.start(), m.end(), repl, rx))
        return occ

    def candidates(self, doc, params, n, rng):
        occ = self._occurrences(doc, params.get("direction", "contract"), params.get("touch", "no_regex"))
        if not occ:
            return []
        k = params.get("k", "all")
        out = []
        for c in range(max(1, n) if k != "all" else 1):
            chosen = occ if k == "all" else rng.sample(occ, min(int(k), len(occ)))
            new = list(doc.sents)
            by_sent: dict[int, list] = {}
            for o in chosen:
                by_sent.setdefault(o[0], []).append(o)
            for i, lst in by_sent.items():
                s = new[i]
                taken = []
                for _i, a, b, repl, _rx in sorted(lst, key=lambda t: -t[1]):
                    if any(not (b <= x or a >= y) for x, y in taken):
                        continue
                    s = s[:a] + repl + s[b:]
                    taken.append((a, b))
                new[i] = s
            v = self.make_text_variant(doc, new, params, meta={
                "n_replaced": len(chosen), "touches_regex": any(o[4] for o in chosen)})
            if v is not None and len(sent_split(v.text)) == len(doc.sents):
                v.cand = c
                out.append(v)
        return out


@register
class T4BlockReorder(Op):
    code, level, expect = "T4", 1, "INV"
    aligned = False

    def candidates(self, doc, params, n, rng):
        ioc_by_sent: dict[int, int] = {}
        for i in doc.iocs:
            ioc_by_sent[i["sent_idx"]] = ioc_by_sent.get(i["sent_idx"], 0) + 1
        j = len(doc.sents)
        while j > 0 and ioc_by_sent.get(j - 1, 0) >= 1:
            j -= 1
        block = list(range(j, len(doc.sents)))
        if len(block) < 2 or sum(ioc_by_sent.get(i, 0) for i in block) < 2 * len(block) or j == 0:
            return []
        new = [doc.sents[i] for i in block] + [doc.sents[i] for i in range(j)]
        v = self.make_text_variant(doc, new, params, meta={"n_moved": len(block)}, aligned=False)
        return [v] if v else []


@register
class T5Distractor(Op):
    code, level, expect = "T5", 1, "INV"
    aligned = False
    # pool=benign: benign to a human reader (the TTP extractor may still label them -> measures
    # false-positive propagation); pool=ttp_free: only sentences without TTP labels (pure text insertion)
    default_grid = {"k": [1, 3, 5], "pool": ["benign", "ttp_free"]}

    def candidates(self, doc, params, n, rng):
        kind = params.get("pool", "benign")
        pool = self.ctx.pools.distractors(kind) if self.ctx.pools else []
        if not pool:
            return []
        ttp_lab = (getattr(self.ctx.pools, "_dis", None) or {}).get("ttp_labels", {}) if self.ctx.pools else {}
        win = int(self.ctx.cfg.get("insert_window", 128))
        k = int(params.get("k", 1))
        out = []
        for c in range(max(1, n)):
            sents = list(doc.sents)
            picks = rng.sample(pool, min(k, len(pool)))
            limit = max(1, min(len(sents), win - k - 1))
            for p in picks:
                pos = rng.randint(0, limit)
                sents.insert(pos, p)
            v = self.make_text_variant(doc, sents, params, aligned=False, meta={
                "inserted": picks, "n_inserted_with_ttp": sum(1 for p in picks if ttp_lab.get(p))})
            if v:
                v.cand = c
                out.append(v)
        return out


@register
class T6DateFormat(Op):
    code, level, expect = "T6", 1, "INV"

    def candidates(self, doc, params, n, rng):
        new, total = [], 0
        for s in doc.sents:
            t, k = reformat_dates(s)
            new.append(t); total += k
        if total == 0:
            return []
        v = self.make_text_variant(doc, new, params, meta={"n_dates": total})
        if v is None or len(sent_split(v.text)) != len(doc.sents):
            return []
        return [v]
