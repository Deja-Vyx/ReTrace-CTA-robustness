"""Base class for the perturbation operators, plus shared helpers.

Every operator (Op) has:
  code   : "T1".."T6", "E1".."E8", "G1".."G5", "ID"
  level  : 1 (text) | 2 (evidence) | 3 (graph) | 0 (identity)
  expect : "INV" (top-1 must not change) | "DIR" (p(true group) may drop, no confident flip)
           | "TGT" (must not be steered to the designated decoy group B)
  candidates(doc, ctx, params, n, rng) -> list[Variant]   (at most n candidates)
  Item-based operators (item_based=True) also provide items() + apply_items() for greedy-k.
"""
import random
import re
from dataclasses import dataclass, field
from typing import Any

from src.ioc_extract import refang
from ..pipeline import Doc, join_sents

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.code] = cls
    return cls


@dataclass
class Variant:
    doc_id: str
    group: str
    op: str
    level: int
    expect: str
    params: dict
    text: str | None = None
    graph: dict | None = None
    edits: list = field(default_factory=list)        # [{"sent": i, "before": ..., "after": ...}]
    aligned: bool = True                             # 1 sentence -> 1 sentence
    meta: dict = field(default_factory=dict)
    cand: int = 0
    strategy: str = "random"
    selected: bool = True


@dataclass
class Ctx:
    """Everything an operator may need: pipeline, llm, pools, donors, config."""
    pipeline: Any
    cfg: dict
    llm: Any = None
    pools: Any = None
    donors: dict = field(default_factory=dict)       # {group: [Doc,...]} used by G4
    zoo: Any = None
    groups: list = field(default_factory=list)

    def get_llm(self):
        """Create the LLMClient on demand (only LLM operators call this)."""
        if self.llm is None:
            from ..llm import get_llm
            self.llm = get_llm(self.cfg)
            ok, why = self.llm.check()
            if not ok:
                # stop immediately instead of failing thousands of times (the runner does not swallow SystemExit)
                raise SystemExit(f"[llm] {why}")
        return self.llm


class Op:
    code = "?"
    level = 0
    expect = "INV"
    needs_llm = False
    item_based = False
    aligned = True

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    # --- default parameters when the config does not declare any
    default_grid: dict = {}

    def candidates(self, doc: Doc, params: dict, n: int, rng: random.Random) -> list[Variant]:
        if self.item_based:
            items = self.items(doc)
            k = params.get("k", 1)
            if not items:
                return []
            out, seen = [], set()
            for c in range(max(1, n) * 3):
                if len(out) >= n:
                    break
                kk = len(items) if k == "all" else min(int(k), len(items))
                chosen = tuple(sorted(rng.sample(range(len(items)), kk)))
                if chosen in seen:
                    continue
                seen.add(chosen)
                v = self.apply_items(doc, [items[i] for i in chosen], params, rng)
                if v is not None:
                    v.cand = len(out)
                    out.append(v)
            return out
        raise NotImplementedError

    # --- item-based operators
    def items(self, doc: Doc) -> list:
        return []

    def apply_items(self, doc: Doc, items: list, params: dict, rng: random.Random) -> Variant | None:
        raise NotImplementedError

    # --- helpers
    def make_text_variant(self, doc: Doc, new_sents: list[str], params: dict, meta=None,
                          aligned: bool | None = None) -> Variant | None:
        edits = []
        if len(new_sents) == len(doc.sents):
            for i, (a, b) in enumerate(zip(doc.sents, new_sents)):
                if a != b:
                    edits.append({"sent": i, "before": a, "after": b})
        else:
            edits.append({"sent": -1, "before": f"{len(doc.sents)} sentences", "after": f"{len(new_sents)} sentences"})
        if len(new_sents) == len(doc.sents) and not edits:
            return None                                     # nothing changed -> skip
        text = join_sents(new_sents)
        return Variant(doc_id=doc.doc_id, group=doc.group, op=self.code, level=self.level,
                       expect=self.expect, params=dict(params), text=text, edits=edits,
                       aligned=self.aligned if aligned is None else aligned, meta=meta or {})

    def make_graph_variant(self, doc: Doc, g: dict, params: dict, meta=None) -> Variant:
        return Variant(doc_id=doc.doc_id, group=doc.group, op=self.code, level=self.level,
                       expect=self.expect, params=dict(params), text=None, graph=g,
                       aligned=True, meta=meta or {})


# ---------------------------------------------------------------------- IOC matching, including defanged forms
def defang_regex(value: str) -> re.Pattern:
    """Match value and its common defanged forms: evil[.]com, evil(.)com, hxxp://, [at]."""
    out = []
    i = 0
    v = value
    while i < len(v):
        ch = v[i]
        if v[i:i + 4].lower() == "http":
            out.append(r"h(?:tt|xx|XX|TT)p"); i += 4; continue
        if ch == ".":
            out.append(r"(?:\.|\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\))")
        elif ch == "@":
            out.append(r"(?:@|\[at\]|\(at\)|\[@\])")
        elif ch == ":":
            out.append(r"(?::|\[:\])")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("".join(out), re.I)


def find_value_spans(text: str, value: str) -> list[tuple[int, int]]:
    return [m.span() for m in defang_regex(value).finditer(text)]


# ---------------------------------------------------------------------- placeholders for the LLM
RE_TECH_ID = re.compile(r"\bT[AS]?\d{4}(?:\.\d{3})?\b")
RE_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)


def protected_spans(sentence: str, extra_values=()) -> list[tuple[int, int]]:
    """Spans the LLM must NOT edit: IOCs, group names, tool names, technique IDs, CVEs."""
    from src.ioc_extract import extract_iocs
    from ..lexicons import dataset_groups, find_group_mentions, find_tool_mentions
    spans = []
    for i in extract_iocs(sentence, [sentence]):
        spans += find_value_spans(sentence, i["value"])
    for v in extra_values:
        spans += find_value_spans(sentence, v)
    for g in dataset_groups():
        spans += [(s, e) for s, e, _a in find_group_mentions(sentence, g)]
    for _name, sp in find_tool_mentions(sentence).items():
        spans += [(s, e) for s, e, _a in sp]
    spans += [m.span() for m in RE_TECH_ID.finditer(sentence)]
    spans += [m.span() for m in RE_CVE.finditer(sentence)]
    # merge overlapping spans
    spans = sorted(set(spans))
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def protect(sentence: str, extra_values=()) -> tuple[str, dict]:
    spans = protected_spans(sentence, extra_values)
    out, last, mapping = [], 0, {}
    for i, (s, e) in enumerate(spans, 1):
        key = f"[P{i}]"
        mapping[key] = sentence[s:e]
        out.append(sentence[last:s]); out.append(key); last = e
    out.append(sentence[last:])
    return "".join(out), mapping


def restore(text: str, mapping: dict) -> str | None:
    """Return None if a placeholder is missing, extra or duplicated."""
    if not isinstance(text, str):
        return None
    for key in mapping:
        if text.count(key) != 1:
            return None
    if re.search(r"\[P\d+\]", re.sub("|".join(re.escape(k) for k in mapping) or r"$^", "", text)):
        return None
    for key, val in mapping.items():
        text = text.replace(key, val)
    return text.strip()


def single_sentence(text: str) -> bool:
    from src.utils import sent_split
    return len(sent_split(text)) == 1


def sentence_has_ioc(sentence: str) -> bool:
    from src.ioc_extract import extract_iocs
    return bool(extract_iocs(sentence, [sentence]))


def refanged(s: str) -> str:
    return refang(s)
