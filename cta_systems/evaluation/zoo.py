"""Load the case-study models by tag and score a batch of texts at the timeline cut-offs.

Every Item holds the text, its sentence list (sent_split) and its `prepare` record (cut-offs +
ThreatMAMBA techniques). The cut-offs are identical to how ReTrace scores text baselines
(models.text_cut), so every model is compared at the same cut position as ThreatMAMBA.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.utils import sent_split

from ..common import KEEPS
from ..prepare import sents_at, techs_at


@dataclass
class Item:
    text: str
    rec: dict
    sents: list | None = None

    def get_sents(self) -> list[str]:
        if self.sents is None:
            self.sents = sent_split(self.text)
        return self.sents


class TTPVecScorer:
    kind = "ttp_vec"

    def __init__(self, model):
        self.model = model
        self.tag = model.tag

    def score(self, items: list[Item], keeps=KEEPS) -> dict:
        return {k: self.model.predict_proba([techs_at(it.rec, k) for it in items]) for k in keeps}

    def extract(self, item: Item) -> dict:
        """The model's actual input at 100%: the technique set (only techniques in the vocabulary)."""
        return {t: 1 for t in sorted(techs_at(item.rec, 1.0)) if t in self.model.index}


class TTPMatchScorer:
    kind = "ttp_match"

    def __init__(self, model, embedder):
        self.model = model
        self.emb = embedder
        self.tag = model.tag

    def _counts(self, items: list[Item], keeps) -> dict:
        wins = {}
        flat = []
        for i, it in enumerate(items):
            for k in keeps:
                ws = self.model.windows(sents_at(it.get_sents(), it.rec, k))
                wins[(i, k)] = (len(flat), len(ws))
                flat += ws
        E = self.emb.encode(flat)
        out = {}
        for (i, k), (a, n) in wins.items():
            idx, sim = self.model.match_windows(E[a:a + n])
            out[(i, k)] = self.model.counts_from_matches(idx, sim)
        return out

    def score(self, items: list[Item], keeps=KEEPS) -> dict:
        cnt = self._counts(items, keeps)
        res = {}
        for k in keeps:
            C = np.stack([cnt[(i, k)] for i in range(len(items))]) if items else \
                np.zeros((0, len(self.model.tech_ids)))
            res[k] = self.model.predict_proba(C) if len(C) else np.zeros((0, self.model.n_classes))
        return res

    def extract(self, item: Item) -> dict:
        c = self._counts([item], [1.0])[(0, 1.0)]
        return {self.model.tech_ids[j]: int(v) for j, v in enumerate(c) if v > 0}


def load_scorers(tags: list[str], device=None, gcfg: dict | None = None) -> dict:
    out = {}
    h_tags = [t for t in tags if t.startswith("ttp_vec_")]
    if h_tags:
        from ..ttp_vec.train import load_models
        hm = load_models()
        for t in h_tags:
            if t not in hm:
                raise SystemExit(f"Model {t} not found in ttp_vec.pkl (available: {sorted(hm)})")
            out[t] = TTPVecScorer(hm[t])
    if "ttp_match" in tags:
        from ..common import load_cfg
        from ..ttp_match.embedder import get_embedder
        from ..ttp_match.train import load_model
        gm = load_model()
        g = gcfg or load_cfg()["ttp_match"]
        emb = get_embedder(g, device=device, settings=getattr(gm, "emb_settings", None) or None)
        if emb.name != gm.embedder_name:
            raise SystemExit(f"The loaded embedding model ({emb.name}) differs from the one used in training ({gm.embedder_name}) - "
                             "retrain: python -m cta_systems.ttp_match.train")
        out["ttp_match"] = TTPMatchScorer(gm, emb)
    unknown = [t for t in tags if t not in out]
    if unknown:
        raise SystemExit(f"Invalid tags: {unknown}. Use ttp_vec_<knn|svm|rf|dt|nb> or ttp_match")
    return out


def summarize_probs(p: np.ndarray, true_idx: int, target_idx: int | None = None) -> dict:
    """Same format and values as retrace/models.summarize_probs (tested when there are no ties). The
    only difference is a STABLE sort (kind="stable"): RF/KNN/DT often give equal top-1 probabilities,
    and an unstable sort may change the top-1 group when ONLY other probabilities change -> spurious
    "flips". With ThreatMAMBA's sigmoid probabilities (no ties) both give the same result."""
    order = np.argsort(-p, kind="stable")
    rank = int(np.where(order == true_idx)[0][0]) + 1
    row = {"top1": int(order[0]), "p_top1": round(float(p[order[0]]), 5),
           "p_true": round(float(p[true_idx]), 5), "rank_true": rank,
           "top3": [int(i) for i in order[:3]],
           "probs": [round(float(x), 4) for x in p]}
    if target_idx is not None:
        row["p_target"] = round(float(p[target_idx]), 5)
    return row


def extract_same(a: dict, b: dict) -> int:
    return int(Counter(a) == Counter(b))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))
