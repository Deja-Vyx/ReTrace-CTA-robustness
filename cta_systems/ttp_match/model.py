"""TTP-Match (Guru et al. 2025): embedding-based technique matching + P(technique | group) profiles + prior.

Mapping to the original code (everything.py / everythingHYDE.py):
  get_txt_data        -> make_windows        (3 lines -> 3 sentences, non-overlapping as in the original)
  search(topk, 0.65)  -> match_windows + counts_from_matches (top-k, optional threshold)
  get_term_counts     -> counts_from_matches (counts; normalised at scoring time)
  process_split       -> TTPMatchModel.fit        (P(t|A) = summed counts over group A's training reports, normalised)
  calculate_probabilities -> TTPMatchModel.log_scores (score = P(A) * sum_t P(t|A) * y_t)

Additions (they do not change the ranking when a report matches at least one technique):
  - small smoothing added to the likelihood: a report that matches no technique is ranked by the prior
    instead of "ties broken by folder order" as in the original code
  - temperature T (chosen on validation): p = softmax(log score / T), so delta_p and the 0.5 threshold are meaningful
"""
import numpy as np

from ..common import softmax_T

KMAX = 5            # candidates stored per window (enough for every topk in the tuning grid)


def make_windows(sents: list[str], size: int = 3) -> list[str]:
    return [" ".join(sents[i:i + size]) for i in range(0, len(sents), size)]


class TTPMatchModel:
    def __init__(self, tech_ids: list[str], tech_emb: np.ndarray, window: int = 3, query_prefix: str = "",
                 smoothing: float = 1e-6, embedder_name: str = "?", emb_settings: dict | None = None):
        self.tech_ids = list(tech_ids)
        self.tech_emb = np.asarray(tech_emb, dtype=np.float32)
        self.window = int(window)
        self.query_prefix = query_prefix or ""
        self.smoothing = float(smoothing)
        self.embedder_name = embedder_name
        self.emb_settings = dict(emb_settings or {})   # pooling/max_length/fp16 used at training time
        self.topk = 1
        self.thr = None             # absolute cosine threshold (None = no threshold)
        self.thr_quantile = None
        self.prior_mode = "expert"
        self.profiles = None        # (C, n_tech)
        self.prior = None           # (C,)
        self.T = 1.0
        self.n_classes = 0

    tag = "ttp_match"

    # ------------------------------------------------------------------ technique matching
    def windows(self, sents: list[str]) -> list[str]:
        return [self.query_prefix + w for w in make_windows(sents, self.window)]

    def match_windows(self, win_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (idx, sim) of shape (n_windows, KMAX), sorted by decreasing cosine."""
        if len(win_emb) == 0:
            return np.zeros((0, KMAX), dtype=np.int32), np.zeros((0, KMAX), dtype=np.float32)
        sims = win_emb @ self.tech_emb.T
        k = min(KMAX, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        return (np.take_along_axis(part, order, axis=1).astype(np.int32),
                np.take_along_axis(ps, order, axis=1).astype(np.float32))

    def counts_from_matches(self, idx: np.ndarray, sim: np.ndarray, topk=None, thr="self") -> np.ndarray:
        topk = self.topk if topk is None else topk
        thr = self.thr if thr == "self" else thr
        c = np.zeros(len(self.tech_ids), dtype=np.float64)
        if len(idx) == 0:
            return c
        ii, ss = idx[:, :topk], sim[:, :topk]
        keep = np.ones_like(ss, dtype=bool) if thr is None else (ss >= thr)
        np.add.at(c, ii[keep], 1.0)
        return c

    # ------------------------------------------------------------------ profiles + prior
    def fit(self, counts: np.ndarray, y: np.ndarray, n_classes: int, prior_mode: str = "expert"):
        self.n_classes = n_classes
        prof = np.zeros((n_classes, counts.shape[1]))
        for c in range(n_classes):
            s = counts[y == c].sum(axis=0)
            tot = s.sum()
            prof[c] = s / tot if tot > 0 else 0.0          # the original code divides by 0 when a group has no technique
        self.profiles = prof
        self.prior_mode = prior_mode
        if prior_mode == "uniform":
            self.prior = np.full(n_classes, 1.0 / n_classes)
        else:                                               # "expert": group frequency in train
            f = np.bincount(y, minlength=n_classes).astype(float) + 1e-9
            self.prior = f / f.sum()
        return self

    def log_scores(self, counts: np.ndarray) -> np.ndarray:
        counts = np.atleast_2d(counts)
        tot = counts.sum(axis=1, keepdims=True)
        ybar = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
        lik = ybar @ self.profiles.T
        return np.log(self.prior)[None, :] + np.log(lik + self.smoothing)

    def predict_proba(self, counts: np.ndarray) -> np.ndarray:
        return softmax_T(self.log_scores(counts), self.T)

    def describe(self) -> dict:
        return {"tag": self.tag, "embedder": self.embedder_name, "embedder_settings": self.emb_settings,
                "n_techniques": len(self.tech_ids),
                "window_sents": self.window, "topk": self.topk, "thr": self.thr, "thr_quantile": self.thr_quantile,
                "prior": self.prior_mode, "T": self.T, "smoothing": self.smoothing}
