"""Shared utilities for cta_systems: paths, config, data, clean-accuracy metrics, calibration.

Only READS from the base repository (data/processed, outputs/retrace) and only WRITES to outputs/cta_systems/.
"""
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrace.common import JsonlAppender, iter_jsonl_any, log, sha1  # noqa: F401  (re-exported)
from src.utils import OUTPUTS, PROCESSED, ROOT, load_json, read_jsonl

# ------------------------------------------------------------------ paths (fully separated)
# Environment variables used by the tests only (tests/test_cta_systems.py points them to a temporary
# folder so the real outputs are never touched): CTA_SYSTEMS_OUT, CTA_SYSTEMS_RETRACE_DIR.
OUT = Path(os.getenv("CTA_SYSTEMS_OUT") or (OUTPUTS / "cta_systems"))
MODELS_DIR = OUT / "models"
CACHE_DIR = OUT / "cache"
RUNS_DIR = OUT / "runs"
TABLES_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"
CONFIG_PATH = ROOT / "configs" / "cta_systems.yaml"

ADV_DIR = Path(os.getenv("CTA_SYSTEMS_RETRACE_DIR") or (OUTPUTS / "retrace"))   # ReTrace results (read-only)
ADV_RUNS = ADV_DIR / "runs"

KEEPS = [0.2, 0.4, 0.6, 0.8, 1.0]

DISPLAY = {
    "main": "ThreatMAMBA",
    "no_cl": "ThreatMAMBA (w/o CL)",
    "no_mamba": "ThreatMAMBA (w/o Mamba)",
    "ttp_match": "TTP-Match (Guru et al. 2025)",
    "ttp_vec_knn": "TTP-Vec-KNN (Hussain & Petrova 2026)",
    "ttp_vec_svm": "TTP-Vec-SVM (Hussain & Petrova 2026)",
    "ttp_vec_rf": "TTP-Vec-RF (Hussain & Petrova 2026)",
    "ttp_vec_dt": "TTP-Vec-DT (Hussain & Petrova 2026)",
    "ttp_vec_nb": "TTP-Vec-NB (Hussain & Petrova 2026)",
}
SHORT = {"main": "ThreatMAMBA", "no_cl": "TM-noCL", "no_mamba": "TM-noMamba", "ttp_match": "TTP-Match",
         "ttp_vec_knn": "TTP-Vec-KNN", "ttp_vec_svm": "TTP-Vec-SVM", "ttp_vec_rf": "TTP-Vec-RF",
         "ttp_vec_dt": "TTP-Vec-DT", "ttp_vec_nb": "TTP-Vec-NB"}
MODEL_ORDER = ["main", "no_cl", "no_mamba", "ttp_match", "ttp_vec_knn", "ttp_vec_svm", "ttp_vec_rf",
               "ttp_vec_dt", "ttp_vec_nb"]


def model_sort_key(m: str):
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m)


# ------------------------------------------------------------------ config
DEFAULTS = {
    "seed": 17, "val_frac": 0.15, "keeps": KEEPS, "levels": [0, 1, 2],
    "runs": ["rand", "worst", "llm"], "threatmamba_models": ["main"],
    "ttp_vec": {"algos": ["knn", "svm", "rf", "dt", "nb"], "weighting": "rarity3", "knn_k": 1,
                "svm_kernel": "linear", "svm_max_iter": 1000000, "cv_folds": 10,
                "main_table_models": ["ttp_vec_knn", "ttp_vec_svm", "ttp_vec_rf"]},
    "ttp_match": {"embedder": "BAAI/bge-large-en-v1.5", "pooling": "cls", "max_length": 512,
             "batch_size": 64, "fp16": True, "query_prefix": "", "window": 3, "desc_chars": 1200,
             "smoothing": 1e-6, "default": {"topk": 1, "thr_quantile": None, "prior": "expert"},
             "tune": True, "grid": {"topk": [1, 2, 3, 5], "thr_quantile": [None, 0.5, 0.75],
                                    "prior": ["expert", "uniform"]}},
    "adaptive": {"models": ["ttp_match", "ttp_vec_rf"], "ops_greedy": ["E1", "E2", "E3"],
                 "ops_bon": ["T5", "E4", "E5", "E7", "E8"], "ops_llm": [], "n": 5, "n_llm": 3},
}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path=None) -> dict:
    import yaml
    p = Path(path) if path else CONFIG_PATH
    user = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    bad = [k for k in cfg["keeps"] if float(k) not in KEEPS]
    if bad:
        raise SystemExit(f"configs/cta_systems.yaml: keeps {bad} are not in {KEEPS} (the ReTrace cut-offs)")
    return cfg


def confident_tau() -> float:
    """The 'confident flip' threshold is read DIRECTLY from configs/retrace.yaml (not declared twice)."""
    from retrace.config import get_config
    return float(get_config().get("confident_tau", 0.5))


# ------------------------------------------------------------------ data
def groups() -> list[str]:
    return load_json(PROCESSED / "groups.json")


def load_docs(split: str | None = None) -> list[dict]:
    rows = read_jsonl(PROCESSED / "docs.jsonl")
    return [r for r in rows if split is None or r["split"] == split]


def train_val_split(train_docs: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """STRATIFIED validation split from the training set (groups with few documents keep >= 1 in train)."""
    by = defaultdict(list)
    for d in train_docs:
        by[d["group"]].append(d)
    rng = random.Random(seed)
    tr, va = [], []
    for g in sorted(by):
        ds = sorted(by[g], key=lambda d: d["doc_id"])
        rng.shuffle(ds)
        n_va = int(round(len(ds) * frac))
        n_va = min(n_va, max(0, len(ds) - 1))
        va += ds[:n_va]
        tr += ds[n_va:]
    return tr, va


# ------------------------------------------------------------------ clean-accuracy metrics
def tie_aware_rank(scores: np.ndarray, true_idx: int) -> float:
    """Rank of the true group (1 = best); ties get the average rank (independent of group order -
    fixes the 'ties broken by folder order' bug of the original TTP-Match code)."""
    s = scores[true_idx]
    higher = float(np.sum(scores > s))
    ties = float(np.sum(scores == s)) - 1.0
    return 1.0 + higher + ties / 2.0


def clean_metrics(probs: np.ndarray, y: np.ndarray, n_classes: int, rank_scores: np.ndarray | None = None) -> dict:
    """probs (n, C); y (n,). top-k by probability order; mean rank with tie handling."""
    from sklearn.metrics import f1_score
    rs = probs if rank_scores is None else rank_scores
    order = np.argsort(-probs, axis=1, kind="stable")
    pred = order[:, 0]
    ranks = np.array([tie_aware_rank(rs[i], int(y[i])) for i in range(len(y))])
    out = {"n": int(len(y)),
           "top1": float(np.mean(pred == y)),
           "top3": float(np.mean([y[i] in order[i, :3] for i in range(len(y))])),
           "top5": float(np.mean([y[i] in order[i, :5] for i in range(len(y))])),
           "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0)),
           "mean_rank": float(ranks.mean())}
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            rec.append(float(np.mean(pred[m] == c)))
    out["macro_top1"] = float(np.mean(rec)) if rec else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def prior_only_metrics(y_train: np.ndarray, y_eval: np.ndarray, n_classes: int) -> dict:
    """Reference point: rank groups only by their training frequency (ignores the report content)."""
    prior = np.bincount(y_train, minlength=n_classes).astype(float)
    prior = prior / prior.sum()
    P = np.tile(prior, (len(y_eval), 1))
    return clean_metrics(P, y_eval, n_classes)


# ------------------------------------------------------------------ calibration (temperature)
def softmax_T(logit: np.ndarray, T: float) -> np.ndarray:
    z = logit / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    """Choose T minimising the validation NLL (log grid). Rank order is unchanged -> top-1 predictions
    are unchanged; this only makes the probabilities meaningful (for delta_p, 'confident flip' > 0.5)."""
    best_T, best = 1.0, float("inf")
    for T in np.exp(np.linspace(np.log(0.02), np.log(50.0), 121)):
        p = softmax_T(logit, T)
        nll = -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


# ------------------------------------------------------------------ model fingerprints
def model_fingerprint(tag: str) -> str:
    """sha256 of the model file (+ tag). Stored scores are bound to this fingerprint: after retraining,
    the old scores of that tag are dropped and re-scored (never mixing scores of two model versions)."""
    from retrace.common import sha256_file
    fp = MODELS_DIR / ("ttp_vec.pkl" if tag.startswith("ttp_vec_") else f"{tag}.pkl")
    return (sha256_file(fp)[:16] + ":" + tag) if fp.exists() else "missing:" + tag


def purge_model_rows(path: Path, tag: str) -> int:
    """Remove every row of model `tag` from a jsonl(.gz) file (rewrites the file). Returns the number removed."""
    if not path.exists():
        return 0
    keep, n = [], 0
    for r in iter_jsonl_any(path):
        if r.get("model") == tag:
            n += 1
        else:
            keep.append(r)
    if n:
        import gzip
        tmp = path.with_name(path.name + ".tmp")
        op = gzip.open if str(path).endswith(".gz") else open
        with op(tmp, "wt", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return n


# ------------------------------------------------------------------ io
def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
