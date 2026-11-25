"""Train TTP-Match (Guru et al.) and evaluate its clean accuracy on CTI2Attacker.

    python -m cta_systems.ttp_match.train                  # tune on validation + evaluate on test
    python -m cta_systems.ttp_match.train --no-tune        # use the paper configuration (top-1, expert prior)

Procedure:
  1. 531 ATT&CK v14 techniques (parent techniques with sub-techniques dropped) -> embed the descriptions
  2. every train/test report -> 3-sentence windows -> embed -> 5 nearest techniques per window
  3. stratified validation split (15%) of train; grid (topk, threshold, prior) -> chosen by MEAN RANK
     on validation (the paper's measure), ties broken by top-1
  4. temperature T chosen on validation; profiles P(t|A) refit on ALL of train; evaluated on the 629 test reports
Outputs: models/ttp_match.pkl, tables/ttp_match_clean_test.csv, tables/ttp_match_tuning_val.csv, tables/ttp_match_meta.json
"""
import argparse
import itertools
import time

import numpy as np

from src.utils import sent_split

from ..common import (MODELS_DIR, TABLES_DIR, clean_metrics, fit_temperature, groups, load_cfg, load_docs,
                      load_pickle, log, prior_only_metrics, save_json, save_pickle, train_val_split, write_csv)
from .attack import load_catalogue
from .embedder import embedder_settings, get_embedder
from .model import TTPMatchModel

MODEL_PATH = MODELS_DIR / "ttp_match.pkl"


def load_model(path=MODEL_PATH) -> TTPMatchModel:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m cta_systems.ttp_match.train")
    return load_pickle(path)["model"]


def doc_matches(model: TTPMatchModel, emb, docs: list[dict], batch_docs: int = 64) -> list[tuple]:
    """Every document -> (idx, sim) of its windows. Embeds batches of several documents to use the GPU efficiently."""
    out = []
    for i in range(0, len(docs), batch_docs):
        chunk = docs[i:i + batch_docs]
        wins = [model.windows(sent_split(d["text"])) for d in chunk]
        flat = [w for ws in wins for w in ws]
        E = emb.encode(flat)
        pos = 0
        for ws in wins:
            out.append(model.match_windows(E[pos:pos + len(ws)]))
            pos += len(ws)
        if (i // batch_docs) % 10 == 9:
            log(f"[ttp_match] technique matching {min(i + batch_docs, len(docs))}/{len(docs)} documents | cache {emb.stats()}")
    return out


def eval_cfg(model: TTPMatchModel, M_fit, y_fit, M_eval, y_eval, n_classes, topk, thr, prior):
    Cf = np.stack([model.counts_from_matches(i, s, topk, thr) for i, s in M_fit])
    Ce = np.stack([model.counts_from_matches(i, s, topk, thr) for i, s in M_eval])
    model.topk, model.thr = topk, thr
    model.fit(Cf, y_fit, n_classes, prior)
    L = model.log_scores(Ce)
    P = np.exp(L - L.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
    return clean_metrics(P, y_eval, n_classes, rank_scores=L), L, Ce


def train(cfg: dict, tr_all: list[dict], te: list[dict], device=None, tune: bool = True) -> dict:
    gcfg = cfg["ttp_match"]
    G = groups(); g2i = {g: i for i, g in enumerate(G)}; C = len(G)
    t0 = time.time()

    settings = embedder_settings(gcfg)
    emb = get_embedder(gcfg, device=device, settings=settings)
    cat = load_catalogue(int(gcfg.get("desc_chars", 1200)))
    tech_emb = emb.encode([t["text"] for t in cat])
    model = TTPMatchModel([t["id"] for t in cat], tech_emb, window=int(gcfg["window"]),
                      query_prefix=gcfg.get("query_prefix", ""), smoothing=float(gcfg["smoothing"]),
                      embedder_name=emb.name, emb_settings=settings)
    log(f"[ttp_match] {len(cat)} ATT&CK v14 techniques (parents with sub-techniques dropped) | embedding model {emb.name}")

    tr, va = train_val_split(tr_all, float(cfg["val_frac"]), int(cfg["seed"]))
    M_tr = doc_matches(model, emb, tr); M_va = doc_matches(model, emb, va); M_te = doc_matches(model, emb, te)
    emb.save()
    y_tr = np.array([g2i[d["group"]] for d in tr]); y_va = np.array([g2i[d["group"]] for d in va])
    y_te = np.array([g2i[d["group"]] for d in te])
    y_all = np.concatenate([y_tr, y_va])

    # threshold from quantiles of the top-1 cosine of training windows (the original 0.65 only suits the OpenAI scale)
    top1 = np.concatenate([s[:, 0] for _i, s in M_tr if len(s)])
    qthr = {q: (None if q is None else float(np.quantile(top1, q))) for q in gcfg["grid"]["thr_quantile"]}
    qthr.setdefault(gcfg["default"]["thr_quantile"],
                    None if gcfg["default"]["thr_quantile"] is None else float(np.quantile(top1, gcfg["default"]["thr_quantile"])))

    # ---- tuning grid on validation
    tune_rows = []
    grid = list(itertools.product(gcfg["grid"]["topk"], gcfg["grid"]["thr_quantile"], gcfg["grid"]["prior"]))
    d0 = gcfg["default"]
    if (d0["topk"], d0["thr_quantile"], d0["prior"]) not in grid:
        grid.append((d0["topk"], d0["thr_quantile"], d0["prior"]))
    for topk, q, prior in grid:
        met, _L, _C = eval_cfg(model, M_tr, y_tr, M_va, y_va, C, topk, qthr[q], prior)
        tune_rows.append({"topk": topk, "thr_quantile": q, "thr": qthr[q], "prior": prior,
                          "paper_default": int((topk, q, prior) == (d0["topk"], d0["thr_quantile"], d0["prior"])),
                          **{f"val_{k}": v for k, v in met.items()}})
    write_csv(TABLES_DIR / "ttp_match_tuning_val.csv", tune_rows)
    if gcfg.get("tune", True) and tune:
        best = min(tune_rows, key=lambda r: (r["val_mean_rank"], -r["val_top1"]))
    else:
        best = next(r for r in tune_rows if r["paper_default"])
    log(f"[ttp_match] selected topk={best['topk']} threshold(q)={best['thr_quantile']} prior={best['prior']} | "
        f"validation: mean rank {best['val_mean_rank']:.2f}, top1 {best['val_top1']:.4f}")

    # ---- temperature T on validation (profiles from the train part), then refit the profiles on all of train
    _met, L_va, _ = eval_cfg(model, M_tr, y_tr, M_va, y_va, C, best["topk"], best["thr"], best["prior"])
    T = fit_temperature(L_va, y_va)
    rows = []

    def final(topk, thr, q, prior, name, T_):
        m, L, _ = eval_cfg(model, M_tr + M_va, y_all, M_te, y_te, C, topk, thr, prior)
        model.thr_quantile = q; model.T = T_
        P = model.predict_proba(_counts(model, M_te))
        met = clean_metrics(P, y_te, C, rank_scores=L)
        rows.append({"model": name, "topk": topk, "thr_quantile": q, "prior": prior, "T": round(T_, 4), **met})
        log(f"[ttp_match] {name:<26} top1={met['top1']:.4f} top3={met['top3']:.4f} macroF1={met['macro_f1']:.4f} "
            f"mean rank={met['mean_rank']:.2f} / {C}")

    d0thr = qthr[d0["thr_quantile"]]
    final(d0["topk"], d0thr, d0["thr_quantile"], d0["prior"], "ttp_match_paper_default", 1.0)
    final(d0["topk"], d0thr, d0["thr_quantile"], "uniform", "ttp_match_paper_uniform_prior", 1.0)
    final(5, None, None, "uniform", "ttp_match_code_everything_py", 1.0)      # everything.py: top-5, no threshold, no prior
    final(best["topk"], best["thr"], best["thr_quantile"], best["prior"], "ttp_match", T)   # configuration used in the robustness tests
    rows.append({"model": "prior_only", **prior_only_metrics(y_all, y_te, C)})
    rows.append({"model": "random (expected)", "n": len(y_te), "top1": round(1 / C, 4), "top3": round(3 / C, 4),
                 "top5": round(5 / C, 4), "mean_rank": (C + 1) / 2})
    write_csv(TABLES_DIR / "ttp_match_clean_test.csv", rows)

    # final model = selected configuration, profiles over all of train (already in `model` after the last final())
    n_win = [len(i) for i, _s in M_tr + M_va + M_te]
    zero = sum(1 for i, s in M_te if model.counts_from_matches(i, s).sum() == 0)
    meta = {"model": model.describe(), "n_train": len(y_all), "n_val_for_tuning": len(y_va), "n_test": len(y_te),
            "windows_per_doc_mean": round(float(np.mean(n_win)), 2), "test_docs_without_match": zero,
            "top1_cos_quantiles": {str(q): round(float(np.quantile(top1, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "seconds": round(time.time() - t0, 1)}
    save_json(TABLES_DIR / "ttp_match_meta.json", meta)
    save_pickle({"model": model, "groups": G, "meta": meta}, MODEL_PATH)
    log(f"-> {MODEL_PATH} | {TABLES_DIR}/ttp_match_clean_test.csv, ttp_match_tuning_val.csv")
    return {"model": model, "rows": rows, "tuning": tune_rows, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(load_cfg(args.config), load_docs("train"), load_docs("test"), args.device, not args.no_tune)


def _counts(model: TTPMatchModel, M) -> np.ndarray:
    return np.stack([model.counts_from_matches(i, s) for i, s in M])


if __name__ == "__main__":
    main()
