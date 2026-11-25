"""Train TTP-Vec (Hussain & Petrova) and evaluate its clean accuracy on CTI2Attacker.

    python -m cta_systems.ttp_vec.train                 # 5 algorithms x (rarity3, none) + 10-fold CV
    python -m cta_systems.ttp_vec.train --no-cv         # skip CV (faster)

Input: the technique set ThreatMAMBA's TTP extractor assigns to each report (read from data/cskg/*.pt;
missing graphs are rebuilt - needs a GPU). Outputs (outputs/cta_systems/):
  models/ttp_vec.pkl                   every model (tag ttp_vec_<algo>[_<weighting>])
  tables/ttp_vec_clean_test.csv        accuracy on the 629 test reports
  tables/ttp_vec_cv_train.csv          the paper's protocol (stratified 10-fold) on train:
                                       'leaky' = as in the original code (scaler + weights computed on
                                       all data before splitting), 'fixed' = computed inside each fold
  tables/ttp_vec_meta.json             feature statistics
"""
import argparse
import time

import numpy as np

from ..common import (MODELS_DIR, TABLES_DIR, clean_metrics, groups, load_cfg, load_docs, load_pickle, log,
                      prior_only_metrics, save_json, save_pickle, write_csv)
from ..prepare import Preparer, techs_at
from .model import TTPVecModel, make_classifier, rarity_weights

MODEL_PATH = MODELS_DIR / "ttp_vec.pkl"


def load_models(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m cta_systems.ttp_vec.train")
    return load_pickle(path)["models"]


def doc_tech_sets(docs: list[dict], prep: Preparer) -> list[set]:
    out = []
    for i, d in enumerate(docs):
        _t, rec = prep.get_doc(d)
        out.append(techs_at(rec, 1.0))
        if (i + 1) % 500 == 0:
            log(f"[ttp_vec] features {i + 1}/{len(docs)} documents (read from data/cskg: {prep.n_cskg}, "
                f"graphs rebuilt: {prep.n_built})")
    prep.flush()
    return out


def cv_paper(ts: list[set], y: np.ndarray, algo: str, weighting: str, folds: int, hcfg: dict, n_classes: int) -> dict:
    """The paper's protocol (StratifiedKFold shuffle, random_state=7) in two variants:
    (a) 'leaky' exactly as the authors' code: scaler + weights computed on ALL data before splitting
    (b) 'fixed': scaler/weights/vocabulary from the training part of each fold only.
    Hand-written loop (instead of cross_val_score) to log progress per fold."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    splits = list(cv.split(np.zeros(len(y)), y))
    mi = int(hcfg.get("svm_max_iter", -1))
    m = TTPVecModel(algo, weighting, n_classes, hcfg["knn_k"], hcfg["svm_kernel"], mi)
    m.vocab = sorted(set().union(*ts)); m.index = {t: i for i, t in enumerate(m.vocab)}
    X = m.featurize(ts)
    Z = StandardScaler().fit_transform(X)
    if weighting.startswith("rarity"):
        Z = Z * rarity_weights(X, y, float(weighting[len("rarity"):] or 1))
    leaky, fixed, n_conv = [], [], 0
    t0 = time.time()
    for i, (tr, va) in enumerate(splits):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = make_classifier(algo, hcfg["knn_k"], hcfg["svm_kernel"], False, mi)   # CV only needs labels
            clf.fit(Z[tr], y[tr])
            leaky.append(float(np.mean(clf.predict(Z[va]) == y[va])))
            mm = TTPVecModel(algo, weighting, n_classes, hcfg["knn_k"], hcfg["svm_kernel"], mi)
            mm.svm_probability = False
            mm.fit([ts[j] for j in tr], y[tr])
            fixed.append(float(np.mean(mm.clf.predict(mm._transform(mm.featurize([ts[j] for j in va]))) == y[va])))
        n_conv += sum(1 for w in caught if issubclass(w.category, ConvergenceWarning)) + mm.n_convergence_warnings
        if algo in ("svm", "rf") or i == len(splits) - 1:
            log(f"[ttp_vec]   CV {algo} fold {i + 1}/{folds}: original {leaky[-1]:.4f} | fixed {fixed[-1]:.4f} "
                f"| {time.time() - t0:.0f}s")
    return {"algo": algo, "weighting": weighting, "folds": folds,
            "leaky_mean": round(float(np.mean(leaky)), 4), "leaky_std": round(float(np.std(leaky)), 4),
            "fixed_mean": round(float(np.mean(fixed)), 4), "fixed_std": round(float(np.std(fixed)), 4),
            "convergence_warnings": n_conv}


def train(cfg: dict, tr_docs: list[dict], te_docs: list[dict], device=None, do_cv: bool = True) -> dict:
    hcfg = cfg["ttp_vec"]
    G = groups(); g2i = {g: i for i, g in enumerate(G)}
    prep = Preparer(device=device)
    t0 = time.time()
    ts_tr = doc_tech_sets(tr_docs, prep)
    ts_te = doc_tech_sets(te_docs, prep)
    prep.save()
    y_tr = np.array([g2i[d["group"]] for d in tr_docs]); y_te = np.array([g2i[d["group"]] for d in te_docs])
    n_tech = [len(t) for t in ts_tr + ts_te]
    meta = {"n_train": len(tr_docs), "n_test": len(te_docs),
            "vocab_train": len(set().union(*ts_tr)),
            "tech_per_doc_mean": round(float(np.mean(n_tech)), 2),
            "tech_per_doc_median": float(np.median(n_tech)),
            "docs_without_tech": int(sum(1 for n in n_tech if n == 0)),
            "feature_source": "ThreatMAMBA TTP extractor (Technique nodes of CSKG)",
            "prepare_seconds": round(time.time() - t0, 1)}
    log(f"[ttp_vec] vocabulary {meta['vocab_train']} techniques | mean {meta['tech_per_doc_mean']} techniques/report | "
        f"{meta['docs_without_tech']} reports without any technique")

    models, rows = {}, []
    for weighting in dict.fromkeys([hcfg["weighting"], "none"]):
        for algo in hcfg["algos"]:
            t1 = time.time()
            m = TTPVecModel(algo, weighting, len(G), hcfg["knn_k"], hcfg["svm_kernel"],
                             int(hcfg.get("svm_max_iter", -1))).fit(ts_tr, y_tr)
            P = m.predict_proba(ts_te)
            met = clean_metrics(P, y_te, len(G))
            rows.append({"model": m.tag, "algo": algo, "weighting": weighting, **met})
            models[m.tag] = m
            log(f"[ttp_vec] {m.tag:<22} top1={met['top1']:.4f} top3={met['top3']:.4f} "
                f"macroF1={met['macro_f1']:.4f} mean rank={met['mean_rank']:.2f} | {time.time() - t1:.0f}s"
                + (f" | WARNING not converged ({m.n_convergence_warnings})" if m.n_convergence_warnings else ""))
    rows.append({"model": "prior_only", "algo": "-", "weighting": "-", **prior_only_metrics(y_tr, y_te, len(G))})
    write_csv(TABLES_DIR / "ttp_vec_clean_test.csv", rows)
    save_pickle({"models": models, "meta": meta, "groups": G}, MODEL_PATH)

    if do_cv:
        cv_rows = []
        log(f"[ttp_vec] {hcfg['cv_folds']}-fold CV with the paper's protocol for {len(hcfg['algos'])} algorithms "
            f"(each fold is fitted twice: original + fixed; SVM takes seconds per fit -> minutes overall, progress is logged)")
        for algo in hcfg["algos"]:
            r = cv_paper(ts_tr, y_tr, algo, hcfg["weighting"], int(hcfg["cv_folds"]), hcfg, len(G))
            cv_rows.append(r)
            log(f"[ttp_vec] CV {hcfg['cv_folds']}-fold {algo}: original (leaky) {r['leaky_mean']:.4f}"
                f"±{r['leaky_std']:.4f} | fixed {r['fixed_mean']:.4f}±{r['fixed_std']:.4f}")
        write_csv(TABLES_DIR / "ttp_vec_cv_train.csv", cv_rows)
        meta["cv"] = cv_rows
    meta["models"] = [m.describe() for m in models.values()]
    save_json(TABLES_DIR / "ttp_vec_meta.json", meta)
    log(f"-> {MODEL_PATH} | {TABLES_DIR}/ttp_vec_clean_test.csv")
    return {"models": models, "rows": rows, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(load_cfg(args.config), load_docs("train"), load_docs("test"), args.device, not args.no_cv)


if __name__ == "__main__":
    main()
