#!/usr/bin/env python3
"""CPU tests for cta_systems (TTP-Vec and TTP-Match on the ReTrace benchmark).

    python tests/test_cta_systems.py            # ~2-4 min on CPU, no GPU/network/Ollama needed

Uses a FAKE encoder/TTP extractor/embedder (THREATMAMBA_FAKE_ENCODER, CTA_SYSTEMS_FAKE_EMB) and writes
EVERYTHING to a temporary folder (CTA_SYSTEMS_OUT, CTA_SYSTEMS_RETRACE_DIR) -> the real outputs/ is never
touched. Checks:
  1. ATT&CK technique list (531, parents dropped)              6. build_rows == metrics.build_table
  2. summarize_probs == the ReTrace version                      7. cut-offs == models.text_cut
  3. rarity weights == the TTP-Vec authors' formula              8. integration: fake ReTrace run -> training
  4. TTP-Vec: scaler/weights from train only, valid probabilities    -> rescoring -> adaptive -> report
  5. TTP-Match: technique counts, profiles, prior fallback, temperature
This does NOT replace a real run on a GPU machine.
"""
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="cta_systems_test_"))
os.environ.setdefault("THREATMAMBA_FAKE_ENCODER", "1")
os.environ.setdefault("THREATMAMBA_SSM", "simple")
os.environ["CTA_SYSTEMS_FAKE_EMB"] = "1"
os.environ["CTA_SYSTEMS_OUT"] = str(TMP / "out")
os.environ["CTA_SYSTEMS_RETRACE_DIR"] = str(TMP / "adv")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
PASS = []


def ok(name):
    PASS.append(name)
    print(f"[PASS] {name}")


def write_jsonl_gz(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fake_pipeline():
    from retrace.pipeline import AdvPipeline
    return AdvPipeline(fake=True)


# ---------------------------------------------------------------------- fake ReTrace run
def make_fake_h2(docs, G, runs=("rand", "worst")):
    """Generate REAL variants with the ReTrace operators (fake pipeline) + patterned fake 'main' scores."""
    from retrace.common import param_key
    from retrace.config import get_config
    from retrace.operators import get_op
    from retrace.operators.base import Ctx
    from retrace.pipeline import join_sents
    from retrace.validate import check_variant
    from cta_systems.evaluation.zoo import summarize_probs
    pipe = fake_pipeline()
    cfg = get_config()
    ctx = Ctx(pipeline=pipe, cfg=cfg, pools=None, groups=G)
    g2i = {g: i for i, g in enumerate(G)}
    adv = Path(os.environ["CTA_SYSTEMS_RETRACE_DIR"])
    elig = []
    for run in runs:
        V, S = [], []
        for d in docs:
            doc = pipe.make_doc(d)
            ru = next(g for g in G if g != d["group"])
            doc.meta["runner_up"] = ru
            if run == runs[0]:
                elig.append({"doc_id": d["doc_id"], "group": d["group"], "runner_up": ru,
                             "name_in_graph": int(d["doc_id"][-1] in "02468")})
            rng = random.Random(int(hashlib.sha1(d["doc_id"].encode()).hexdigest()[:8], 16))   # stable across runs
            ovid = f"{d['doc_id']}|ORIG|-|c0"
            V.append({"vid": ovid, "doc_id": d["doc_id"], "group": d["group"], "op": "ORIG",
                      "text": join_sents(doc.sents), "runner_up": ru, "trace": doc.trace})
            base = np.full(len(G), 0.3 / (len(G) - 1)); base[g2i[d["group"]]] = 0.7
            for k in (0.2, 0.4, 0.6, 0.8, 1.0):
                S.append({"vid": ovid, "model": "main", "keep": k, **summarize_probs(base, g2i[d["group"]])})
            for code, params in (("ID", {}), ("T3", {"direction": "expand", "touch": "no_regex", "k": "all"}),
                                 ("T6", {}), ("E1", {"k": 2}), ("E2", {"k": 1, "mode": "drop"}),
                                 ("E4", {"scope": "true"}), ("E5", {"mode": "swap", "k": 1})):
                op = get_op(code, ctx)
                for v in op.candidates(doc, params, 1, rng):
                    v.strategy = "random" if run == "rand" else "best_of_n"
                    vid = f"{d['doc_id']}|{v.op}|{param_key(v.params)}|c{v.cand}"
                    val = check_variant(join_sents(doc.sents), v, cfg)
                    tr = pipe.build("x", d["group"], v.text)[1]
                    from retrace.pipeline import diff_trace
                    V.append({"run": run, "vid": vid, "doc_id": d["doc_id"], "group": d["group"], "op": v.op,
                              "level": v.level, "expect": v.expect, "params": v.params, "strategy": v.strategy,
                              "cand": v.cand, "valid": val, "trace": diff_trace(doc.trace, tr),
                              "meta": dict(v.meta or {}), "text": v.text, "runner_up": ru})
                    p = base.copy()
                    if code != "ID" and rng.random() < 0.3:
                        p = np.roll(p, 1)
                    for k in (0.2, 0.4, 0.6, 0.8, 1.0):
                        S.append({"vid": vid, "model": "main", "keep": k, **summarize_probs(p, g2i[d["group"]])})
        write_jsonl_gz(adv / "runs" / run / "variants.jsonl.gz", V)
        write_jsonl_gz(adv / "runs" / run / "scores.jsonl.gz", S)
    with open(adv / "eligible_v0.jsonl", "w", encoding="utf-8") as f:
        for e in elig:
            f.write(json.dumps(e) + "\n")
    return pipe


def main():
    from cta_systems.common import groups, load_cfg, load_docs, tie_aware_rank, fit_temperature, softmax_T
    G = groups()

    # 1. technique list
    from cta_systems.ttp_match.attack import load_catalogue
    cat = load_catalogue()
    ids = [c["id"] for c in cat]
    assert len(ids) == len(set(ids)) == 531, len(ids)
    parents = {i.split(".")[0] for i in ids if "." in i}
    assert not (parents & set(ids)), "parent techniques with sub-techniques must be dropped"
    assert all("(Citation" not in c["text"] for c in cat)
    ok("ATT&CK v14: 531 techniques, keyed by ID, parents dropped, citations removed")

    # 2. summarize_probs as in ReTrace
    from cta_systems.evaluation.zoo import summarize_probs
    from retrace.models import summarize_probs as sp0
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(20))
        t, g = int(rng.integers(20)), int(rng.integers(20))
        assert summarize_probs(p, t, g) == sp0(p, t, g) and summarize_probs(p, t) == sp0(p, t)
    ok("summarize_probs matches retrace/models.summarize_probs")

    # 3. rarity weights = the authors' formula (get_amplified_rarity_weights)
    import pandas as pd
    from cta_systems.ttp_vec.model import TTPVecModel, rarity_weights
    X = (rng.random((60, 12)) < 0.25).astype(float); X[:, 5] = 0
    y = rng.integers(0, 6, 60)
    df = pd.DataFrame(X, columns=[f"ttps_T{i}" for i in range(12)]); df["apt_num"] = y
    usage = {c: df[df[c] == 1]["apt_num"].nunique() for c in df.columns if c.startswith("ttps_")}
    mx = max(usage.values())
    ref = np.array([((1 / c) ** 3) / ((1 / mx) ** 3) if c > 0 else 0 for c in usage.values()])
    assert np.allclose(rarity_weights(X, y, 3), ref)
    ok("rarity weights (1/number of groups)^3 match the authors' code")

    # 4. TTP-Vec: learns from train only, valid probabilities, unknown techniques ignored
    ts = [{f"T{j}" for j in np.nonzero(X[i])[0]} for i in range(60)]
    for algo in ("knn", "svm", "rf", "dt", "nb"):
        m = TTPVecModel(algo, "rarity3", n_classes=20).fit(ts[:40], y[:40])
        P = m.predict_proba(ts[40:] + [{"T999.999"}])
        assert P.shape == (21, 20) and np.allclose(P.sum(1), 1)
        assert np.allclose(m.scaler.mean_, m.featurize(ts[:40]).mean(0)), "the scaler must be fitted on train"
    ok("TTP-Vec: 5 algorithms, scaler/weights from train only, probabilities over all 20 classes")

    # 5. TTP-Match
    from cta_systems.ttp_match.model import TTPMatchModel
    E = np.eye(6, dtype=np.float32)
    gm = TTPMatchModel([f"T{i}" for i in range(6)], E, window=3)
    idx, sim = gm.match_windows(np.array([[0.9, 0.1, 0, 0, 0, 0], [0, 0, 0.2, 0.8, 0, 0]], dtype=np.float32))
    assert idx[0, 0] == 0 and idx[1, 0] == 3 and sim[1, 0] > sim[1, 1]
    c1 = gm.counts_from_matches(idx, sim, topk=1, thr=None)
    assert c1.tolist() == [1, 0, 0, 1, 0, 0]
    c2 = gm.counts_from_matches(idx, sim, topk=2, thr=0.5)
    assert c2.tolist() == [1, 0, 0, 1, 0, 0]
    counts = np.array([[3, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 4, 1, 0, 0], [1, 0, 0, 0, 0, 0]], float)
    gm.fit(counts, np.array([0, 1, 2, 0]), n_classes=3, prior_mode="expert")
    assert np.allclose(gm.profiles.sum(1), 1)
    L0 = gm.log_scores(np.zeros((1, 6)))
    assert int(np.argmax(L0[0])) == 0, "no technique matched -> ranked by prior (group with the most documents)"
    assert tie_aware_rank(np.array([1.0, 1.0, 0.5]), 1) == 1.5
    Ls = rng.normal(size=(3000, 5)) * 2
    Py = softmax_T(Ls, 3.0)
    yy = np.array([rng.choice(5, p=p) for p in Py])
    That = fit_temperature(Ls, yy)
    assert 2.4 < That < 3.8, That
    ok(f"TTP-Match: top-k/threshold matching, normalised profiles, prior fallback, tie-aware rank, temperature (T^={That:.2f})")

    # 6 + 7. fake ReTrace run -> build_rows == build_table, cut-offs == text_cut
    docs_te = [d for d in load_docs("test")][:40]
    pick = []
    seen = {}
    for d in docs_te:
        if seen.get(d["group"], 0) < 2:
            pick.append(d); seen[d["group"]] = seen.get(d["group"], 0) + 1
        if len(pick) >= 8:
            break
    pipe = make_fake_h2(pick, G)
    import retrace.metrics as M
    from cta_systems.evaluation.rows import build_rows, load_scores, load_variants
    adv_runs = Path(os.environ["CTA_SYSTEMS_RETRACE_DIR"]) / "runs"
    M.run_dir = lambda name: adv_runs / name
    ref_rows = M.build_table("rand", tau=0.5)
    mine = build_rows(load_variants(adv_runs / "rand"), load_scores(adv_runs / "rand" / "scores.jsonl.gz"), tau=0.5)
    assert len(ref_rows) == len(mine) > 0
    for a, b in zip(ref_rows, mine):
        assert all(a[k] == b[k] for k in a), (a, b)
    ok(f"build_rows matches metrics.build_table ({len(mine)} rows)")

    from cta_systems.prepare import cut_sent, rec_from_graph, sents_at
    from retrace.models import text_cut
    from retrace.pipeline import join_sents
    from src.utils import sent_split
    for d in pick[:4]:
        doc = pipe.make_doc(d)
        rec = rec_from_graph(doc.graph)
        for k in (0.2, 0.4, 0.6, 0.8, 1.0):
            assert join_sents(sents_at(sent_split(doc.text), rec, k)) == text_cut(sent_split(doc.text), doc.graph, k)
            assert cut_sent(doc.graph, k) == rec["cut"][[0.2, 0.4, 0.6, 0.8, 1.0].index(k)]
    ok("timeline cut-offs match retrace/models.text_cut")

    from cta_systems.prepare import Preparer, techs_at
    p2 = Preparer(device="cpu", path=TMP / "prep_fast_check.jsonl.gz")
    for d in pick:
        text = join_sents(sent_split(d["text"]))
        fast = p2.techs_fast(text)
        assert fast == techs_at(p2.get(text), 1.0), (fast, techs_at(p2.get(text), 1.0))
    ok("fast path techs_fast (no IOC extraction) matches the graph's technique set")

    # 8. integration
    cfg = load_cfg()
    cfg["ttp_vec"]["cv_folds"] = 3
    tr = [d for d in load_docs("train")]
    random.Random(0).shuffle(tr)
    from cta_systems.ttp_vec.train import train as h_train
    from cta_systems.ttp_match.train import train as g_train
    h_train(cfg, tr[:300], docs_te[:60], device="cpu", do_cv=True)
    g_train(cfg, tr[:300], docs_te[:60], device="cpu", tune=True)
    ok("TTP-Vec training (5 algorithms + CV) and TTP-Match training (tuning) on a subset")

    from cta_systems.evaluation.rescore import default_tags, rescore_run
    from cta_systems.evaluation.zoo import load_scorers
    from cta_systems.prepare import Preparer
    tags = default_tags(cfg)
    scorers = load_scorers(tags, device="cpu", gcfg=cfg["ttp_match"])
    prep = Preparer(device="cpu")
    for run in ("rand", "worst"):
        st = rescore_run(run, scorers, prep, cfg, batch=16)
        assert st["n_variants"] > 0 and st["state_match"] == st["state_check"], st
    out_runs = Path(os.environ["CTA_SYSTEMS_OUT"]) / "runs"
    sc = load_scores(out_runs / "rand" / "scores.jsonl.gz")
    n_var = sum(1 for _ in load_variants(adv_runs / "rand"))
    assert len(sc) == n_var * len(tags) * 5, (len(sc), n_var)
    st2 = rescore_run("rand", scorers, prep, cfg, batch=16)          # re-run: nothing is scored twice
    assert len(load_scores(out_runs / "rand" / "scores.jsonl.gz")) == len(sc) and st2["n_scored"] == 0
    ok(f"rescore: {n_var} variants x {len(tags)} models x 5 cut-offs; re-run without duplicates; graphs match the original run")

    # retraining TTP-Vec -> the old ttp_vec_* scores are dropped and rescored, no duplicate rows
    h_train(cfg, tr[300:600], docs_te[:60], device="cpu", do_cv=False)
    scorers = load_scorers(tags, device="cpu", gcfg=cfg["ttp_match"])
    st3 = rescore_run("rand", scorers, prep, cfg, batch=16)
    sc3 = load_scores(out_runs / "rand" / "scores.jsonl.gz")
    assert st3["n_scored"] == n_var and len(sc3) == len(sc), (st3, len(sc3), len(sc))
    ok("model fingerprint: retraining -> old scores dropped and rescored, never mixing two versions")

    # ties: stable top-1 (depends only on the tied probabilities, not on the rest)
    p1 = np.array([0.4, 0.4] + [0.2 / 18] * 18); p2 = p1.copy(); p2[5], p2[6] = p2[6] + 0.01, p2[5] - 0.01
    assert summarize_probs(p1, 0)["top1"] == summarize_probs(p2, 0)["top1"] == 0
    ok("ties at top-1: stable sort, no spurious flips")

    from cta_systems.evaluation.adaptive import AdaptiveRunner
    from retrace.config import get_config
    from retrace.runner import _params_for
    adv_cfg = get_config()
    passes = [("greedy", ["E1", "E2"], _params_for(["E1", "E2"], adv_cfg), 3, "greedy"),
              ("bon", ["E4", "E5"], _params_for(["E4", "E5"], adv_cfg), 3, "best_of_n")]
    for tag in ("ttp_match", "ttp_vec_rf"):
        ar = AdaptiveRunner(tag, scorers[tag], prep, adv_cfg, cfg)
        for d in pick[:2]:
            ar.run_doc(d, passes)
        av = load_variants(out_runs / f"adapt_{tag}")
        asc = load_scores(out_runs / f"adapt_{tag}" / "scores.jsonl.gz")
        assert any(v["op"] == "E1" and v.get("strategy") == "greedy" for v in av.values())
        assert len(asc) == len(av) * 5
    ok("adaptive: greedy-k + best-of-N against the case-study system itself, same format as the ReTrace runs")

    from cta_systems.evaluation.report import build
    for fmt in ("dec", "pct"):
        md, nums = build(cfg, fmt)
        assert " error: " not in md, [line for line in md.splitlines() if " error: " in line][:3]
        for sec in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.", "## 10."):
            assert sec in md, sec
    (Path(os.environ["CTA_SYSTEMS_OUT"]) / "REPORT_TEST.md").write_text(md, encoding="utf-8")
    ok("comparison report has all 10 sections, in both decimal and % formats")

    print(f"\n{len(PASS)} tests PASS. Temporary folder: {TMP}")
    if not os.getenv("KEEP_TMP"):
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
