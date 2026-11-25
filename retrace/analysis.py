"""Root-cause and shortcut analyses. EVERY conclusion is COMPUTED FROM THE DATA with 95% bootstrap
confidence intervals over documents - no hard-coded conclusions.

    python -m retrace.analysis all          --run rand    # every analysis below for one run
    python -m retrace.analysis size-probe   --n 2000      # S1 graph size in contrastive learning
    python -m retrace.analysis ttp-fp                     # S2 TTP extractor labels benign sentences
    python -m retrace.analysis name-leak    --run rand    # S3 the model "reads the group name"
    python -m retrace.analysis order        --run rand    # S4 order sensitivity (Mamba branch)
    python -m retrace.analysis sink         --run rand    # S5 groups that "attract" wrong predictions
    python -m retrace.analysis specificity  --run rand    # S6 common vs specific TTPs
    python -m retrace.analysis stage-decomp --run rand    # errors in the extraction or classification stage
    python -m retrace.analysis by-group     --run rand    # robustness per group + unweighted mean over groups
    python -m retrace.analysis curves       --run rand    # flip over k and over the timeline
    python -m retrace.analysis figures      --run rand
    python -m retrace.analysis latex        --run rand

Verdicts: "supported" (95% CI entirely on the effect side AND effect >= 0.01) | "weak" (only ONE of
the two conditions) | "not_supported" (effect < 0.01 AND CI touches 0) | "na" (not enough data).
"""
import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from src.utils import PROCESSED, load_config, load_json
from .common import ADV_DIR, iter_jsonl_any, log
from .config import run_dir
from .metrics import aggregate, bootstrap_ci, build_table, run_exists, write_csv

KEEP_FULL = 1.0
MIN_DOCS = 5


# ---------------------------------------------------------------------- helpers
MIN_EFFECT = 0.01     # probability effects below 1 percentage point are considered negligible


def verdict(pt, lo, thr=0.0, n_docs=None, min_effect=MIN_EFFECT):
    """Two conditions: (a) the 95% CI lies entirely above thr (lo > thr), (b) the effect is large
    enough (pt >= thr + min_effect). supported = both | weak = only one | not_supported = neither
    (e.g. an effect of 0.001 with a CI touching 0) | na = fewer than MIN_DOCS documents."""
    if pt is None or (n_docs is not None and n_docs < MIN_DOCS):
        return "na"
    sig = lo is not None and lo > thr
    big = pt >= thr + min_effect
    if sig and big:
        return "supported"
    if sig or big:
        return "weak"
    return "not_supported"


def _per_doc(rows, metric):
    d = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None:
            d[r["doc_id"]].append(v)
    return d


def ci_of(rows, metric):
    pd = _per_doc(rows, metric)
    pt, lo, hi = bootstrap_ci(pd)
    return {"mean": pt, "lo": lo, "hi": hi, "n_docs": len(pd), "n": sum(len(v) for v in pd.values())}


def paired_diff(rows_a, rows_b, metric):
    """Paired difference (A - B) by document, with a document-level bootstrap CI."""
    a, b = _per_doc(rows_a, metric), _per_doc(rows_b, metric)
    docs = sorted(set(a) & set(b))
    diff = {d: [float(np.mean(a[d]) - np.mean(b[d]))] for d in docs}
    pt, lo, hi = bootstrap_ci(diff)
    return {"mean": pt, "lo": lo, "hi": hi, "n_docs": len(docs)}


def _fmt(c):
    if not c or c.get("mean") is None:
        return "-"
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] (n_doc={c['n_docs']})"


def _rows(run, **flt):
    rows = [r for r in build_table(run) if r["valid"]]
    for k, v in flt.items():
        if callable(v):
            rows = [r for r in rows if v(r[k])]
        else:
            rows = [r for r in rows if r[k] == v]
    return rows


def _models(run):
    return sorted({r["model"] for r in build_table(run)})


def _elig():
    return {r["doc_id"]: r for r in iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl")}


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _save_rows(run, name, rows):
    if rows:
        out = run_dir(run) / "tables"; out.mkdir(exist_ok=True)
        write_csv(out / name, rows)


# ---------------------------------------------------------------------- S1: graph size in CL
def compute_size_probe(n=2000, config=None):
    """Generate contrastive negatives in two ways (old splice / matched Eq. 8) and train a logistic
    probe that separates negatives from positives using ONLY the node count. A high probe accuracy
    -> contrastive learning can be solved by counting nodes (shortcut)."""
    import random
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from src.cskg_builder import CSKG_DIR
    from src.contrastive import neg_splice, neg_splice_matched
    from src.utils import ATTCK_DIR, read_jsonl, torch_load
    docs = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == "train"]
    graphs = []
    for d in docs:
        fp = CSKG_DIR / f"{d['doc_id']}.pt"
        if fp.exists():
            graphs.append(torch_load(fp))
        if len(graphs) >= n:
            break
    if len(graphs) < 20:
        return {"verdict": "na", "rows": [], "note": "too few training graphs"}
    attck = load_json(ATTCK_DIR / "attck_v14.json")
    tac_order = [t["shortname"] for t in attck["tactic_order"]]
    labels = [g["group"] for g in graphs]

    def eval_mode(splice_fn, name, seed=17):
        rng = random.Random(seed)
        pos, neg = [], []
        for i, g in enumerate(graphs):
            pos.append(len(g["node_types"]))
            cands = [j for j in range(len(graphs)) if labels[j] != labels[i]] or \
                [j for j in range(len(graphs)) if j != i]
            neg.append(len(splice_fn(g, graphs[rng.choice(cands)], tac_order, rng=rng)["node_types"]))
        X = np.log1p(np.array(pos + neg, dtype=float)).reshape(-1, 1)
        y = np.array([1] * len(pos) + [0] * len(neg))
        acc = float(np.mean(cross_val_score(LogisticRegression(), X, y, cv=5)))
        return {"mode": name, "probe_acc_cv5": round(acc, 4),
                "size_ratio": round(float(np.mean(neg) / max(1, np.mean(pos))), 3),
                "pos_nodes": round(float(np.mean(pos)), 1), "neg_nodes": round(float(np.mean(neg)), 1)}

    rows = [eval_mode(neg_splice, "splice(old)"), eval_mode(neg_splice_matched, "matched(Eq.8)")]
    old, new = rows[0]["probe_acc_cv5"], rows[1]["probe_acc_cv5"]
    v = "supported" if old >= 0.70 and new <= 0.60 else ("weak" if old - new >= 0.10 else "not_supported")
    return {"verdict": v, "rows": rows, "n_graphs": len(graphs),
            "summary": f"node-count-only probe: old splice {old:.3f} vs matched {new:.3f} (0.5 = chance)"}


def cmd_size_probe(args):
    res = compute_size_probe(args.n, args.config)
    out = ADV_DIR / "analysis"; out.mkdir(parents=True, exist_ok=True)
    if res["rows"]:
        write_csv(out / "size_probe.csv", res["rows"])
    _save_json(out / "size_probe.json", res)
    print(f"[S1 size-probe] {res.get('n_graphs', 0)} training graphs")
    for r in res["rows"]:
        print(f"  {r['mode']:<14} probe acc (CV5)={r['probe_acc_cv5']:.3f} | negative/positive size="
              f"{r['size_ratio']:.2f} (positive~{r['pos_nodes']:.0f} nodes, negative~{r['neg_nodes']:.0f})")
    print(f"=> verdict: {res['verdict']} - {res.get('summary', res.get('note', ''))}")


# ---------------------------------------------------------------------- S2: TTP false positives
def compute_ttp_fp():
    from .common import POOLS_DIR
    out = {"verdict": "na"}
    fp = POOLS_DIR / "distractors.json"
    if fp.exists():
        dis = json.load(open(fp, encoding="utf-8"))
        if isinstance(dis, dict) and "stats" in dis:
            st = dis["stats"]
            out.update({"n_benign": st["n_benign"], "n_ttp_free": st["n_ttp_free"],
                        "fp_rate": st["ttp_fp_rate"], "top_labels": st.get("top_labels", [])})
            k = st["n_benign"] - st["n_ttp_free"]
            n = max(1, st["n_benign"])
            # Wilson CI for a proportion
            p = k / n; z = 1.96
            den = 1 + z * z / n
            c = (p + z * z / (2 * n)) / den
            h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            out["fp_ci"] = [round(max(0.0, float(c - h)), 4), round(min(1.0, float(c + h)), 4)]
            out["verdict"] = "supported" if c - h > 0.2 else ("weak" if p > 0.2 else "not_supported")
    fc = POOLS_DIR / "carriers.json"
    if fc.exists():
        car = json.load(open(fc, encoding="utf-8"))
        tt = car.get("_ttp", {})
        out["carrier_with_ttp"] = f"{sum(1 for v in tt.values() if v)}/{len(tt)}"
    return out


def cmd_ttp_fp(args):
    res = compute_ttp_fp()
    _save_json(ADV_DIR / "analysis" / "ttp_fp.json", res)
    if res.get("n_benign") is None:
        print("[S2 ttp-fp] pools/distractors.json (new format) not found - run `pools build` first")
        return
    print(f"[S2 ttp-fp] the TTP extractor assigns >=1 label to {res['n_benign']-res['n_ttp_free']}/{res['n_benign']} "
          f"benign sentences = {res['fp_rate']:.2f} (CI {res['fp_ci']}) | IOC carrier templates with TTP labels: "
          f"{res.get('carrier_with_ttp', '-')}")
    print(f"  most frequent labels: {res['top_labels'][:6]}")
    print(f"=> verdict: {res['verdict']}")


# ---------------------------------------------------------------------- S3: reading the group name
def compute_name_leak(run):
    elig = _elig()
    res = {"models": {}}
    for m in _models(run):
        e4 = _rows(run, op="E4", model=m, keep=KEEP_FULL)
        e4t = [r for r in e4 if "scope=true" in r["params"].lower()]
        e4a = [r for r in e4 if "scope=all" in r["params"].lower()]
        ing = [r for r in e4t if elig.get(r["doc_id"], {}).get("name_in_graph")]
        nig = [r for r in e4t if not elig.get(r["doc_id"], {}).get("name_in_graph")]
        e8 = [r for r in _rows(run, op="E8", model=m, keep=KEEP_FULL) if "kind=name" in r["params"]]
        for r in e8:
            r["p_target_gain"] = (None if r["p_target"] is None or r["p_target_orig"] is None
                                  else r["p_target"] - r["p_target_orig"])
        res["models"][m] = {
            "E4_true_dp_name_in_graph": ci_of(ing, "delta_p"),
            "E4_true_flip_name_in_graph": ci_of(ing, "flip"),
            "E4_true_dp_name_not_in_graph": ci_of(nig, "delta_p"),
            "E4_true_flip_name_not_in_graph": ci_of(nig, "flip"),
            "E4_all_dp": ci_of(e4a, "delta_p"),
            "E8_name_asr": ci_of(e8, "target_hit"),
            "E8_name_p_target_gain": ci_of(e8, "p_target_gain"),
        }
    mm = res["models"].get("main") or next(iter(res["models"].values()), None)
    if mm:
        a = mm["E4_true_dp_name_in_graph"]; b = mm["E8_name_p_target_gain"]
        va, vb = verdict(a["mean"], a["lo"], 0.0, a["n_docs"]), verdict(b["mean"], b["lo"], 0.0, b["n_docs"])
        # two DIFFERENT questions -> two separate verdicts:
        #  S3a: does the model RELY on the true group name (mask the name -> does p drop?)
        #  S3b: does ANOTHER group's name pull the prediction (insert B's name -> does p(B) rise?)
        res["verdict_mask"], res["verdict_inject"] = va, vb
        order = ["na", "not_supported", "weak", "supported"]
        res["verdict"] = max(va, vb, key=order.index)
        res["summary"] = (f"S3a masking the true name -> p(true group) drops by {_fmt(a)} [{va}]; "
                          f"S3b inserting B's name -> p(B) rises by {_fmt(b)} [{vb}]")
    else:
        res["verdict"] = "na"
    return res


def cmd_name_leak(args):
    res = compute_name_leak(args.run)
    print(f"[S3 name-leak] run={args.run} @keep=1.0")
    for m, d in res["models"].items():
        print(f"  {m}:")
        print(f"    E4 mask true group name | name in an evidence sentence: delta_p {_fmt(d['E4_true_dp_name_in_graph'])}"
              f" | flip {_fmt(d['E4_true_flip_name_in_graph'])}")
        print(f"    E4 mask true group name | name NOT in an evidence sentence: delta_p "
              f"{_fmt(d['E4_true_dp_name_not_in_graph'])}")
        print(f"    E4 mask all 20 group names: delta_p {_fmt(d['E4_all_dp'])}")
        print(f"    E8 insert B's name: rate steered to B {_fmt(d['E8_name_asr'])} | p(B) gain "
              f"{_fmt(d['E8_name_p_target_gain'])}")
    print(f"=> verdict: {res['verdict']} - {res.get('summary', '')}")
    _save_json(run_dir(args.run) / "analysis" / "name_leak.json", res)


# ---------------------------------------------------------------------- S4: order
def compute_order(run):
    # G3 reverses States, E5 swaps sentences, T2 splits/merges sentences (LLM) - all change the State sequence Mamba reads
    rows = [r for r in _rows(run, keep=KEEP_FULL)
            if r["op"] in ("G3", "T2") or (r["op"] == "E5" and "mode=swap" in r["params"])]
    models = _models(run)
    out = {"rows": [], "verdict": "na"}
    keys = sorted({(r["op"], r["params"]) for r in rows})
    for op, pk in keys:
        sub = [r for r in rows if r["op"] == op and r["params"] == pk]
        rec = {"op": op, "params": pk}
        for m in models:
            c = ci_of([r for r in sub if r["model"] == m], "flip")
            rec[f"flip_{m}"] = c["mean"]; rec[f"dp_{m}"] = ci_of([r for r in sub if r["model"] == m], "delta_p")["mean"]
            rec["n_docs"] = c["n_docs"]
        if "main" in models and "no_mamba" in models:
            d = paired_diff([r for r in sub if r["model"] == "main"],
                            [r for r in sub if r["model"] == "no_mamba"], "delta_p")
            rec["dp_main_minus_no_mamba"] = d["mean"]; rec["dp_diff_lo"] = d["lo"]; rec["dp_diff_hi"] = d["hi"]
        out["rows"].append(rec)
    g3r = [r for r in out["rows"] if r["op"] == "G3" and "reverse" in r["params"]]
    if not g3r:      # run without G3 (e.g. the LLM run) -> use the largest-k T2 as the main check
        t2 = sorted([r for r in out["rows"] if r["op"] == "T2"], key=lambda r: r["params"])
        g3r = t2[-1:]
    if g3r and "dp_main_minus_no_mamba" in g3r[0]:
        r = g3r[0]
        out["verdict"] = verdict(r["dp_main_minus_no_mamba"], r["dp_diff_lo"], 0.0, r["n_docs"])
        out["primary"] = r
        out["summary"] = (f"{r['op']} {r['params']}: delta_p main - no_mamba = "
                          f"{r['dp_main_minus_no_mamba']:.3f} [{r['dp_diff_lo']:.3f}, {r['dp_diff_hi']:.3f}]; "
                          f"flip main={r.get('flip_main')}, no_mamba={r.get('flip_no_mamba')}")
    return out


def cmd_order(args):
    res = compute_order(args.run)
    print(f"[S4 order] run={args.run} @keep=1.0 - order perturbations (G3, E5 swap)")
    for r in res["rows"]:
        fl = " ".join(f"{k[5:]}={v:.3f}" for k, v in r.items() if k.startswith("flip_") and v is not None)
        extra = (f" | delta_p main-no_mamba={r['dp_main_minus_no_mamba']:.3f} "
                 f"[{r['dp_diff_lo']:.3f},{r['dp_diff_hi']:.3f}]" if r.get("dp_main_minus_no_mamba") is not None else "")
        print(f"  {r['op']} {r['params']:<22} flip: {fl}{extra}")
    print(f"=> verdict: {res['verdict']} - {res.get('summary', '')}")
    _save_rows(args.run, "order.csv", res["rows"])
    _save_json(run_dir(args.run) / "analysis" / "order.json", res)


# ---------------------------------------------------------------------- S5: attractor groups
def compute_sink(run, model="main"):
    groups = load_json(PROCESSED / "groups.json")
    elig = _elig()
    prev = Counter(r["group"] for r in elig.values())
    rows = [r for r in _rows(run, keep=KEEP_FULL, model=model) if r["op"] != "ID" and r["flip"]]
    # the new top-1 is needed -> read from scores (rows_flat has no top1) -> reload
    from .metrics import load_run
    _d, _v, scores = load_run(run)
    top1 = {(s["vid"], s["model"], s["keep"]): s["top1"] for s in scores}
    dest = Counter()
    for r in rows:
        t = top1.get((r["vid"], model, KEEP_FULL))
        if t is not None:
            dest[groups[t]] += 1
    total = sum(dest.values())
    out = {"model": model, "n_flips": total, "rows": [], "verdict": "na"}
    n_el = max(1, sum(prev.values()))
    for g, c in dest.most_common(10):
        share = c / total
        base = prev.get(g, 0) / n_el
        out["rows"].append({"group_to": g, "n_flips_into": c, "share": round(share, 4),
                            "share_in_eligible": round(base, 4),
                            "ratio": round(share / base, 2) if base > 0 else None})
    if total >= 20 and out["rows"]:
        # "sink" group = the group that receives >=15% of the flips and most exceeds its share of the eligible set
        cand = [r for r in out["rows"] if r["share"] >= 0.15] or out["rows"][:1]
        top = max(cand, key=lambda r: (r["ratio"] if r["ratio"] is not None else 99.0))
        ratio = top["ratio"] if top["ratio"] is not None else 99.0
        out["attractor"] = top
        out["verdict"] = "supported" if top["share"] >= 0.20 and ratio >= 5 else (
            "weak" if top["share"] >= 0.15 and ratio >= 2 else "not_supported")
        out["summary"] = (f"{top['share']*100:.0f}% of flips go to {top['group_to']} "
                          f"(this group is {top['share_in_eligible']*100:.0f}% of the eligible set, {ratio}x)")
    elif total < 20:
        out["summary"] = f"only {total} flips - not enough to conclude"
    return out


def cmd_sink(args):
    res = compute_sink(args.run)
    print(f"[S5 sink] run={args.run} model=main @keep=1.0: {res['n_flips']} flips")
    for r in res["rows"]:
        print(f"  -> {r['group_to']:<18} {r['n_flips_into']:>5} ({r['share']*100:.1f}%) | "
              f"{r['share_in_eligible']*100:.1f}% of eligible | ratio {r['ratio']}")
    print(f"=> verdict: {res['verdict']} - {res.get('summary', '')}")
    _save_rows(args.run, "sink.csv", res["rows"])
    _save_json(run_dir(args.run) / "analysis" / "sink.json", res)


# ---------------------------------------------------------------------- S6: TTP specificity
def compute_specificity(run, model="main"):
    from src.utils import ATTCK_DIR
    attck = load_json(ATTCK_DIR / "attck_v14.json")
    df = Counter()
    for _n, info in attck["groups"].items():
        for t in set(x.split(".")[0] for x in info.get("techniques", [])):
            df[t] += 1
    variants = {v["vid"]: v for v in iter_jsonl_any(run_dir(run) / "variants.jsonl.gz")}
    rows = []
    for r in _rows(run, op="E2", model=model, keep=KEEP_FULL):
        techs = ((variants.get(r["vid"]) or {}).get("meta") or {}).get("items", [])
        if not techs:
            continue
        r = dict(r)
        r["tech_df"] = float(np.mean([df.get(str(t).split(".")[0], 0) for t in techs]))
        rows.append(r)
    out = {"verdict": "na", "rows": []}
    if len(rows) < 10:
        out["summary"] = "not enough E2 variants"
        return out
    med = float(np.median([r["tech_df"] for r in rows]))
    common = [r for r in rows if r["tech_df"] > med]
    spec = [r for r in rows if r["tech_df"] <= med]
    cc, cs = ci_of(common, "delta_p"), ci_of(spec, "delta_p")
    out["rows"] = [{"bucket": f"common (>{med:.0f} groups use it)", **cc},
                   {"bucket": f"specific (<={med:.0f} groups use it)", **cs}]
    # (specific - common) difference is unpaired -> two-sample bootstrap
    rng = np.random.default_rng(17)
    a = np.array([np.mean(v) for v in _per_doc(spec, "delta_p").values()])
    b = np.array([np.mean(v) for v in _per_doc(common, "delta_p").values()])
    if len(a) >= 2 and len(b) >= 2:
        boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        d = float(a.mean() - b.mean())
        out["diff_spec_minus_common"] = [round(d, 4), round(float(lo), 4), round(float(hi), 4)]
        # "supported" = the model relies on SPECIFIC techniques (discriminative behaviour) more than on common ones
        out["verdict"] = verdict(d, float(lo), 0.0, min(len(a), len(b)))
        out["summary"] = (f"masking specific techniques lowers p(true group) by {cs['mean']:.3f}, common ones by {cc['mean']:.3f}; "
                          f"difference {d:.3f} [{lo:.3f}, {hi:.3f}]")
    return out


def cmd_specificity(args):
    res = compute_specificity(args.run)
    print(f"[S6 specificity] run={args.run} E2 (drop TTP sentences) main @keep=1.0 - delta_p by technique prevalence")
    for r in res["rows"]:
        print(f"  {r['bucket']:<28} delta_p={r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] (n_doc={r['n_docs']})")
    print(f"=> verdict: {res['verdict']} - {res.get('summary', '')}")
    _save_rows(args.run, "specificity.csv", res["rows"])
    _save_json(run_dir(args.run) / "analysis" / "specificity.json", res)


# ---------------------------------------------------------------------- failing stage
def compute_stage_decomp(run, model="main"):
    variants = {v["vid"]: v for v in iter_jsonl_any(run_dir(run) / "variants.jsonl.gz")}
    by = defaultdict(list)
    for r in _rows(run, model=model, keep=KEEP_FULL):
        tr = (variants.get(r["vid"]) or {}).get("trace")
        if not tr:
            continue
        same = (tr.get("ioc_jacc", 1) == 1 and tr.get("tech_jacc", 1) == 1 and tr.get("d_states", 0) == 0)
        by[r["op"]].append({**r, "extract_same": int(same), "ioc_jacc": tr.get("ioc_jacc", 1),
                            "tech_jacc": tr.get("tech_jacc", 1), "d_states": tr.get("d_states", 0)})
    rows = []
    for op, rs in sorted(by.items()):
        flips = [r for r in rs if r["flip"]]
        rows.append({"op": op, "level": rs[0]["level"], "n": len(rs),
                     "flip_rate": round(float(np.mean([r["flip"] for r in rs])), 4),
                     "ioc_jacc": round(float(np.mean([r["ioc_jacc"] for r in rs])), 3),
                     "tech_jacc": round(float(np.mean([r["tech_jacc"] for r in rs])), 3),
                     "d_states": round(float(np.mean([r["d_states"] for r in rs])), 2),
                     "pct_variants_extract_same": round(float(np.mean([r["extract_same"] for r in rs])), 3),
                     "n_flips": len(flips),
                     "pct_flips_extract_same": (round(float(np.mean([r["extract_same"] for r in flips])), 3)
                                                if flips else None)})
    return {"rows": rows}


def cmd_stage_decomp(args):
    res = compute_stage_decomp(args.run)
    print(f"[stage-decomp] run={args.run} main @keep=1.0. Jaccard=1 means the extraction stage did NOT change.")
    print(f"  {'op':<5}{'flip':>7}{'IOC_J':>7}{'Tech_J':>8}{'dState':>8}{'%var_extr_same':>15}{'%flip_extr_same':>16}")
    for r in res["rows"]:
        pf = "-" if r["pct_flips_extract_same"] is None else f"{r['pct_flips_extract_same']:.2f}"
        print(f"  {r['op']:<5}{r['flip_rate']:>7.3f}{r['ioc_jacc']:>7.2f}{r['tech_jacc']:>8.2f}"
              f"{r['d_states']:>8.2f}{r['pct_variants_extract_same']:>15.2f}{pf:>16}")
    print("  %flip_extr_same = among flips, how often the IOC/technique/State sets did NOT change -> the error\n"
          "  comes from the embeddings/classifier; the rest come from changes in the extraction stage.")
    _save_rows(args.run, "stage_decomp.csv", res["rows"])
    _save_json(run_dir(args.run) / "analysis" / "stage_decomp.json", res)


# ---------------------------------------------------------------------- per group
def _macro_ci(per_group, n_boot=1000, seed=17):
    """UNWEIGHTED mean over groups (weight 1 per group) + stratified bootstrap CI: resample documents
    WITHIN each group. per_group: {group: np.array of per-document values}."""
    gs = [g for g, a in per_group.items() if len(a)]
    if not gs:
        return None, None, None
    pt = float(np.mean([per_group[g].mean() for g in gs]))
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for g in gs:
        a = per_group[g]
        boots += a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    boots /= len(gs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(pt, 4), round(float(lo), 4), round(float(hi), 4)


def compute_by_group(run, keep=KEEP_FULL, metric="flip"):
    """Robustness per SOURCE GROUP (true group of the document) for each level and each model.
    per_group: mean flip over the group's documents. summary: mean over documents (micro, as in the
    other tables) and UNWEIGHTED mean over groups (macro) - macro does not depend on the group mix of
    the document set, so it is comparable between the pilot and the full run."""
    rows = [r for r in _rows(run, keep=keep) if r["op"] != "ID" and r.get(metric) is not None]
    per, summ = [], []
    if not rows:
        return {"per_group": per, "summary": summ}
    for m in _models(run):
        for level in sorted({r["level"] for r in rows if r["level"] is not None}):
            sub = [r for r in rows if r["model"] == m and r["level"] == level]
            if not sub:
                continue
            docs = defaultdict(list)
            grp = {}
            for r in sub:
                docs[r["doc_id"]].append(r[metric]); grp[r["doc_id"]] = r["group"]
            by_g = defaultdict(list)
            for d, vs in docs.items():
                by_g[grp[d]].append(float(np.mean(vs)))
            by_g = {g: np.array(v) for g, v in by_g.items()}
            for g, a in sorted(by_g.items()):
                per.append({"model": m, "level": level, "group": g, "n_docs": len(a),
                            metric: round(float(a.mean()), 4)})
            mi = bootstrap_ci({d: [float(np.mean(v))] for d, v in docs.items()})
            ma = _macro_ci(by_g)
            ma5 = _macro_ci({g: a for g, a in by_g.items() if len(a) >= MIN_DOCS})   # drop groups with too few documents
            gm = {g: float(a.mean()) for g, a in by_g.items()}
            gmax = max(gm, key=gm.get); gmin = min(gm, key=gm.get)
            summ.append({"model": m, "level": level, "n_docs": len(docs), "n_groups": len(by_g),
                         "micro": mi[0], "micro_lo": mi[1], "micro_hi": mi[2],
                         "macro": ma[0], "macro_lo": ma[1], "macro_hi": ma[2],
                         "n_groups_min5": sum(1 for a in by_g.values() if len(a) >= MIN_DOCS),
                         "macro_min5": ma5[0], "macro_min5_lo": ma5[1], "macro_min5_hi": ma5[2],
                         "max_group": gmax, "max_value": round(gm[gmax], 4),
                         "min_group": gmin, "min_value": round(gm[gmin], 4)})
    return {"per_group": per, "summary": summ, "metric": metric, "keep": keep}


def cmd_by_group(args):
    res = compute_by_group(args.run)
    print(f"[by-group] run={args.run} @keep=1.0 - flip per source group; micro = mean over documents, "
          f"macro = unweighted mean over groups")
    for s in res["summary"]:
        print(f"  {s['model']:<9} level {s['level']}: micro {s['micro']:.3f} [{s['micro_lo']:.3f},{s['micro_hi']:.3f}]"
              f" | macro {s['macro']:.3f} [{s['macro_lo']:.3f},{s['macro_hi']:.3f}] ({s['n_groups']} groups)"
              f" | highest {s['max_group']} {s['max_value']:.3f}, lowest {s['min_group']} {s['min_value']:.3f}")
    _save_rows(args.run, "by_group.csv", res["per_group"])
    _save_rows(args.run, "by_group_summary.csv", res["summary"])
    _save_json(run_dir(args.run) / "analysis" / "by_group.json", res)


# ---------------------------------------------------------------------- curves
def compute_curves(run):
    valid = [r for r in build_table(run) if r["valid"]]
    by_k = aggregate([r for r in valid if r["keep"] == KEEP_FULL and r["k"] != "-"],
                     ("op", "k", "model"), metrics=("flip", "delta_p", "violation"))
    by_keep = aggregate(valid, ("op", "model", "keep"), metrics=("flip", "delta_p", "violation", "acc_drop"))
    by_level = aggregate(valid, ("level", "model", "keep"), metrics=("flip", "delta_p", "violation", "acc_drop"))
    # accuracy of the ORIGINAL documents at each cut-off (each document counted once) -> reference for acc_drop
    oc = {}
    for r in valid:
        oc.setdefault((r["model"], r["keep"]), {})[r["doc_id"]] = r["orig_correct"]
    base_acc = [{"model": m, "keep": k, "acc_orig": round(float(np.mean(list(d.values()))), 4), "n_docs": len(d)}
                for (m, k), d in sorted(oc.items(), key=lambda t: (t[0][0], t[0][1]))]
    # curves over k on a FIXED document set: for each operator keep only the k values present for at
    # least 50% of the documents of the smallest k, and use only documents present at EVERY such k
    # (avoids artificial "drops" from shrinking sets, e.g. greedy k=4,5 keeps only IOC-rich documents)
    fixed = []
    full = [r for r in valid if r["keep"] == KEEP_FULL and r["k"] != "-"]
    for op in sorted({r["op"] for r in full}):
        ro = [r for r in full if r["op"] == op]
        docs_by_k = {}
        for r in ro:
            docs_by_k.setdefault(r["k"], set()).add(r["doc_id"])

        def _kv(k):
            return 1e9 if k == "all" else (float(k) if k.replace(".", "").isdigit() else 1e8)
        ks = sorted(docs_by_k, key=_kv)
        n0 = len(docs_by_k[ks[0]])
        ks = [k for k in ks if len(docs_by_k[k]) >= max(5, 0.5 * n0)]
        common = set.intersection(*(docs_by_k[k] for k in ks)) if ks else set()
        if len(common) < 5:
            continue
        sub = [r for r in ro if r["k"] in ks and r["doc_id"] in common]
        fixed += aggregate(sub, ("op", "k", "model"), metrics=("flip", "delta_p", "violation"))
    return {"by_k": by_k, "by_k_fixed": fixed, "by_keep": by_keep, "by_level": by_level, "base_acc": base_acc}


def cmd_curves(args):
    res = compute_curves(args.run)
    _save_rows(args.run, "curve_by_k.csv", res["by_k"])
    _save_rows(args.run, "curve_by_k_fixed_docs.csv", res["by_k_fixed"])
    _save_rows(args.run, "curve_by_keep.csv", res["by_keep"])
    _save_rows(args.run, "curve_by_level.csv", res["by_level"])
    _save_rows(args.run, "acc_orig_by_keep.csv", res["base_acc"])
    print(f"[curves] run={args.run}: flip over k (main, @100%, fixed document set per operator)")
    for r in [x for x in res["by_k_fixed"] if x["model"] == "main"]:
        print(f"  {r['op']:<4} k={r['k']:<4} flip={r['flip']:.3f} [{r['flip_lo']:.3f},{r['flip_hi']:.3f}] "
              f"delta_p={r['delta_p']:.3f}")
    log(f"-> {run_dir(args.run)}/tables/curve_by_k.csv, curve_by_keep.csv, curve_by_level.csv")


# ---------------------------------------------------------------------- figures
def cmd_figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = compute_curves(args.run)
    figs = run_dir(args.run) / "figures"; figs.mkdir(exist_ok=True)
    by_keep = res["by_keep"]
    ops = sorted({r["op"] for r in by_keep})
    keeps = sorted({r["keep"] for r in by_keep})
    models = sorted({r["model"] for r in by_keep})

    def save(name):
        plt.tight_layout()
        plt.savefig(figs / f"{name}.pdf"); plt.savefig(figs / f"{name}.png", dpi=150); plt.close()

    # 1) operator x keep heatmap for every model
    for m in models:
        M = np.full((len(ops), len(keeps)), np.nan)
        for i, op in enumerate(ops):
            for j, k in enumerate(keeps):
                r = next((x for x in by_keep if x["op"] == op and x["keep"] == k and x["model"] == m), None)
                if r:
                    M[i, j] = r["flip"]
        plt.figure(figsize=(6.5, max(3, len(ops) * 0.38)))
        vmax = max(0.2, float(np.nanmax(M)) if np.isfinite(M).any() else 0.2)
        plt.imshow(M, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        for i in range(len(ops)):
            for j in range(len(keeps)):
                if np.isfinite(M[i, j]):
                    plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
        plt.xticks(range(len(keeps)), [f"{int(k*100)}%" for k in keeps])
        plt.yticks(range(len(ops)), ops)
        plt.colorbar(label="flip rate")
        plt.title(f"Flip rate: operator x timeline cut-off ({m})")
        save(f"heatmap_op_keep_{m}")
    # 2) flip and delta_p over k (main, fixed document set per operator)
    by_k = [r for r in res.get("by_k_fixed") or res["by_k"] if r["model"] == "main"]
    kops = sorted({r["op"] for r in by_k})
    if kops:
        def kv(k):
            return 1e9 if k == "all" else (float(k) if k.replace(".", "").isdigit() else 1e8)
        klabels = sorted({r["k"] for r in by_k}, key=kv)
        pos = {k: i for i, k in enumerate(klabels)}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for op in kops:
            rs = sorted([r for r in by_k if r["op"] == op], key=lambda r: kv(r["k"]))
            xs = [pos[r["k"]] for r in rs]
            for ax, m in zip(axes, ("flip", "delta_p")):
                ax.errorbar(xs, [r[m] for r in rs],
                            yerr=[[r[m] - r[f"{m}_lo"] for r in rs], [r[f"{m}_hi"] - r[m] for r in rs]],
                            marker="o", capsize=3, label=op)
        for ax, lab in zip(axes, ("flip rate", "delta_p (drop in p(true group))")):
            ax.set_xticks(range(len(klabels))); ax.set_xticklabels(klabels)
            ax.set_xlabel("k (number of evidence items perturbed)"); ax.set_ylabel(lab)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle("Sensitivity to the amount of evidence (main, 100% timeline, fixed document set)")
        save("flip_vs_k_main")
    # 3) flip over the timeline, per level
    lv = res["by_level"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for level in sorted({r["level"] for r in lv if r["level"] not in (0, None)}):
        for m in models:
            rs = sorted([r for r in lv if r["level"] == level and r["model"] == m], key=lambda r: r["keep"])
            if rs:
                for ax, key in zip(axes, ("flip", "acc_drop")):
                    ax.plot([int(r["keep"] * 100) for r in rs], [r[key] for r in rs], marker="o",
                            linestyle="-" if m == "main" else "--", label=f"level {level} / {m}")
    for ax, lab in zip(axes, ("flip rate", "Delta acc (top-1 accuracy drop)")):
        ax.set_xlabel("% of timeline kept"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sensitivity over timeline cut-offs: flip (left) and accuracy drop (right)")
    save("flip_vs_timeline_by_level")
    # 4) violations per operator @100% (3 models, CI)
    full = [r for r in by_keep if r["keep"] == KEEP_FULL]
    plt.figure(figsize=(max(7, len(ops) * 0.55), 4.5))
    w = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        ys, lo, hi = [], [], []
        for op in ops:
            r = next((x for x in full if x["op"] == op and x["model"] == m), None)
            ys.append(r["violation"] if r else np.nan)
            lo.append(r["violation"] - r["violation_lo"] if r else 0)
            hi.append(r["violation_hi"] - r["violation"] if r else 0)
        plt.bar(np.arange(len(ops)) + mi * w, ys, w, yerr=[lo, hi], capsize=2, label=m)
    plt.xticks(np.arange(len(ops)) + w * (len(models) - 1) / 2, ops)
    plt.ylabel("expectation violation rate (95% CI)"); plt.legend(fontsize=8)
    plt.title("Expectation violations per operator (100% timeline)")
    save("violation_by_op")
    log(f"-> {figs}/ heatmap_op_keep_<model>, flip_vs_k_main, flip_vs_timeline_by_level, violation_by_op (.pdf + .png)")


def cmd_latex(args):
    rows = [r for r in compute_curves(args.run)["by_keep"] if r["keep"] == KEEP_FULL]
    tables = run_dir(args.run) / "tables"; tables.mkdir(exist_ok=True)
    ops = sorted({r["op"] for r in rows})
    models = sorted({r["model"] for r in rows})
    lines = [r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
             r"Operator & " + " & ".join(m.replace("_", r"\_") for m in models) + r" \\", r"\midrule"]
    for op in ops:
        cells = []
        for m in models:
            r = next((x for x in rows if x["op"] == op and x["model"] == m), None)
            cells.append(f"{r['flip']:.2f} [{r['flip_lo']:.2f}, {r['flip_hi']:.2f}]" if r else "-")
        lines.append(f"{op} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "flip_table.tex").write_text("\n".join(lines), encoding="utf-8")
    log(f"-> {tables}/flip_table.tex")


# ---------------------------------------------------------------------- all
def cmd_all(args):
    if not run_exists(args.run):
        raise SystemExit(f"Run '{args.run}' not found")
    for fn in (cmd_name_leak, cmd_order, cmd_sink, cmd_specificity, cmd_stage_decomp, cmd_by_group,
               cmd_curves, cmd_figures, cmd_latex):
        try:
            fn(args)
        except Exception as e:  # noqa: BLE001
            log(f"[analysis] {fn.__name__} error: {type(e).__name__}: {e}")
        print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("all", "name-leak", "order", "sink", "specificity", "stage-decomp", "by-group", "curves",
                 "figures", "latex"):
        p = sub.add_parser(name); p.add_argument("--run", required=True)
    p = sub.add_parser("size-probe"); p.add_argument("--n", type=int, default=2000); p.add_argument("--config", default=None)
    sub.add_parser("ttp-fp")
    args = ap.parse_args()
    fn = {"all": cmd_all, "size-probe": cmd_size_probe, "ttp-fp": cmd_ttp_fp,
          "name-leak": cmd_name_leak, "order": cmd_order, "sink": cmd_sink,
          "specificity": cmd_specificity, "stage-decomp": cmd_stage_decomp, "by-group": cmd_by_group,
          "curves": cmd_curves,
          "figures": cmd_figures, "latex": cmd_latex}[args.cmd]
    fn(args)


if __name__ == "__main__":
    main()
