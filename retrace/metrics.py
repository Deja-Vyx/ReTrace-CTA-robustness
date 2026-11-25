"""Aggregate metrics from variants.jsonl.gz + scores.jsonl.gz.

    python -m retrace.metrics summarize --run rand

Metrics (one row = one variant x one model x one timeline cut-off, compared with the ORIG
document for the same model and cut-off):
  flip          : the variant's top-1 differs from the ORIG top-1
  delta_p       : p(true group | ORIG) - p(true group | variant)   (positive = less confident)
  acc_drop      : correct(ORIG) - correct(variant); its mean is the top-1 accuracy drop (Delta acc).
                  At 100% every eligible original is correct, so acc_drop = flip; at early cut-offs
                  many originals are already wrong, and a "wrong -> other wrong" flip does not
                  reduce accuracy.
  confident_flip: flip AND p(new top-1) > confident_tau (configs/retrace.yaml)
  target_hit    : (E8) the variant's top-1 == the planted decoy group B
  violation     : the operator's expectation is violated - INV: flip | DIR: confident_flip | TGT: target_hit
  validity      : the variant passed the validity checks (only valid variants enter the tables)
Every rate comes with a 95% bootstrap confidence interval (resampling DOCUMENTS, not variants).

Tables written to runs/<run>/tables/:
  flip_by_op_model_keep.csv        : op x model x keep  (main table)
  flip_by_op_params_model_keep.csv : adds the parameters (k, mode, ...) -> curves over k
  flip_by_level_model_keep.csv     : level 0/1/2/3
  targeted_asr.csv, validity_rate.csv, rows_flat.csv.gz
"""
import argparse
import csv
import gzip
import re

import numpy as np

from .common import RUNS_DIR, iter_jsonl_any, log
from .config import get_config, run_dir


def run_exists(run: str) -> bool:
    return (RUNS_DIR / run / "scores.jsonl.gz").exists()


def load_run(run: str):
    d = run_dir(run)
    variants = {v["vid"]: v for v in iter_jsonl_any(d / "variants.jsonl.gz")}
    scores = list(iter_jsonl_any(d / "scores.jsonl.gz"))
    return d, variants, scores


def _orig_index(scores):
    """(doc_id, model, keep) -> the ORIG row."""
    out = {}
    for s in scores:
        if "|ORIG|" in s["vid"]:
            out[(s["vid"].split("|")[0], s["model"], s["keep"])] = s
    return out


def _pk(params):
    if not params:
        return "-"
    return ",".join(f"{k}={params[k]}" for k in sorted(params) if k != "greedy")


def k_of(params) -> str:
    if isinstance(params, dict):
        return str(params["k"]) if "k" in params else "-"
    m = re.search(r"(?:^|,)k=([^,]+)", params or "")
    return m.group(1) if m else "-"


_CACHE: dict = {}


def build_table(run: str, tau: float | None = None):
    key = (run, tau)
    if key in _CACHE:
        return _CACHE[key]
    if tau is None:
        tau = float(get_config().get("confident_tau", 0.5))
    from src.utils import PROCESSED, load_json
    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}
    d, variants, scores = load_run(run)
    orig = _orig_index(scores)
    rows = []
    for s in scores:
        vid = s["vid"]
        if "|ORIG|" in vid:
            continue
        v = variants.get(vid)
        if v is None:
            continue
        did = v["doc_id"]
        o = orig.get((did, s["model"], s["keep"]))
        if o is None:
            continue
        expect = v.get("expect") or "INV"
        flip = int(s["top1"] != o["top1"])
        conf_flip = int(flip and s["p_top1"] > tau)
        tgt = (v.get("meta") or {}).get("target")
        target_hit = int(tgt is not None and s["top1"] == g2i.get(tgt, -1))
        if expect == "TGT":
            viol = target_hit
        elif expect == "DIR":
            viol = conf_flip
        else:
            viol = flip
        params = v.get("params", {})
        rows.append({
            "vid": vid, "doc_id": did, "group": v["group"], "op": v["op"], "level": v.get("level"),
            "expect": expect, "params": _pk(params), "k": k_of(params),
            "strategy": v.get("strategy") or "random",
            "model": s["model"], "keep": s["keep"],
            "valid": int((v.get("valid") or {}).get("passed", True)),
            "flip": flip, "confident_flip": conf_flip, "target_hit": target_hit, "violation": viol,
            "delta_p": round(o["p_true"] - s["p_true"], 5),
            "p_true": s["p_true"], "p_true_orig": o["p_true"], "rank_true": s["rank_true"],
            "top1_correct": int(s["rank_true"] == 1), "orig_correct": int(o["rank_true"] == 1),
            # accuracy drop: 1 = original correct, variant wrong; -1 = the reverse; 0 = unchanged
            "acc_drop": int(o["rank_true"] == 1) - int(s["rank_true"] == 1),
            "p_target": s.get("p_target"), "target": tgt,
            "p_target_orig": (round(float(o["probs"][g2i[tgt]]), 5)
                              if tgt in g2i and o.get("probs") else None),
            "runner_up": v.get("runner_up"),
        })
    _CACHE[key] = rows
    return rows


def bootstrap_ci(values_by_doc: dict, n_boot=1000, seed=17):
    """values_by_doc: {doc_id: [values]} -> (mean, lo, hi); resamples doc_ids."""
    docs = list(values_by_doc)
    if not docs:
        return None, None, None
    flat = np.array([np.mean(values_by_doc[dd]) for dd in docs], dtype=float)
    point = float(flat.mean())
    if len(flat) == 1:
        return round(point, 4), round(point, 4), round(point, 4)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(flat), size=(n_boot, len(flat)))
    boots = flat[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


def aggregate(rows, by, metrics=("flip", "violation", "delta_p")):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[b] for b in by), []).append(r)
    out = []
    for k in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        rs = groups[k]
        rec = dict(zip(by, k))
        rec["n"] = len(rs)
        rec["n_docs"] = len({r["doc_id"] for r in rs})
        for m in metrics:
            per_doc = {}
            for r in rs:
                per_doc.setdefault(r["doc_id"], []).append(r[m])
            pt, lo, hi = bootstrap_ci(per_doc)
            rec[m] = pt; rec[f"{m}_lo"] = lo; rec[f"{m}_hi"] = hi
        rec["confident_flip"] = round(float(np.mean([r["confident_flip"] for r in rs])), 4)
        rec["robust_top1"] = round(float(np.mean([r["top1_correct"] for r in rs])), 4)
        out.append(rec)
    return out


def summarize(run: str, by=("op", "model", "keep")):
    rows = build_table(run)
    valid_rows = [r for r in rows if r["valid"]]
    out = aggregate(valid_rows, by)
    # keep the old column names for the plotting/LaTeX helpers
    for r in out:
        r["flip_rate"] = r["flip"]; r["flip_lo"] = r["flip_lo"]; r["flip_hi"] = r["flip_hi"]
    return out, rows


def targeted_asr(run: str):
    rows = [r for r in build_table(run) if r["op"] == "E8" and r["valid"] and r["target"]]
    by = {}
    for r in rows:
        kind = re.search(r"kind=([^,]+)", r["params"])
        by.setdefault((kind.group(1) if kind else "?", r["model"], r["keep"]), {}).setdefault(
            r["doc_id"], []).append(r["target_hit"])
    out = []
    for k, per_doc in sorted(by.items(), key=lambda t: tuple(str(x) for x in t[0])):
        pt, lo, hi = bootstrap_ci(per_doc)
        out.append({"kind": k[0], "model": k[1], "keep": k[2], "targeted_asr": pt,
                    "asr_lo": lo, "asr_hi": hi, "n": sum(len(v) for v in per_doc.values())})
    return out


def validity_rates(run: str):
    d, variants, _ = load_run(run)
    by = {}
    for v in variants.values():
        if v.get("op") in ("ORIG",) or v.get("text") is None:
            continue
        by.setdefault(v["op"], []).append(int((v.get("valid") or {}).get("passed", True)))
    return [{"op": k, "validity_rate": round(float(np.mean(v)), 4), "n": len(v)} for k, v in sorted(by.items())]


def write_csv(path, rows):
    if not rows:
        return
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


_write_csv = write_csv


def cmd_summarize(args):
    if not run_exists(args.run):
        raise SystemExit(f"Run '{args.run}' not found (missing scores.jsonl.gz)")
    d = run_dir(args.run)
    tables = d / "tables"; tables.mkdir(exist_ok=True)
    rows = build_table(args.run)
    valid = [r for r in rows if r["valid"]]
    main_t = aggregate(valid, ("op", "expect", "model", "keep"))
    for r in main_t:
        r["flip_rate"] = r["flip"]
    write_csv(tables / "flip_by_op_model_keep.csv", main_t)
    write_csv(tables / "flip_by_op_params_model_keep.csv",
              aggregate(valid, ("op", "params", "k", "model", "keep")))
    write_csv(tables / "flip_by_level_model_keep.csv", aggregate(valid, ("level", "model", "keep")))
    write_csv(tables / "targeted_asr.csv", targeted_asr(args.run))
    write_csv(tables / "validity_rate.csv", validity_rates(args.run))
    write_csv(tables / "rows_flat.csv.gz", rows)

    models = sorted({r["model"] for r in main_t})
    print(f"\nRun {args.run}: @keep=1.0, valid variants. flip = top-1 changed; viol = expectation violated "
          f"(INV: flip, DIR: confident flip, TGT: steered to decoy B)")
    head = f"{'op':<5}{'exp':<5}{'n_doc':>6}" + "".join(f"{m[:10] + ' flip':>17}" for m in models) + \
        "".join(f"{m[:10] + ' viol':>17}" for m in models)
    print(head)
    for op in sorted({r["op"] for r in main_t}):
        cells_f, cells_v, exp, nd = [], [], "", 0
        for m in models:
            r = next((x for x in main_t if x["op"] == op and x["model"] == m and x["keep"] == 1.0), None)
            cells_f.append(f"{r['flip']:.2f}" if r else "-")
            cells_v.append(f"{r['violation']:.2f}" if r else "-")
            if r:
                exp, nd = r["expect"], r["n_docs"]
        print(f"{op:<5}{exp:<5}{nd:>6}" + "".join(f"{c:>17}" for c in cells_f) +
              "".join(f"{c:>17}" for c in cells_v))
    log(f"-> {tables}/ (flip_by_op_model_keep, _params_, _level_, targeted_asr, validity_rate, rows_flat)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize"])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    {"summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
