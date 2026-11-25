"""Comparison report: ThreatMAMBA vs TTP-Match (Guru et al. 2025) vs TTP-Vec (Hussain & Petrova 2026) on the
ReTrace benchmark (levels 1 and 2), with the same metrics, the same document-level bootstrap CIs and the same verdict rule.

    python -m cta_systems.evaluation.report                 # decimal numbers (as in the ReTrace REPORT.md)
    python -m cta_systems.evaluation.report --fmt pct       # rates in %, differences in percentage points

Writes: outputs/cta_systems/REPORT_CTA_SYSTEMS.md, report_numbers.json, tables/report_*.csv, figures/*.png|pdf

Metric conventions (as in retrace/metrics.py):
  flip      : the variant's top-1 differs from the original's top-1 (same model, same cut-off)
  flip_c    : flip ONLY on documents whose original the model attributes CORRECTLY = evasion rate. For
              ThreatMAMBA at 100% every benchmark document is correct, so flip_c = flip in the ReTrace paper.
  acc_drop  : Delta acc = correct(original) - correct(variant), averaged over documents
  violation : expectation violated (INV: flip | DIR: confident flip p>tau | TGT: steered to group B)
  delta_p   : p(true group | original) - p(true group | variant)
  ASR (E8)  : rate of variants attributed to group B (on documents whose original the model attributes correctly)
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np

from retrace.analysis import _macro_ci, verdict
from retrace.metrics import aggregate

from ..common import (ADV_DIR, DISPLAY, FIG_DIR, OUT, RUNS_DIR, SHORT, TABLES_DIR, iter_jsonl_any, load_cfg, log,
                      model_sort_key, save_json, write_csv)
from .rows import adaptive_source, build_rows, ci, orig_rows, paired_diff, per_doc, transfer_source

FMT = {"mode": "dec"}
LEVEL_NAME = {0: "0 · control", 1: "1 · text", 2: "2 · evidence"}
OP_DESC = {
    "ID": "Rebuild unchanged (control)", "T1": "LLM paraphrases k sentences", "T2": "LLM splits/merges k sentences",
    "T3": "Abbreviation ↔ full form", "T4": "Move the IOC appendix to the top", "T5": "Insert benign sentences",
    "T6": "Change the date format (negative control)", "E1": "Mask k IOCs", "E2": "Drop / neutralise technique sentences",
    "E3": "Swap tool aliases", "E4": "Mask group names", "E5": "Remove temporal connectives / swap sentences",
    "E6": "LLM generalises tool names", "E7": "Insert unrelated IOCs", "E8": "False flag of group B",
}


# ------------------------------------------------------------------ number formatting
def f(x, digits=2):
    """Rate: decimal (0.29) or percentage with one decimal (29.0%)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if FMT["mode"] == "pct":
        return f"{100 * x:.1f}%"
    return f"{x:.{digits}f}"


def fpp(x, digits=3):
    """Difference of two rates / probabilities (delta_p, delta_acc): decimal or percentage points (pp)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if FMT["mode"] == "pct":
        return f"{100 * x:.1f} pp"
    return f"{x:.{digits}f}"


def fci(c, digits=2):
    if not c or c.get("mean") is None:
        return "–"
    if FMT["mode"] == "pct":
        return f"{100 * c['mean']:.1f}% [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"
    return f"{c['mean']:.{digits}f} [{c['lo']:.{digits}f}, {c['hi']:.{digits}f}]"


def fdiff(c, digits=3):
    """Signed difference of two rates: decimal or percentage points (pp)."""
    if not c or c.get("mean") is None:
        return "–"
    if FMT["mode"] == "pct":
        return f"{100 * c['mean']:+.1f} pp [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]"
    return f"{c['mean']:+.{digits}f} [{c['lo']:+.{digits}f}, {c['hi']:+.{digits}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _cell(rows, metric="flip", min_docs=1):
    c = ci(rows, metric)
    if c["n_docs"] < min_docs:
        return "–"
    return f"{fci(c)} ({c['n_docs']})"


def two_sided_verdict(c, min_effect=0.01):
    if not c or c.get("mean") is None or c["n_docs"] < 5:
        return "na"
    sig = c["lo"] > 0 or c["hi"] < 0
    big = abs(c["mean"]) >= min_effect
    return "supported" if sig and big else ("weak" if sig or big else "not_supported")


def _direction(v: str, mean: float) -> str:
    """Only say "less/more robust" when the evidence supports it; weak evidence is only a "trend"."""
    if v in ("not_supported", "na") or mean is None:
        return ""
    word = "less robust" if mean > 0 else "more robust"
    return f" → {word}" if v == "supported" else f" → tends to be {word} (not conclusive)"


VERDICT_DIFF = {"supported": "supported difference", "weak": "weak difference", "not_supported": "no difference",
                "na": "not enough data"}


# ------------------------------------------------------------------ data loading
class Data:
    def __init__(self, cfg: dict, models_cmp: list[str] | None = None):
        self.cfg = cfg
        self.tm = list(cfg["threatmamba_models"])
        self.levels = set(cfg["levels"])
        self.rows, self.origs, self.variants = {}, {}, {}
        for run in cfg["runs"]:
            src = transfer_source(run, self.tm, models_cmp)
            if src is None:
                continue
            v, sc = src
            self.variants[run] = v
            rs = [r for r in build_rows(v, sc) if r["valid"] and r["level"] in self.levels]
            self.rows[run] = rs
            self.origs[run] = orig_rows(sc, v)
        present = {r["model"] for rs in self.rows.values() for r in rs}
        self.cmp = sorted([m for m in present if m not in self.tm], key=model_sort_key)
        self.models = sorted(present, key=model_sort_key)
        evaluation = [m for m in cfg["ttp_vec"]["main_table_models"] if m in present]
        self.main_models = [m for m in self.models if m in self.tm or m == "ttp_match" or m in evaluation]
        self.extra_models = [m for m in self.models if m not in self.main_models]
        self.adapt = {}
        for m in self.cmp:
            src = adaptive_source(m)
            if src:
                v, sc = src
                self.adapt[m] = [r for r in build_rows(v, sc) if r["valid"] and r["level"] in self.levels]
                self.variants[f"adapt_{m}"] = v
        self.extract = defaultdict(dict)
        for run in list(self.rows) + [f"adapt_{m}" for m in self.adapt]:
            for x in iter_jsonl_any(RUNS_DIR / run / "extract.jsonl.gz"):
                self.extract[run][(x["vid"], x["model"])] = x["x"]
        self.elig = {r["doc_id"]: r for r in iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl")}

    def sel(self, run, keep=1.0, correct=False, **flt):
        rs = self.rows.get(run, [])
        out = []
        for r in rs:
            if keep is not None and r["keep"] != keep:
                continue
            if correct and not r["orig_correct"]:
                continue
            if all((v(r[k]) if callable(v) else r[k] == v) for k, v in flt.items()):
                out.append(r)
        return out


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ report sections
def sec_setup(D: Data, L: list, nums: dict):
    L.append("## 0. Setup\n")
    L.append("- **Case-study systems** (re-implemented from the authors' code, trained on the same CTI2Attacker training "
             "set of 2,515 reports, evaluated on the same 629 test reports and the same 144-report ReTrace benchmark):")
    L.append("  - **TTP-Match (Guru et al. 2025)** (arXiv 2505.11547): 3-sentence windows → embedding-based ATT&CK technique "
             "matching → P(technique | group) profiles × prior. OpenAI embeddings are replaced with an open model run locally.")
    L.append("  - **TTP-Vec (Hussain & Petrova 2026)** (Future Internet 18(8):433): one-hot techniques → StandardScaler → rarity "
             "weights (1/number of groups)^3 → KNN (k=1) / linear SVM / RF / DT / NB. There are no victim fields, so only "
             "the \"TTP-only\" configuration is used; techniques come from **ThreatMAMBA's own TTP extractor**.")
    L.append("- **Operators**: only level 1 (T1–T6) and level 2 (E1–E8) plus the ID control; level 3 edits the CSKG directly "
             "and does not apply to models without a graph.")
    L.append("- **Transfer vs adaptive**: the `rand` (random) runs do not depend on the model. The \"strongest\" variants of "
             "the `worst`/`llm` runs and the decoy group B are selected with the scores of ThreatMAMBA `main` → for the "
             "other systems these are **transfer attacks**. Section 4 adds **adaptive attacks** (search repeated with the "
             "system itself) if `cta_systems.evaluation.adaptive` has been run.")
    L.append("- **Timeline cut-offs**: the text is cut exactly at the sentence ThreatMAMBA uses as the x% cut-off (the CSKG "
             "is rebuilt for every variant), so every model sees the same part of the report.")
    rows = []
    for run, rs in D.rows.items():
        meta = {}
        mp = RUNS_DIR / run / "meta.json"
        if mp.exists():
            meta = json.load(open(mp, encoding="utf-8"))
        docs = {r["doc_id"] for r in rs}
        vids = {r["vid"] for r in rs}
        ms = sorted({r["model"] for r in rs}, key=model_sort_key)
        rows.append([run, len(docs), len(vids), ", ".join(SHORT.get(m, m) for m in ms),
                     f"{meta.get('state_match', '–')}/{meta.get('state_check', '–')}"])
    L.append("")
    L.append(table(["Run", "Documents", "Valid level 0–2 variants", "Models scored",
                    "State counts matching the original run (CSKG rebuild check)"], rows))
    L.append("")
    nums["setup"] = rows


def sec_clean(D: Data, L: list, nums: dict):
    L.append("## 1. Clean accuracy\n")
    rows = []
    struct = {}
    from src.utils import OUTPUTS
    tm = _read_csv(OUTPUTS / "metrics_all.csv")        # evaluation results of the base repository (src.evaluate)
    for r in tm:
        if r.get("model") in D.tm:
            struct[r["model"]] = {"top1": float(r["top1_micro"]), "macro_f1": float(r["f1_macro"])}
            rows.append([DISPLAY.get(r["model"], r["model"]), f(float(r["top1_micro"]), 3), f(float(r["top3_micro"]), 3),
                         f(float(r["top5_micro"]), 3), f(float(r["f1_macro"]), 3), "–", "outputs/metrics_all.csv"])
    for r in _read_csv(TABLES_DIR / "ttp_match_clean_test.csv"):
        name = {"ttp_match": "TTP-Match (configuration selected on validation, used for the robustness tests)",
                "ttp_match_paper_default": "TTP-Match (paper Algorithm 1: top-1, expert prior)",
                "ttp_match_paper_uniform_prior": "TTP-Match (top-1, uniform prior)",
                "ttp_match_code_everything_py": "TTP-Match (as everything.py: top-5, no prior)",
                "prior_only": "Group frequency only (prior)",
                "random (expected)": "Random (expected)"}.get(r["model"], r["model"])
        if r["model"] in ("prior_only", "random (expected)") and any(x[0].startswith("Group frequency only") for x in rows):
            continue
        if r.get("top1"):
            struct[r["model"]] = {"top1": float(r["top1"]), "macro_f1": float(r["macro_f1"]) if r.get("macro_f1") else None,
                                  "mean_rank": float(r["mean_rank"]) if r.get("mean_rank") else None}
        rows.append([name] + [f(float(r[k]), 3) if r.get(k) not in (None, "") else "–"
                              for k in ("top1", "top3", "top5", "macro_f1")] +
                    [f"{float(r['mean_rank']):.2f}" if r.get("mean_rank") else "–", "tables/ttp_match_clean_test.csv"])
    for r in _read_csv(TABLES_DIR / "ttp_vec_clean_test.csv"):
        if r["model"] == "prior_only":
            continue
        wt = "rarity weights" if r["weighting"].startswith("rarity") else "unweighted"
        struct[r["model"]] = {"top1": float(r["top1"]), "macro_f1": float(r["macro_f1"]), "mean_rank": float(r["mean_rank"])}
        rows.append([f"TTP-Vec ({r['algo'].upper()}, {wt})"] +
                    [f(float(r[k]), 3) for k in ("top1", "top3", "top5", "macro_f1")] +
                    [f"{float(r['mean_rank']):.2f}", "tables/ttp_vec_clean_test.csv"])
    L.append("### 1.1 Test set (629 reports)\n")
    L.append(table(["Model", "top-1", "top-3", "top-5", "macro-F1", "Mean rank (/20)", "Source"], rows))
    cv = _read_csv(TABLES_DIR / "ttp_vec_cv_train.csv")
    if cv:
        L.append("\nProtocol of the TTP-Vec paper (stratified 10-fold on train): the original code fits the scaler and the "
                 "weights on all labelled data **before** splitting into folds (label leakage); the \"fixed\" column computes them inside each fold.\n")
        L.append(table(["Algorithm", "Original code (leaky)", "Fixed"],
                       [[r["algo"].upper(), f"{f(float(r['leaky_mean']), 3)} ± {f(float(r['leaky_std']), 3)}",
                         f"{f(float(r['fixed_mean']), 3)} ± {f(float(r['fixed_std']), 3)}"] for r in cv]))
    # on the benchmark
    run = "rand" if "rand" in D.origs else next(iter(D.origs), None)
    if run:
        br = []
        by = defaultdict(dict)
        for o in D.origs[run]:
            by[(o["model"], o["keep"])][o["doc_id"]] = o["correct"]
        keeps = sorted({k for _m, k in by})
        for m in D.models:
            br.append([DISPLAY.get(m, m)] + [f(np.mean(list(by[(m, k)].values())), 3) if (m, k) in by else "–"
                                             for k in keeps] + [len(by.get((m, 1.0), {}))])
        L.append(f"\n### 1.2 On the {len(by.get((D.models[0], 1.0), {}))} benchmark reports (originals, by timeline cut-off)\n")
        L.append(table(["Model"] + [f"{int(k * 100)}%" for k in keeps] + ["Documents"], br))
        L.append("\nThe benchmark consists of reports that all three ThreatMAMBA variants attribute correctly at 100%, so "
                 "ThreatMAMBA scores 1.0 in the 100% column by construction; other models may already be wrong on part of "
                 "the set. The robustness tables below therefore report **flip_c** on the reports that *the model itself* "
                 "attributes correctly (number in brackets = documents), and section 7 compares on the intersection for fairness.\n")
        nums["benchmark_acc"] = br
    nums["clean"] = rows
    nums["clean_struct"] = struct


def _ops_table(D: Data, run: str, level: int, by_params=False, strategy=None, models=None, correct=True):
    models = models or D.main_models
    rs = [r for r in D.sel(run, level=level, correct=correct) if strategy is None or r["strategy"] == strategy]
    keys = sorted({(r["op"], r["params"] if by_params else "-", r["strategy"]) for r in rs},
                  key=lambda t: (t[0][0] != "I", t[0], t[1], t[2]))
    out = []
    for op, pk, st in keys:
        sub = [r for r in rs if r["op"] == op and (not by_params or r["params"] == pk) and r["strategy"] == st]
        exp = sub[0]["expect"] if sub else ""
        cells = [_cell([r for r in sub if r["model"] == m]) for m in models]
        out.append([op, OP_DESC.get(op, ""), exp] + ([pk] if by_params else []) + ([st] if strategy is None else []) + cells)
    return out, models


def sec_level1(D: Data, L: list, nums: dict):
    L.append("## 2. Level 1 — meaning-preserving rewrites (expectation INV), 100% timeline\n")
    if "rand" in D.rows:
        rows, ms = _ops_table(D, "rand", 1, strategy="random")
        L.append("### 2.1 Rule-based operators, random variants (run `rand`) — flip_c [95% CI] (documents)\n")
        L.append(table(["Operator", "Description", "Expectation"] + [SHORT.get(m, m) for m in ms], rows))
        idr = D.sel("rand", op="ID", correct=False)
        if idr:
            L.append("\nID control (unchanged rebuild): " + ", ".join(
                f"{SHORT.get(m, m)} {f(ci([r for r in idr if r['model'] == m], 'flip')['mean'], 3)}" for m in D.models) +
                " — must be 0 (every model is deterministic).\n")
        nums["level1_rand"] = rows
    if "llm" in D.rows:
        rows, ms = _ops_table(D, "llm", 1, by_params=True)
        L.append("### 2.2 LLM operators (run `llm`, best-of-3 selected on ThreatMAMBA → transfer)\n")
        L.append(table(["Operator", "Description", "Expectation", "Parameters", "Strategy"] + [SHORT.get(m, m) for m in ms], rows))
        L.append("")
        nums["level1_llm"] = rows


def sec_level2(D: Data, L: list, nums: dict):
    L.append("## 3. Level 2 — evidence manipulation, random variants (run `rand`), 100% timeline\n")
    if "rand" not in D.rows:
        L.append("_No rand run yet._\n")
        return
    rows, ms = _ops_table(D, "rand", 2, strategy="random")
    L.append("### 3.1 flip_c [95% CI] (documents)\n")
    L.append(table(["Operator", "Description", "Expectation"] + [SHORT.get(m, m) for m in ms], rows))
    # violations and delta_p
    rs = D.sel("rand", level=2, correct=True, strategy="random")
    vr = []
    for op in sorted({r["op"] for r in rs}):
        sub = [r for r in rs if r["op"] == op]
        vr.append([op, sub[0]["expect"]] + [f"{fci(ci([r for r in sub if r['model'] == m], 'violation'))} / "
                                            f"{fpp(ci([r for r in sub if r['model'] == m], 'delta_p')['mean'])}"
                                            for m in ms])
    L.append("\n### 3.2 Expectation violations [CI] / mean Δp\n")
    L.append(table(["Operator", "Expectation"] + [SHORT.get(m, m) for m in ms], vr))
    L.append("\nINV: violation = flip; DIR: violation = **confident** flip (new top-1 p > 0.5); TGT: steered to group B. "
             "Δp of a model with 0/1 probabilities (KNN k=1) only takes the values 0 or ±1.\n")
    nums["level2_rand"] = rows
    nums["level2_rand_viol_dp"] = vr


def sec_worst(D: Data, L: list, nums: dict):
    L.append("## 4. Strongest attacks, 100% timeline\n")
    ms = D.main_models
    src = [(run, lv) for run, lv in (("worst", 1), ("worst", 2), ("llm", 2)) if run in D.rows]
    if src:
        L.append("### 4.1 Transfer: variants selected on ThreatMAMBA `main` — flip_c [CI] (documents)\n")
        rows = []
        for run, lv in src:
            r2, _ = _ops_table(D, run, lv, by_params=(run == "llm"))
            for r in r2:
                rows.append([run] + r)
        hdr = ["Run", "Operator", "Description", "Expectation"]
        # _ops_table: [op, desc, exp, (pk), st, cells...] -> align the columns
        norm = []
        for r in rows:
            if r[0] == "llm":
                norm.append(r[:4] + [f"{r[5]} ({r[4]})"] + r[6:])
            else:
                norm.append(r[:4] + [r[4]] + r[5:])
        L.append(table(hdr + ["Strategy"] + [SHORT.get(m, m) for m in ms], norm))
        L.append("\nThe ThreatMAMBA column is an **adaptive** black-box attack (search with `main`'s own scores); the other "
                 "columns use the same variants and are therefore **transfer** attacks — usually underestimating how fragile that model is.\n")
        nums["worst_transfer"] = norm
    if D.adapt:
        L.append("### 4.2 Adaptive: search repeated with each system itself — flip_c [CI] (documents)\n")
        wrs = [r for r in D.rows.get("worst", []) if r["keep"] == 1.0 and r["orig_correct"] and r["model"] == D.tm[0]]
        keys = sorted({(r["op"], r["strategy"]) for m in D.adapt for r in D.adapt[m] if r["keep"] == 1.0})
        rows = []
        for op, st in keys:
            cells = [_cell([r for r in wrs if r["op"] == op and r["strategy"] == st])]
            for m in D.adapt:
                cells.append(_cell([r for r in D.adapt[m] if r["keep"] == 1.0 and r["orig_correct"]
                                    and r["op"] == op and r["strategy"] == st]))
            rows.append([op, OP_DESC.get(op, ""), st] + cells)
        L.append(table(["Operator", "Description", "Strategy", "ThreatMAMBA (run worst)"] +
                       [f"{SHORT.get(m, m)} (adaptive)" for m in D.adapt], rows))
        L.append("\nSame operators, same parameter grid, same per-document seeds and same search strategies as the ReTrace "
                 "`worst` run; only the black-box model differs (and the E8 decoy B is the second-ranked group of that "
                 "system itself).\n")
        nums["worst_adaptive"] = rows


def sec_asr(D: Data, L: list, nums: dict):
    L.append("## 5. E8 false flags — rate of being steered to group B (ASR), 100% timeline\n")
    rows = []
    for run, lab in (("rand", "random"), ("worst", "targeted, transfer")):
        rs = [r for r in D.sel(run, op="E8", correct=True) if r["target"]]
        for kind in ("ioc", "ttp", "name"):
            sub = [r for r in rs if f"kind={kind}" in r["params"]]
            if sub:
                rows.append([kind, lab] + [_cell([r for r in sub if r["model"] == m], "target_hit") for m in D.main_models])
    for m, rs in D.adapt.items():
        sub0 = [r for r in rs if r["keep"] == 1.0 and r["orig_correct"] and r["op"] == "E8" and r["target"]]
        for kind in ("ioc", "ttp", "name"):
            sub = [r for r in sub0 if f"kind={kind}" in r["params"]]
            if sub:
                rows.append([kind, f"targeted, adaptive ({SHORT.get(m, m)})"] +
                            [_cell(sub, "target_hit") if mm == m else "" for mm in D.main_models])
    if not rows:
        L.append("_No valid E8 variants in the scored runs yet._\n")
    if rows:
        L.append(table(["Trace of B", "Selection"] + [SHORT.get(m, m) for m in D.main_models], rows))
        L.append("\nOnly reports whose original the model attributes correctly. In transfer rows, B is ThreatMAMBA's "
                 "second-ranked group; in adaptive rows, B is the second-ranked group of the system itself.\n")
    nums["asr"] = rows


def sec_timeline(D: Data, L: list, nums: dict):
    if "rand" not in D.rows:
        return
    L.append("## 6. By timeline cut-off — flip / Δacc per level (run `rand`, all benchmark documents)\n")
    rs = [r for r in D.rows["rand"] if r["strategy"] == "random" and r["level"] in (1, 2)]
    lv = aggregate(rs, ("level", "model", "keep"), metrics=("flip", "acc_drop"))
    keeps = sorted({r["keep"] for r in lv})
    acc0 = defaultdict(dict)
    for o in D.origs["rand"]:
        acc0[(o["model"], o["keep"])][o["doc_id"]] = o["correct"]
    rows = []
    for m in D.main_models:
        rows.append(["Accuracy of originals", SHORT.get(m, m)] +
                    [f(np.mean(list(acc0[(m, k)].values())), 3) if (m, k) in acc0 else "–" for k in keeps])
    for level in (1, 2):
        for m in D.main_models:
            d = {r["keep"]: r for r in lv if r["level"] == level and r["model"] == m}
            rows.append([LEVEL_NAME[level], SHORT.get(m, m)] +
                        [f"{f(d[k]['flip'], 3)} / {fpp(d[k]['acc_drop'])}" if k in d else "–" for k in keeps])
    L.append(table(["Level", "Model"] + [f"{int(k * 100)}%" for k in keeps], rows))
    L.append("\nEach cell: **flip / Δacc** over all benchmark documents (as in table 3.3 of the ReTrace report). Cut-off x% = "
             "the part of the report up to the sentence ThreatMAMBA uses as the x% State cut-off.\n")
    nums["timeline"] = lv


def sec_paired(D: Data, L: list, nums: dict):
    L.append("## 7. Direct comparison with ThreatMAMBA (paired by document, 100% timeline)\n")
    base = D.tm[0]
    srcs = []
    if "rand" in D.rows:
        srcs += [("Level 1, rule-based (random)", D.sel("rand", level=1, strategy="random"), None),
                 ("Level 2 (random)", D.sel("rand", level=2, strategy="random"), None)]
    if "llm" in D.rows:
        srcs += [("Level 1, LLM (T1, T2; transfer)", D.sel("llm", level=1), None)]
    if "worst" in D.rows:
        srcs += [("Level 2 strongest (transfer)", D.sel("worst", level=2), None)]
    rows, out = [], []
    for lab, rs, _ in srcs:
        for m in D.cmp:
            a = [r for r in rs if r["model"] == m and r["orig_correct"]]
            b = [r for r in rs if r["model"] == base and r["orig_correct"]]
            # the same VARIANT SET (not only the same documents): avoids bias when one model has only scored part of it
            cv = {r["vid"] for r in a} & {r["vid"] for r in b}
            a = [r for r in a if r["vid"] in cv]; b = [r for r in b if r["vid"] in cv]
            docs = set(per_doc(a, "flip")) & set(per_doc(b, "flip"))
            d = paired_diff(a, b, "flip", docs)
            v = two_sided_verdict(d)
            direction = _direction(v, d["mean"])
            rows.append([lab, SHORT.get(m, m), fci(ci([r for r in a if r["doc_id"] in docs], "flip")),
                         fci(ci([r for r in b if r["doc_id"] in docs], "flip")), fdiff(d), d["n_docs"],
                         VERDICT_DIFF[v] + direction])
            out.append({"source": lab, "model": m, **{f"diff_{k}": d[k] for k in ("mean", "lo", "hi", "n_docs")},
                        "verdict": v})
    if D.adapt and "worst" in D.rows:
        wrs = [r for r in D.rows["worst"] if r["keep"] == 1.0 and r["model"] == base and r["orig_correct"]]
        for m, rs in D.adapt.items():
            a = [r for r in rs if r["keep"] == 1.0 and r["orig_correct"] and r["level"] == 2]
            ops = {(r["op"], r["strategy"]) for r in a} & {(r["op"], r["strategy"]) for r in wrs}
            a = [r for r in a if (r["op"], r["strategy"]) in ops]
            b = [r for r in wrs if (r["op"], r["strategy"]) in ops]
            docs = set(per_doc(a, "flip")) & set(per_doc(b, "flip"))
            d = paired_diff(a, b, "flip", docs)
            v = two_sided_verdict(d)
            direction = _direction(v, d["mean"])
            rows.append(["Level 2 strongest (**both adaptive**)", SHORT.get(m, m),
                         fci(ci([r for r in a if r["doc_id"] in docs], "flip")),
                         fci(ci([r for r in b if r["doc_id"] in docs], "flip")), fdiff(d), d["n_docs"],
                         VERDICT_DIFF[v] + direction])
            out.append({"source": "adaptive_level2", "model": m, **{f"diff_{k}": d[k] for k in ("mean", "lo", "hi", "n_docs")},
                        "verdict": v})
    if rows:
        L.append(table(["Variant source", "Model", "flip_c model", "flip_c ThreatMAMBA", "Difference (model − ThreatMAMBA)",
                        "Documents", "Verdict"], rows))
        L.append("\nOnly reports that **both** models attribute correctly; each document averages the flips of its "
                 "variants, then documents are paired; 1000-sample document-level bootstrap CI. The verdict uses the "
                 "ReTrace two-condition rule (CI excludes 0 and |difference| ≥ 0.01).\n")
    nums["paired"] = out


def sec_group(D: Data, L: list, nums: dict):
    if "rand" not in D.rows:
        return
    L.append("## 8. By group — mean over documents (micro) and unweighted mean over groups (macro)\n")
    rows, out = [], []
    for level in (1, 2):
        for m in D.main_models:
            rs = [r for r in D.sel("rand", level=level, correct=True, strategy="random") if r["model"] == m]
            docs, grp = defaultdict(list), {}
            for r in rs:
                docs[r["doc_id"]].append(r["flip"]); grp[r["doc_id"]] = r["group"]
            if not docs:
                continue
            by = defaultdict(list)
            for d_, vs in docs.items():
                by[grp[d_]].append(float(np.mean(vs)))
            by = {g: np.array(v) for g, v in by.items()}
            mi = ci(rs, "flip")
            ma = _macro_ci(by)
            gm = {g: float(a.mean()) for g, a in by.items()}
            gmax = max(gm, key=gm.get)
            rows.append([LEVEL_NAME[level], SHORT.get(m, m), fci(mi), fci({"mean": ma[0], "lo": ma[1], "hi": ma[2]}),
                         len(by), f"{gmax} {f(gm[gmax], 3)}"])
            out.append({"level": level, "model": m, "micro": mi["mean"], "macro": ma[0], "n_groups": len(by),
                        "per_group": {g: round(v, 4) for g, v in gm.items()}})
    L.append(table(["Level", "Model", "micro [CI]", "macro [CI]", "Groups", "Least robust group"], rows))
    L.append("\nflip_c, random variants, 100% timeline. Full per-group table: `tables/report_by_group.csv`.\n")
    nums["by_group"] = out
    flat = [{"level": o["level"], "model": o["model"], "group": g, "flip_c": v} for o in out for g, v in o["per_group"].items()]
    write_csv(TABLES_DIR / "report_by_group.csv", flat)


def sec_stage(D: Data, L: list, nums: dict):
    L.append("## 9. Extraction stage or classification stage? (100% timeline)\n")
    rows, out = [], []
    for run in [r for r in ("rand", "llm") if r in D.rows]:
        V = D.variants[run]
        X = D.extract.get(run, {})
        rs = D.sel(run, correct=True)
        for m in D.models:
            for op in sorted({r["op"] for r in rs if r["model"] == m and r["op"] != "ID"}):
                sub = [r for r in rs if r["model"] == m and r["op"] == op]
                flips = [r for r in sub if r["flip"]]
                if not sub:
                    continue
                if m in D.tm:
                    def same(r):
                        tr = (V.get(r["vid"]) or {}).get("trace") or {}
                        if not tr:
                            return None
                        return int(tr.get("ioc_jacc", 1) == 1 and tr.get("tech_jacc", 1) == 1 and tr.get("d_states", 0) == 0)
                else:
                    def same(r, m=m):
                        o = X.get((f"{r['doc_id']}|ORIG|-|c0", m)); x = X.get((r["vid"], m))
                        if o is None or x is None:
                            return None
                        return int(dict(o) == dict(x))
                sv = [same(r) for r in sub]; sf = [same(r) for r in flips]
                sv = [s for s in sv if s is not None]; sf = [s for s in sf if s is not None]
                rows.append([run, op, SHORT.get(m, m), f(float(np.mean([r["flip"] for r in sub])), 3),
                             f(float(np.mean(sv)), 2) if sv else "–", len(flips), f(float(np.mean(sf)), 2) if sf else "–"])
                out.append({"run": run, "op": op, "model": m, "n": len(sub), "n_flips": len(flips),
                            "pct_variants_extract_same": float(np.mean(sv)) if sv else None,
                            "pct_flips_extract_same": float(np.mean(sf)) if sf else None})
    if rows:
        L.append(table(["Run", "Operator", "Model", "flip_c", "% variants with unchanged extraction", "Flips",
                        "% flips with unchanged extraction"], rows))
        L.append("\nExtraction = the classifier's actual input: ThreatMAMBA (IOC set + techniques + State count), TTP-Vec "
                 "(technique set assigned by ThreatMAMBA's extractor), TTP-Match (how often each technique is matched). For "
                 "TTP-Vec the prediction is a deterministic function of the technique set, so every flip comes from a change in extraction.\n")
    nums["stage"] = out
    write_csv(TABLES_DIR / "report_stage.csv", out)


def sec_shortcuts(D: Data, L: list, nums: dict):
    if "rand" not in D.rows:
        return
    L.append("## 10. Group names and groups that \"attract\" wrong predictions (run `rand`, 100% timeline)\n")
    rows, out = [], []
    from ..common import groups as _groups
    G = _groups()
    prev = Counter(r["group"] for r in D.elig.values())
    n_el = max(1, sum(prev.values()))
    for m in D.main_models:
        e4 = [r for r in D.sel("rand", op="E4", correct=True) if r["model"] == m and "scope=true" in r["params"].lower()]
        ing = [r for r in e4 if D.elig.get(r["doc_id"], {}).get("name_in_graph")]
        c_ing = ci(ing, "delta_p")
        e8 = [r for r in D.sel("rand", op="E8", correct=True) if r["model"] == m and "kind=name" in r["params"]]
        for r in e8:
            r["p_target_gain"] = None if r["p_target"] is None or r["p_target_orig"] is None else r["p_target"] - r["p_target_orig"]
        c_gain = ci(e8, "p_target_gain")
        fl = [r for r in D.sel("rand", correct=True) if r["model"] == m and r["op"] != "ID" and r["flip"]]
        dest = Counter(G[r["top1"]] for r in fl)
        tot = sum(dest.values())
        sink = "–"
        if tot >= 20:
            g, c = dest.most_common(1)[0]
            base = prev.get(g, 0) / n_el
            sink = f"{g}: {c / tot * 100:.0f}% of flips ({base * 100:.1f}% of the benchmark)"
        rows.append([SHORT.get(m, m), f"{fci(c_ing, 3)} ({c_ing['n_docs']})", verdict(c_ing["mean"], c_ing["lo"], 0.0, c_ing["n_docs"]),
                     f"{fci(c_gain, 3)} ({c_gain['n_docs']})", verdict(c_gain["mean"], c_gain["lo"], 0.0, c_gain["n_docs"]),
                     f"{tot}", sink])
        out.append({"model": m, "S3a_dp": c_ing, "S3b_gain": c_gain, "flips": tot, "sink": sink})
    L.append(table(["Model", "S3a: mask true group name → Δp", "verdict", "S3b: insert group B's name → p(B) gain", "verdict",
                    "Flips", "S5: strongest attractor group"], rows))
    L.append("\nS3a only counts reports whose group name appears in an evidence sentence (`name_in_graph`). Verdict: "
             "supported / weak / not_supported / na, using the ReTrace rule.\n")
    nums["shortcuts"] = out


def sec_extra(D: Data, L: list, nums: dict):
    if not D.extra_models or "rand" not in D.rows:
        return
    L.append("## Appendix A. Remaining TTP-Vec algorithms (run `rand`, flip_c, 100%)\n")
    rows = []
    for level in (1, 2):
        r2, ms = _ops_table(D, "rand", level, strategy="random", models=D.extra_models)
        rows += r2
    L.append(table(["Operator", "Description", "Expectation"] + [SHORT.get(m, m) for m in D.extra_models], rows))
    L.append("")
    nums["extra_models"] = rows


def summary_lines(D: Data, nums: dict) -> list[str]:
    out = []
    cs = nums.get("clean_struct") or {}
    parts = [f"{SHORT.get(m, m)} {f(cs[m]['top1'], 3)}" for m in D.main_models if m in cs]
    if parts:
        out.append("Top-1 accuracy on the 629 test reports: " + ", ".join(parts) +
                   (f"; group frequency only {f(cs['prior_only']['top1'], 3)}" if "prior_only" in cs else "") + ".")
    for p in nums.get("paired") or []:
        if p["verdict"] in ("supported", "weak") and p["model"] in D.main_models:
            dirn = "less robust than" if p["diff_mean"] > 0 else "more robust than"
            if p["verdict"] == "weak":
                dirn = "tends to be " + dirn
            c = {"mean": p["diff_mean"], "lo": p["diff_lo"], "hi": p["diff_hi"]}
            out.append(f"{SHORT.get(p['model'], p['model'])} {dirn} ThreatMAMBA on \"{p['source']}\": flip_c difference "
                       f"{fdiff(c)} over {p['diff_n_docs']} reports both attribute correctly "
                       f"({VERDICT_DIFF[p['verdict']]}).")
    same = [p for p in nums.get("paired") or [] if p["verdict"] == "not_supported" and p["model"] in D.main_models]
    if same:
        out.append("No difference from ThreatMAMBA (CI contains 0 and |difference| < 0.01): " +
                   "; ".join(f"{SHORT.get(p['model'], p['model'])} – {p['source']}" for p in same) + ".")
    return out


def sec_notes(D: Data, L: list):
    L.append("## Notes and limitations\n")
    notes = [
        "No 2025–2026 work reports results on CTI2Attacker, so every comparison number here comes from our own "
        "training and evaluation; the numbers are not directly comparable with those in each system's original paper.",
        "TTP-Vec reaches its best accuracy in its paper with victim features (country, sector); CTI2Attacker has no "
        "such fields, so only the TTP-only configuration is used (the paper reports 55–66% for it). Techniques come from "
        "ThreatMAMBA's TTP extractor instead of ChatGPT + manual review; rarity weights and the scaler are fitted on train only.",
        "TTP-Match uses OpenAI embeddings and (optionally) GPT-4 HyDE; here an open embedding model runs locally and "
        "HyDE is not used. The top-k/threshold/prior configuration is selected on a validation split of train; table 1 "
        "also lists the exact paper configuration.",
        "The 144-report benchmark is selected with ThreatMAMBA (all three variants correct) → this favours ThreatMAMBA "
        "on clean accuracy; the robustness tables use flip_c (documents the model gets right) and section 7 uses the intersection to reduce this bias.",
        "The strongest variants of the worst/llm runs are optimised for ThreatMAMBA → transfer attacks for the other systems. "
        "The fair worst-case comparison is section 4.2 / the \"both adaptive\" row in section 7.",
        "Every rate is computed on valid variants; 95% CIs use a document-level bootstrap (1000 resamples); same verdict rule as ReTrace.",
    ]
    L += [f"- {n}" for n in notes]
    L.append("")


# ------------------------------------------------------------------ figures
COLORS = {"main": "#2a78d6", "ttp_match": "#eb6834", "ttp_vec_knn": "#1baf7a", "ttp_vec_svm": "#eda100",
          "ttp_vec_rf": "#e87ba4", "ttp_vec_dt": "#008300", "ttp_vec_nb": "#4a3aa7", "no_cl": "#e34948",
          "no_mamba": "#52514e"}


def figures(D: Data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "axes.grid.axis": "y", "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "font.size": 9, "axes.axisbelow": True})
    made = []
    ms = D.main_models
    if "rand" in D.rows:
        rs = D.sel("rand", correct=True, strategy="random", level=lambda x: x in (1, 2))
        ops = sorted({r["op"] for r in rs}, key=lambda o: (o[0] != "T", int(o[1:]) if o[1:].isdigit() else 0))
        w = 0.8 / max(1, len(ms))
        fig, ax = plt.subplots(figsize=(max(7, len(ops) * 0.75), 3.6))
        for i, m in enumerate(ms):
            ys, lo, hi = [], [], []
            for op in ops:
                c = ci([r for r in rs if r["model"] == m and r["op"] == op], "flip")
                ys.append(c["mean"] if c["mean"] is not None else np.nan)
                lo.append((c["mean"] - c["lo"]) if c["mean"] is not None else 0)
                hi.append((c["hi"] - c["mean"]) if c["mean"] is not None else 0)
            ax.bar(np.arange(len(ops)) + i * w, ys, w * 0.92, yerr=[lo, hi], capsize=1.5,
                   color=COLORS.get(m, "#888"), label=SHORT.get(m, m),
                   error_kw={"elinewidth": 0.6, "ecolor": "#52514e", "capthick": 0.6})
        ax.set_xticks(np.arange(len(ops)) + w * (len(ms) - 1) / 2); ax.set_xticklabels(ops)
        ax.set_ylabel("flip_c (CI 95%)")
        ax.set_title("Random variants, 100% timeline (T = level 1, E = level 2)", pad=26)
        ax.legend(frameon=False, ncol=min(5, len(ms)), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"flip_by_op_random.{ext}", dpi=200)
        plt.close(fig); made.append("flip_by_op_random")
        # level-2 timeline
        lvr = [r for r in D.rows["rand"] if r["strategy"] == "random" and r["level"] == 2]
        lv = aggregate(lvr, ("model", "keep"), metrics=("flip", "acc_drop"))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for m in ms:
            d = sorted([r for r in lv if r["model"] == m], key=lambda r: r["keep"])
            if not d:
                continue
            for ax, key in zip(axes, ("flip", "acc_drop")):
                ax.plot([int(r["keep"] * 100) for r in d], [r[key] for r in d], marker="o", markersize=4,
                        linewidth=1.6, color=COLORS.get(m, "#888"), label=SHORT.get(m, m))
        axes[0].set_ylabel("flip (all documents)"); axes[1].set_ylabel("Δacc")
        for ax in axes:
            ax.set_xlabel("% timeline")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Level 2, random variants, by timeline cut-off")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"timeline_level2.{ext}", dpi=200)
        plt.close(fig); made.append("timeline_level2")
    return made


# ------------------------------------------------------------------ main
def build(cfg, fmt="dec") -> tuple[str, dict]:
    FMT["mode"] = fmt
    D = Data(cfg)
    nums: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "fmt": fmt, "models": D.models,
                  "main_models": D.main_models, "runs": list(D.rows), "adaptive": list(D.adapt)}
    L = ["# Robustness comparison: ThreatMAMBA vs TTP-Match vs TTP-Vec (ReTrace, levels 1–2)\n",
         f"_Generated automatically at {nums['generated']} by `python -m cta_systems.evaluation.report`. Runs: "
         f"{', '.join(D.rows) or '(none)'}; adaptive attacks: {', '.join(D.adapt) or '(not run)'}._\n"]
    if not D.rows:
        L.append("No data yet: run `bash scripts/run_cta_systems.sh` first.\n")
        return "\n".join(L), nums
    for fn in (sec_setup, sec_clean, sec_level1, sec_level2, sec_worst, sec_asr, sec_timeline, sec_paired,
               sec_group, sec_stage, sec_shortcuts, sec_extra):
        try:
            fn(D, L, nums)
        except Exception as e:  # noqa: BLE001
            log(f"[report] {fn.__name__} error: {type(e).__name__}: {e}")
            L.append(f"_({fn.__name__} error: {type(e).__name__}: {e})_\n")
    sec_notes(D, L)
    summ = summary_lines(D, nums)
    if summ:
        idx = next(i for i, x in enumerate(L) if x.startswith("## 0."))
        L[idx:idx] = ["## Summary of the main numbers\n"] + [f"- {x}" for x in summ] + [""]
    try:
        made = figures(D)
        if made:
            L.append("Figures: " + ", ".join(f"`figures/{m}.png`" for m in made) + "\n")
    except Exception as e:  # noqa: BLE001
        log(f"[report] figure error: {e}")
    return "\n".join(L) + "\n", nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=["dec", "pct"], default="dec")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    md, nums = build(cfg, args.fmt)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "REPORT_CTA_SYSTEMS.md" if args.fmt == "dec" else "REPORT_CTA_SYSTEMS_pct.md"
    (OUT / name).write_text(md, encoding="utf-8")
    save_json(OUT / "report_numbers.json", nums)
    log(f"-> {OUT / name} + report_numbers.json + tables/report_*.csv + figures/")


if __name__ == "__main__":
    main()
