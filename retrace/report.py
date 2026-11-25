"""Generate the ReTrace summary report (Markdown) from the completed runs.

    python -m retrace.report                      # default runs: rand, worst, llm
    python -m retrace.report --runs rand,worst

Writes outputs/retrace/REPORT.md and outputs/retrace/report_numbers.json (every number in the
report, for reuse in papers). Figures are referenced relative to runs/<run>/figures/*.png.
"""
import argparse
import json
import time

from .common import ADV_DIR, iter_jsonl_any, log
from .metrics import aggregate, build_table, run_exists, targeted_asr, validity_rates

OP_DESC = {
    "ID": "Rebuild unchanged (deterministic control)",
    "T1": "LLM paraphrases sentences, keeping meaning and IOCs",
    "T2": "LLM splits/merges sentences, active ↔ passive voice",
    "T3": "Abbreviation ↔ full form (RDP ↔ Remote Desktop Protocol)",
    "T4": "Move the IOC appendix to the top of the text",
    "T5": "Insert benign sentences (no IOC, no group/tool names)",
    "T6": "Change the date format (negative control)",
    "E1": "Mask k IOCs with a generic phrase",
    "E2": "Drop, or LLM-neutralise, the sentences carrying k techniques",
    "E3": "Rename tools to another alias of the same ATT&CK software",
    "E4": "Mask the true group name (scope=true) or all 20 group names (scope=all)",
    "E5": "Remove temporal connectives (markers) / swap evidence sentences (swap)",
    "E6": "LLM generalises tool names/commands",
    "E7": "Insert k valid but unrelated IOCs",
    "E8": "False flag: insert IOCs / TTP sentences / the name of group B (runner-up)",
    "G1": "Delete nodes by type (IOC / Technique / Tactic)",
    "G2": "Delete or rewire IOC–TTP edges",
    "G3": "Reorder States (reverse / shuffle / swap1)",
    "G4": "Insert noise nodes from another group",
    "G5": "Shuffle edge relation types",
}
LEVEL_NAME = {0: "0 · control", 1: "1 · text", 2: "2 · evidence", 3: "3 · graph (diagnostic)"}
EXPECT_NAME = {"INV": "INV — the label must not change", "DIR": "DIR — p may drop, no confident flip",
               "TGT": "TGT — must not be steered to group B"}
VERDICT_TEXT = {"supported": "**Supported** (CI excludes 0 and effect ≥ 0.01)",
              "weak": "Weak (only one of: CI excludes 0 / effect ≥ 0.01)",
              "not_supported": "Not supported (effect < 0.01 and CI contains 0)", "na": "Not enough data"}


def _j(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _ci(r, m="flip"):
    if not r or r.get(m) is None:
        return "–"
    return f"{r[m]:.2f} [{r[f'{m}_lo']:.2f}, {r[f'{m}_hi']:.2f}]"


def _c(c):
    if not c or c.get("mean") is None:
        return "–"
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def _llm_name():
    try:
        from .config import get_config
        return get_config().get("llm", {}).get("model", "?")
    except Exception:  # noqa: BLE001
        return "?"


def _best(rows, key="flip", model="main", n=3, filt=None):
    rs = [r for r in rows if r.get("model") == model and (filt is None or filt(r)) and r.get(key) is not None]
    return sorted(rs, key=lambda r: -r[key])[:n]


def _summary(nums, R):
    """Summary of the main numbers (numbers only, no over-interpretation)."""
    out = []
    san = nums.get("sanity", {}).get("ID", {})
    par = nums.get("parity") or {}
    if san:
        mx = max((v.get("max_dp", 0) for v in (par.get("pred") or {}).values()), default=None)
        out.append(f"The pipeline is deterministic and faithful to the reproduction: `ID` flip = "
                   f"{max(san.values()):.3f} for every model" + (f"; parity max |Δp| = {mx:.1e}, no label change" if mx is not None else "") + ".")
    rf = nums.get("rand_full") or []
    if rf:
        b = _best(rf, n=3, filt=lambda r: r["op"] != "ID")
        out.append("Random perturbations (main, 100%): highest flip for " +
                   ", ".join(f"{r['op']} {r['flip']:.2f} [{r['flip_lo']:.2f}, {r['flip_hi']:.2f}]" for r in b) + ".")
    wf = nums.get("worst_full") or []
    if wf:
        b = _best(wf, n=3)
        out.append("Strongest black-box attack (main, 100%): " +
                   ", ".join(f"{r['op']} ({r['strategy']}) {r['flip']:.2f} [{r['flip_lo']:.2f}, {r['flip_hi']:.2f}]" for r in b) + ".")
    wa = [r for r in (nums.get("worst_asr") or []) if r["model"] == "main"]
    if wa:
        t = max(wa, key=lambda r: r["targeted_asr"])
        out.append(f"Targeted false flags steer {t['targeted_asr']*100:.0f}% [{t['asr_lo']*100:.0f}, "
                   f"{t['asr_hi']*100:.0f}] of variants to the decoy group B (trace type `{t['kind']}`, main).")
    lf = nums.get("llm_full") or []
    if lf:
        b = _best(lf, n=2, filt=lambda r: r["op"] in ("T1", "T2"))
        if b:
            out.append("Meaning-preserving LLM rewrites (INV) still flip: " +
                       ", ".join(f"{r['op']} {r['params']} {r['flip']:.2f} [{r['flip_lo']:.2f}, {r['flip_hi']:.2f}]" for r in b) + ".")
    lv = nums.get("rand_by_level") or []
    l2 = {r["keep"]: r for r in lv if r["level"] == 2 and r["model"] == "main"}
    acc0 = {a["keep"]: a["acc_orig"] for a in (nums.get("base_acc") or []) if a["model"] == "main"}
    if 0.2 in l2 and 1.0 in l2:
        a, b = l2[0.2], l2[1.0]
        txt = (f"Early cut-offs are less stable: evidence perturbations (level 2) flip {a['flip']:.2f} at 20% of the "
               f"timeline vs {b['flip']:.2f} at 100% (main)")
        if a.get("acc_drop") is not None:
            txt += (f", but accuracy only drops by {a['acc_drop']:.3f} at 20% (originals correct: "
                    f"{acc0.get(0.2, float('nan')):.3f}) vs {b['acc_drop']:.3f} at 100% — most early flips "
                    "move from one wrong group to another")
        out.append(txt + ".")
    bgs = nums.get("by_group") or {}
    for lab in ("Level 2 (strongest attack)", "Level 2 (random)"):
        s = next((y for y in (bgs.get(lab) or {}).get("summary", []) if y["model"] == "main"), None)
        if s:
            out.append(f"Per group ({lab.lower()}, main, 100%): mean over documents {s['micro']:.3f}, unweighted "
                       f"mean over {s['n_groups']} groups {s['macro']:.3f} [{s['macro_lo']:.3f}, {s['macro_hi']:.3f}]; least "
                       f"robust {s['max_group']} ({s['max_value']:.3f}), most robust {s['min_group']} ({s['min_value']:.3f}).")
            break
    vs = []
    for key, name in (("S1_size", "S1"), ("S2_ttp_fp", "S2")):
        if nums.get(key):
            vs.append(f"{name}: {nums[key].get('verdict')}")
    if nums.get("S3_name"):
        vs.append(f"S3a: {nums['S3_name'].get('verdict_mask')}, S3b: {nums['S3_name'].get('verdict_inject')}")
    for key, name in (("S4_order", "S4"), ("S5_sink", "S5"), ("S6_specificity", "S6")):
        if nums.get(key):
            vs.append(f"{name}: {nums[key].get('verdict')}")
    if vs:
        out.append("Shortcut verdicts (supported/weak/not_supported/na): " + "; ".join(vs) + " — details in section 4.")
    return out


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _models_of(rows):
    order = ["main", "no_cl", "no_mamba"]
    ms = sorted({r["model"] for r in rows}, key=lambda m: (order.index(m) if m in order else 9, m))
    return ms


def build_report(runs: list[str]) -> tuple[str, dict]:
    from .analysis import (compute_by_group, compute_curves, compute_name_leak, compute_order, compute_sink,
                           compute_specificity, compute_stage_decomp, compute_ttp_fp)
    from .operators import REGISTRY
    runs = [r for r in runs if run_exists(r)]
    # role of each run by name prefix: rand* (random), worst* (strongest attack), llm*
    R = {role: next((r for r in runs if r.startswith(role)), None) for role in ("rand", "worst", "llm")}
    nums: dict = {"runs": runs, "roles": R}
    L: list[str] = []
    L.append("# ReTrace report — adversarial robustness of CTA models\n")
    L.append(f"_Generated automatically at {time.strftime('%Y-%m-%d %H:%M')} by `python -m retrace.report`. "
             f"Runs used: {', '.join(runs) if runs else '(no runs yet)'}._\n")
    L.append("**Threat model.** The attacker does not touch the model or its training data; they only change "
             "the traces they leave, so the CTI reports written about them (and the CSKGs built from those "
             "reports) change accordingly. Level 1 rewrites the text but keeps its meaning; level 2 masks or "
             "inserts evidence; level 3 edits the graph directly and is diagnostic only. Every operator uses "
             "only the model's output scores (black-box).\n")

    # ------------------------------------------------------------------ 0. setup
    L.append("## 0. Setup and health checks\n")
    fr = _j(ADV_DIR / "freeze.json") or {}
    par = _j(ADV_DIR / "parity.json") or {}
    elig = list(iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl"))
    groups_el = sorted({r["group"] for r in elig})
    nums["eligible"] = {"n_docs": len(elig), "n_groups": len(groups_el),
                        "name_in_graph": sum(r.get("name_in_graph", 0) for r in elig)}
    L.append(f"- Commit: `{str(fr.get('git_commit') or 'no git')[:12]}`; checkpoints are SHA-256 hashed in `freeze.json` "
             f"(preflight: {'PASS' if fr.get('passed') else 'not run/FAIL'}).")
    L.append(f"- Eligible set (real CTI, all 3 models correct at 100%): **{len(elig)} documents / "
             f"{len(groups_el)} groups**; at most 25 documents per group, seeded random sample. "
             f"The group name appears in an evidence sentence in {nums['eligible']['name_in_graph']}/{len(elig)} documents.")
    if par:
        pred = par.get("pred", {})
        ptxt = "; ".join(f"{m}: max |Δp| = {v['max_dp']:.1e}, label changes = {v['label_changes']}" for m, v in pred.items())
        L.append(f"- Parity (rebuilt CSKG vs `data/cskg`): {par.get('struct_ok')}/{par.get('n')} structurally identical"
                 f"{' — ' + ptxt if ptxt else ''}. Result: {'PASS' if par.get('passed') else 'FAIL'}.")
        nums["parity"] = {k: par.get(k) for k in ("n", "struct_ok", "passed", "pred")}
    base = runs[0] if runs else None
    if base:
        rows = [r for r in build_table(base) if r["valid"] and r["keep"] == 1.0]
        for op in ("ID", "T6"):
            rs = [r for r in rows if r["op"] == op]
            if rs:
                agg = aggregate(rs, ("model",))
                txt = ", ".join(f"{a['model']} {a['flip']:.3f}" for a in agg)
                L.append(f"- Flip rate of `{op}` (run {base}): {txt} "
                         f"{'(must be 0 — passed)' if op == 'ID' and all(a['flip'] == 0 for a in agg) else ''}")
                nums.setdefault("sanity", {})[op] = {a["model"]: a["flip"] for a in agg}
    L.append("")

    # ------------------------------------------------------------------ 1. taxonomy
    L.append("## 1. Perturbation taxonomy\n")
    counts, valid = {}, {}
    for run in runs:
        for v in validity_rates(run):
            valid.setdefault(v["op"], []).append((v["validity_rate"], v["n"]))
        vids = {}
        for r in build_table(run):
            if r["valid"]:
                vids.setdefault(r["op"], set()).add(r["vid"])
        for op, st in vids.items():
            counts[op] = counts.get(op, 0) + len(st)
    trs = []
    for code in ["ID", "T1", "T2", "T3", "T4", "T5", "T6", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
                 "G1", "G2", "G3", "G4", "G5"]:
        cls = REGISTRY.get(code)
        if cls is None:
            continue
        vr = valid.get(code)
        vtxt = (f"{sum(a * n for a, n in vr) / max(1, sum(n for _a, n in vr)):.2f}" if vr else "–")
        trs.append([code, LEVEL_NAME.get(cls.level, cls.level), cls.expect, OP_DESC.get(code, ""),
                    counts.get(code, 0), vtxt])
    L.append(_table(["Code", "Level", "Expectation", "Description", "Valid variants", "Validity rate"], trs))
    L.append("\nExpectations: " + "; ".join(EXPECT_NAME.values()) + ". Variants that fail the validity checks "
             "(`validate.py`) are excluded from every table.\n")
    nums["taxonomy_counts"] = counts

    # ------------------------------------------------------------------ 2. benchmark
    L.append("## 2. Benchmark\n")
    brs = []
    for run in runs:
        tb = build_table(run)
        brs.append([run, len({r['doc_id'] for r in tb}), len({r['vid'] for r in tb}),
                    ", ".join(_models_of(tb)), ", ".join(f"{int(k*100)}%" for k in sorted({r['keep'] for r in tb})),
                    ", ".join(sorted({r['strategy'] for r in tb}))])
    if brs:
        L.append(_table(["Run", "Documents", "Variants", "Models", "Timeline cut-offs", "Strategies"], brs))
    L.append("\nEvery variant is stored in `runs/<run>/variants.jsonl.gz` (text, parameters, check results, "
             "changes in the extraction stage); every (variant × model × cut-off) is one row of `scores.jsonl.gz`. "
             "Seeds are fixed (`configs/retrace.yaml`, `PYTHONHASHSEED=0`), so re-runs give the same variants.\n")

    # ------------------------------------------------------------------ 3. sensitivity
    L.append("## 3. Sensitivity by evidence type and timeline cut-off\n")
    if R["rand"]:
        rr = R["rand"]
        tb = [r for r in build_table(rr) if r["valid"]]
        ms = _models_of(tb)
        full = aggregate([r for r in tb if r["keep"] == 1.0], ("op", "expect", "model"))
        L.append("### 3.1 Random perturbations, 100% timeline — flip rate [95% CI] by operator and model\n")
        rows_t = []
        for op in sorted({r["op"] for r in full}, key=lambda o: (o[0] != "I", o)):
            cells = []
            exp = ""
            for m in ms:
                r = next((x for x in full if x["op"] == op and x["model"] == m), None)
                cells.append(_ci(r))
                exp = r["expect"] if r else exp
            rm = next((x for x in full if x["op"] == op and x["model"] == ms[0]), None)
            rows_t.append([op, exp, rm["n_docs"] if rm else "–"] + cells + [_ci(rm, "violation"), _ci(rm, "delta_p")])
        L.append(_table(["Operator", "Expectation", "Documents"] + [f"flip {m}" for m in ms] +
                        [f"violation ({ms[0]})", f"Δp ({ms[0]})"], rows_t))
        L.append("\nflip = top-1 label differs from the original; violation = the operator's expectation is "
                 "violated; Δp = drop in the probability of the true group. Figures: "
                 f"`runs/{rr}/figures/violation_by_op.png`, `runs/{rr}/figures/heatmap_op_keep_main.png`.\n")
        nums["rand_full"] = full
        cur = compute_curves(rr)
        by_k = [r for r in cur["by_k_fixed"] if r["model"] == "main"]
        if by_k:
            L.append("### 3.2 By amount of perturbed evidence (k), model main, 100% timeline\n")
            L.append(_table(["Operator", "k", "flip [CI]", "Δp [CI]", "Documents"],
                            [[r["op"], r["k"], _ci(r), _ci(r, "delta_p"), r["n_docs"]] for r in by_k]))
            L.append("\nEach operator is computed on the **same document set** at every k (k values covering fewer "
                     f"than half of the documents are dropped), so the curve is not distorted by a shrinking set. Figure: `runs/{rr}/figures/flip_vs_k_main.png`.\n")
            nums["rand_by_k"] = by_k
        lv = cur["by_level"]
        if lv:
            L.append("### 3.3 By timeline cut-off — flip / Δacc per level and model\n")
            keeps = sorted({r["keep"] for r in lv})

            def _fa(r):
                return f"{r['flip']:.3f} / {r['acc_drop']:.3f}" if r.get("acc_drop") is not None else f"{r['flip']:.3f}"
            rows_l = []
            acc0 = {(a["model"], a["keep"]): a["acc_orig"] for a in cur.get("base_acc", [])}
            for m in ms:
                rows_l.append(["Accuracy of originals", m] +
                              [f"{acc0[(m, k)]:.3f}" if (m, k) in acc0 else "–" for k in keeps])
            for level in sorted({r["level"] for r in lv if r["level"] not in (0, None)}):
                for m in ms:
                    rs = {r["keep"]: r for r in lv if r["level"] == level and r["model"] == m}
                    rows_l.append([LEVEL_NAME.get(level, level), m] + [_fa(rs[k]) if k in rs else "–" for k in keeps])
            if R["llm"]:
                lvl = compute_curves(R["llm"])["by_level"]
                for m in ms:
                    rs = {r["keep"]: r for r in lvl if r["level"] == 1 and r["model"] == m}
                    if rs:
                        rows_l.append(["1 · text (LLM: T1, T2)", m] + [_fa(rs[k]) if k in rs else "–" for k in keeps])
            L.append(_table(["Level", "Model"] + [f"{int(k*100)}%" for k in keeps], rows_l))
            L.append("\nEach cell: **flip / Δacc**. flip = top-1 label change rate; Δacc = top-1 accuracy of the originals "
                     "minus that of the variants (same model, same cut-off). At 100% every eligible original is correct, "
                     "so Δacc = flip; at early cut-offs many originals are already wrong (row “Accuracy of originals”) "
                     "and most flips move between wrong groups, so Δacc is much smaller than flip: a high early flip "
                     "rate reflects **unstable** predictions, not that the attack removes more correct predictions "
                     "(negative Δacc = the variants are right more often than the originals). "
                     "Cut-off x% = keep only the first x% of States. Insertion operators "
                     "(T5, E7, E8) shift the cut position, so at early cut-offs part of the flips is a mechanical effect of the cut. "
                     f"Figure: `runs/{rr}/figures/flip_vs_timeline_by_level.png`.\n")
            nums["rand_by_level"] = lv
            nums["base_acc"] = cur.get("base_acc")
        asr = [r for r in targeted_asr(rr) if r["keep"] == 1.0]
        if asr:
            L.append("### 3.4 E8 false flags (random) — rate of being steered to group B\n")
            L.append(_table(["Trace type", "Model", "ASR [CI]", "n"],
                            [[r["kind"], r["model"], f"{r['targeted_asr']:.3f} [{r['asr_lo']:.3f}, {r['asr_hi']:.3f}]",
                              r["n"]] for r in asr]))
            L.append("")
            nums["rand_e8_asr"] = asr
    if R["worst"]:
        rw = R["worst"]
        tb = [r for r in build_table(rw) if r["valid"] and r["keep"] == 1.0]
        ms = _models_of(tb)
        full = aggregate(tb, ("op", "strategy", "model"))
        L.append("### 3.5 Strongest attacks (greedy over k / best-of-N / targeted), 100% timeline\n")
        L.append("Variants are selected with the scores of **main** (black-box); the other models are scored on "
                 "the same variants (transferability).\n")
        rows_w = []
        for op, st in sorted({(r["op"], r["strategy"]) for r in full}):
            cells = [_ci(next((x for x in full if x["op"] == op and x["strategy"] == st and x["model"] == m), None))
                     for m in ms]
            rm = next((x for x in full if x["op"] == op and x["strategy"] == st and x["model"] == ms[0]), None)
            rows_w.append([op, st, rm["n_docs"] if rm else "–"] + cells + [_ci(rm, "delta_p")])
        L.append(_table(["Operator", "Strategy", "Documents"] + [f"flip {m}" for m in ms] + [f"Δp ({ms[0]})"], rows_w))
        wk = [r for r in compute_curves(rw)["by_k_fixed"] if r["model"] == "main"]
        if wk:
            L.append("\nStrongest attack by k (main, same document set at every k):\n")
            L.append(_table(["Operator", "k", "flip [CI]", "Δp [CI]", "Documents"],
                            [[r["op"], r["k"], _ci(r), _ci(r, "delta_p"), r["n_docs"]] for r in wk]))
            nums["worst_by_k"] = wk
        asr = [r for r in targeted_asr(rw) if r["keep"] == 1.0]
        nums["worst_asr"] = asr
        if asr:
            L.append("\nE8 targeted — rate of being steered to group B:\n")
            L.append(_table(["Trace type", "Model", "ASR [CI]", "n"],
                            [[r["kind"], r["model"], f"{r['targeted_asr']:.3f} [{r['asr_lo']:.3f}, {r['asr_hi']:.3f}]",
                              r["n"]] for r in asr]))
        L.append("")
        nums["worst_full"] = full
    if R["llm"]:
        rl = R["llm"]
        tb = [r for r in build_table(rl) if r["valid"] and r["keep"] == 1.0]
        ms = _models_of(tb)
        full = aggregate(tb, ("op", "params", "model"))
        L.append("### 3.6 LLM operators (level 1 and E2 neutral / E6), 100% timeline\n")
        rows_m = []
        for op, pk in sorted({(r["op"], r["params"]) for r in full}):
            cells = [_ci(next((x for x in full if x["op"] == op and x["params"] == pk and x["model"] == m), None))
                     for m in ms]
            rm = next((x for x in full if x["op"] == op and x["params"] == pk and x["model"] == ms[0]), None)
            rows_m.append([op, pk, rm["n_docs"] if rm else "–"] + cells +
                          [_ci(rm, "violation"), _ci(rm, "delta_p")])
        L.append(_table(["Operator", "Parameters", "Documents"] + [f"flip {m}" for m in ms] +
                        [f"violation ({ms[0]})", f"Δp ({ms[0]})"], rows_m))
        L.append("\nT1/T2 are **meaning-preserving** operators (expectation INV), so every flip is a violation. Variants "
                 f"are generated by the LLM `{_llm_name()}`, with IOCs/names/technique IDs protected by placeholders.")
        nli = _j(ADV_DIR / "runs" / rl / "tables" / "nli.json")
        if nli and nli.get("by_op"):
            passed = set(nli.get("pass_vids", []))
            rows_n = []
            for op in sorted(nli["by_op"]):
                x = nli["by_op"][op]
                sub = [r for r in tb if r["op"] == op and r["model"] == ms[0]]
                sub_ok = [r for r in sub if r["vid"] in passed]
                a_all = aggregate(sub, ("op",))[0] if sub else None
                a_ok = aggregate(sub_ok, ("op",))[0] if sub_ok else None
                rows_n.append([op, x["n"], f"{x['pass_rate']*100:.0f}%", f"{x['median_nli']:.2f}",
                               _ci(a_all), _ci(a_ok)])
            L.append(f"\nAutomatic semantic-equivalence check (bidirectional NLI, `{nli.get('model')}`, threshold "
                     f"{nli.get('threshold')}): flip of {ms[0]} on all variants and on NLI-passing variants only.\n")
            L.append(_table(["Operator", "Variants", "NLI pass", "Median entailment", f"flip {ms[0]} (all)",
                             f"flip {ms[0]} (NLI pass only)"], rows_n))
            nums["llm_nli"] = nli.get("by_op")
        vr = validity_rates(rl)
        if vr:
            L.append("\nValidity rate of LLM variants: " +
                     ", ".join(f"{v['op']} {v['validity_rate']:.2f} (n={v['n']})" for v in vr) + "\n")
        nums["llm_full"] = full

    # ------------------------------------------------------------------ 3.7 per group
    src = []
    if R["rand"]:
        src += [("Level 1 (random)", R["rand"], 1), ("Level 2 (random)", R["rand"], 2),
                ("Level 3 (random)", R["rand"], 3)]
    if R["worst"]:
        src += [("Level 2 (strongest attack)", R["worst"], 2)]
    if R["llm"]:
        src += [("Level 1 LLM (T1, T2)", R["llm"], 1)]
    bg = {run: compute_by_group(run) for run in {s[1] for s in src}}
    src = [s for s in src if any(x["level"] == s[2] and x["model"] == "main" for x in bg[s[1]]["summary"])]
    if src:
        L.append("### 3.7 Robustness per source group and unweighted mean over groups (main, 100% timeline)\n")
        n_el = {}
        for r in elig:
            n_el[r["group"]] = n_el.get(r["group"], 0) + 1
        groups_all = sorted({x["group"] for run in bg for x in bg[run]["per_group"] if x["model"] == "main"},
                            key=lambda g: (-n_el.get(g, 0), g))
        rows_g = []
        for g in groups_all:
            cells = []
            for _lab, run, level in src:
                x = next((y for y in bg[run]["per_group"] if y["model"] == "main" and y["level"] == level
                          and y["group"] == g), None)
                if x is None:
                    cells.append("–")
                elif run == R["llm"] and R["rand"] and x["n_docs"] != n_el.get(g):
                    cells.append(f"{x['flip']:.3f} (n={x['n_docs']})")
                else:
                    cells.append(f"{x['flip']:.3f}")
            rows_g.append([g, n_el.get(g, "–")] + cells)
        for lab_s, key in (("**Mean over documents (micro)**", "micro"), ("**Unweighted mean over groups (macro)**", "macro")):
            cells = []
            for _lab, run, level in src:
                s = next((y for y in bg[run]["summary"] if y["model"] == "main" and y["level"] == level), None)
                cells.append(f"{s[key]:.3f} [{s[key + '_lo']:.3f}, {s[key + '_hi']:.3f}]" if s else "–")
            rows_g.append([lab_s, len(elig)] + cells)
        L.append(_table(["Group", "Documents"] + [s[0] for s in src], rows_g))
        L.append("\nEach cell is the mean flip rate over the documents of that group. **micro** = mean over documents "
                 "(as in the other tables; groups with more documents weigh more); **macro** = every group weighs the "
                 "same (bootstrap CI stratified by group). A gap between the two means that the result depends on "
                 "the group mix of the document set — which is why the pilot and the full run differ. Groups with fewer "
                 "than 5 documents should be read as examples only (the macro CI does not capture their full "
                 "uncertainty). Column “(n=…)” for LLM operators: number of the group's documents in the pilot set. "
                 "Full tables: `runs/<run>/tables/by_group.csv`, `by_group_summary.csv`.\n")
        rows_s = []

        def _m5(s):
            if s.get("macro_min5") is None:
                return "–"
            return (f"{s['macro_min5']:.3f} [{s['macro_min5_lo']:.3f}, {s['macro_min5_hi']:.3f}] "
                    f"({s['n_groups_min5']} groups)")
        for lab, run, level in src:
            for s in [y for y in bg[run]["summary"] if y["level"] == level]:
                rows_s.append([lab, s["model"], f"{s['micro']:.3f} [{s['micro_lo']:.3f}, {s['micro_hi']:.3f}]",
                               f"{s['macro']:.3f} [{s['macro_lo']:.3f}, {s['macro_hi']:.3f}] ({s['n_groups']} groups)",
                               _m5(s), f"{s['max_group']} {s['max_value']:.3f}", f"{s['min_group']} {s['min_value']:.3f}"])
        L.append("Summary for all 3 models (macro ≥5 = only groups with at least 5 documents, so that a group with "
                 "1–2 documents cannot dominate the mean):\n")
        L.append(_table(["Source", "Model", "micro [CI]", "macro [CI]", "macro ≥5 documents [CI]",
                         "Least robust group", "Most robust group"], rows_s))
        L.append("")
        nums["by_group"] = {lab: {"run": run, "level": level,
                                  "summary": [y for y in bg[run]["summary"] if y["level"] == level],
                                  "per_group_main": [y for y in bg[run]["per_group"]
                                                     if y["level"] == level and y["model"] == "main"]}
                            for lab, run, level in src}

    # ------------------------------------------------------------------ 4. shortcuts
    L.append("## 4. Root causes and shortcuts\n")
    base = R["rand"] or (runs[0] if runs else None)
    sc_rows = []
    size = _j(ADV_DIR / "analysis" / "size_probe.json")
    if size and size.get("rows"):
        o, n = size["rows"][0], size["rows"][1]
        sc_rows.append(["S1 · Graph size in contrastive learning",
                        f"Old splice negatives are {o['size_ratio']:.2f}x larger than positives; a classifier that only "
                        f"counts nodes separates negatives/positives with {o['probe_acc_cv5']:.3f} (CV5), and with "
                        f"same-size negatives (Eq. 8) {n['probe_acc_cv5']:.3f} (0.5 = chance)",
                        {"supported": "**Size shortcut** (splice probe ≥ 0.70, matched ≤ 0.60)",
                         "weak": "Trend (splice probe ≥ 0.10 above matched)",
                         "not_supported": "No", "na": "Not enough data"}[size["verdict"]]])
        nums["S1_size"] = size
    fp = compute_ttp_fp()
    if fp.get("n_benign") is not None:
        sc_rows.append(["S2 · The TTP extractor labels benign sentences",
                        f"{fp['n_benign'] - fp['n_ttp_free']}/{fp['n_benign']} benign sentences receive ≥1 ATT&CK label "
                        f"({fp['fp_rate']:.2f}, Wilson CI {fp['fp_ci']}); most frequent labels: "
                        f"{', '.join(str(a) for a, _b in fp.get('top_labels', [])[:4]) or '–'}",
                        {"supported": "**Yes** (more than 20% of benign sentences are labelled)", "weak": "Trend",
                         "not_supported": "No", "na": "Not enough data"}[fp["verdict"]]])
        nums["S2_ttp_fp"] = fp
    if base:
        nl = compute_name_leak(base)
        nums["S3_name"] = nl
        mm = nl["models"].get("main") or next(iter(nl["models"].values()), None)
        if mm:
            txt_a = (f"E4 masks the true group name, name in an evidence sentence: Δp = {_c(mm['E4_true_dp_name_in_graph'])}, "
                     f"flip = {_c(mm['E4_true_flip_name_in_graph'])} (n_doc = {mm['E4_true_dp_name_in_graph'].get('n_docs')}); "
                     f"masking all 20 group names: Δp = {_c(mm['E4_all_dp'])}")
            sc_rows.append(["S3a · The model relies on the true group name (main)", txt_a,
                            VERDICT_TEXT[nl.get("verdict_mask", "na")]])
            txt_b = (f"E8 inserts B's name into an evidence sentence: p(B) rises by {_c(mm['E8_name_p_target_gain'])}, "
                     f"rate steered to B = {_c(mm['E8_name_asr'])}")
            if R["worst"]:
                nlw = compute_name_leak(R["worst"])["models"].get("main")
                if nlw:
                    txt_b += (f"; with the strongest (targeted) variant: p(B) rises by {_c(nlw['E8_name_p_target_gain'])}, "
                              f"rate steered = {_c(nlw['E8_name_asr'])}")
            sc_rows.append(["S3b · Another group's name pulls the prediction (main)", txt_b,
                            VERDICT_TEXT[nl.get("verdict_inject", "na")]])
        od = compute_order(base)
        nums["S4_order"] = od
        g3 = next((r for r in od["rows"] if r["op"] == "G3" and "reverse" in r["params"]), None)
        if g3 and g3.get("dp_main_minus_no_mamba") is not None:
            e5 = [r for r in od["rows"] if r["op"] == "E5"]
            e5t = "; ".join(f"E5 {r['params']}: flip main {r.get('flip_main', 0):.3f} / no_mamba {r.get('flip_no_mamba', 0):.3f}"
                            for r in e5)
            txt = (f"Reversing the State order (G3): flip main = {g3.get('flip_main', 0):.3f}, no_mamba = "
                   f"{g3.get('flip_no_mamba', 0):.3f}; Δp difference (main − no_mamba) = {g3['dp_main_minus_no_mamba']:.3f} "
                   f"[{g3['dp_diff_lo']:.3f}, {g3['dp_diff_hi']:.3f}], n_doc = {g3['n_docs']}" + (f". {e5t}" if e5t else ""))
            if R["llm"]:
                t2 = [r for r in compute_order(R["llm"])["rows"] if r["op"] == "T2"
                      and r.get("dp_main_minus_no_mamba") is not None]
                for r in t2:
                    txt += (f". T2 split/merge ({r['params']}): flip main {r.get('flip_main', 0):.3f} / no_mamba "
                            f"{r.get('flip_no_mamba', 0):.3f}, Δp difference = {r['dp_main_minus_no_mamba']:.3f} "
                            f"[{r['dp_diff_lo']:.3f}, {r['dp_diff_hi']:.3f}]")
            sc_rows.append(["S4 · Order sensitivity from the Mamba branch (main check: G3 reverse)", txt,
                            VERDICT_TEXT[od["verdict"]]])
        sk = compute_sink(base)
        nums["S5_sink"] = sk
        if sk["rows"] and sk["n_flips"] >= 20:
            top = sk.get("attractor") or sk["rows"][0]
            others = ", ".join(f"{r['group_to']} {r['share']*100:.0f}% (×{r['ratio']})" for r in sk["rows"][:4]
                               if r is not top and r.get("ratio"))
            txt = (f"{top['share']*100:.0f}% of {sk['n_flips']} flips (main, 100%) go to {top['group_to']}, "
                   f"while this group is only {top['share_in_eligible']*100:.1f}% of the eligible set"
                   + (f" ({top['ratio']}x)" if top.get("ratio") else "") + (f". Next: {others}" if others else ""))
            sc_rows.append(["S5 · Groups that “attract” wrong predictions", txt,
                            {"supported": "**Yes** (one group receives ≥20% of flips and ≥5x its share)",
                             "weak": "Trend (≥15% of flips, ≥2x its share)", "not_supported": "No",
                             "na": "Not enough data"}[sk["verdict"]]])
        else:
            sc_rows.append(["S5 · Groups that “attract” wrong predictions",
                            f"Only {sk['n_flips']} flips for main/100% (needs ≥ 20)", VERDICT_TEXT["na"]])
        spec_run = R["worst"] if R["worst"] and compute_specificity(R["worst"])["verdict"] != "na" else base
        sp = compute_specificity(spec_run)
        nums["S6_specificity"] = sp
        if sp.get("diff_spec_minus_common"):
            v = sp["verdict"]
            d = sp["diff_spec_minus_common"]
            r0, r1 = sp["rows"][0], sp["rows"][1]
            num = (f"Dropping sentences with specific techniques: Δp = {_c(r1)}; common techniques: Δp = {_c(r0)}; "
                   f"difference (specific − common) = {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]")
            txt = {"supported": "**Relies on specific techniques** (group-discriminative behaviour — not a shortcut)",
                   "weak": "Leans towards specific techniques but not conclusive (CI touches 0 or difference < 0.01)",
                   "not_supported": "No difference: masking specific or common techniques lowers p equally",
                   "na": "Not enough data"}[v]
            if d[2] < 0 and d[0] <= -0.01:     # common techniques hurt more than specific ones, significantly
                txt = "**Relies on common techniques** more than specific ones → sign of frequency reliance"
            sc_rows.append([f"S6 · Specific vs common TTPs (run {spec_run})", num, txt])
    if sc_rows:
        L.append(_table(["Hypothesis", "Numbers (95% document-level bootstrap CI)", "Verdict"], sc_rows))
        L.append("")
    if base:
        sd = compute_stage_decomp(base)
        if R["llm"]:
            for r in compute_stage_decomp(R["llm"])["rows"]:
                sd["rows"].append({**r, "op": f"{r['op']} (LLM)"})
        nums["stage_decomp"] = sd
        if sd["rows"]:
            L.append("### 4.1 Extraction stage or classification stage? (main, 100% timeline)\n")
            L.append(_table(["Operator", "flip", "IOC Jaccard", "Technique Jaccard", "ΔState", "% flips with UNCHANGED extraction"],
                            [[r["op"], f"{r['flip_rate']:.3f}", f"{r['ioc_jacc']:.2f}", f"{r['tech_jacc']:.2f}",
                              f"{r['d_states']:.2f}",
                              "–" if r["pct_flips_extract_same"] is None else f"{r['pct_flips_extract_same']:.2f}"]
                             for r in sd["rows"]]))
            L.append("\nJaccard = 1 means the IOC/technique set did not change. Flips with unchanged extraction come "
                     "from the sentence embeddings/classifier; the rest come from changes in the IOC/TTP extractors.\n")
    L.append("## 5. Notes\n")
    notes = []
    if elig and len(groups_el) < 20:
        notes.append(f"Only {len(groups_el)}/20 groups have eligible documents (the others have too few test documents "
                     "that all 3 models attribute correctly), so per-group conclusions only apply to these groups.")
    if not R["llm"]:
        notes.append(f"LLM operators (T1, T2, E2 neutral, E6) not run yet — they need Ollama + `{_llm_name()}`; see README.")
    else:
        n_llm = len({r["doc_id"] for r in build_table(R["llm"])})
        n_rand = len({r["doc_id"] for r in build_table(R["rand"])}) if R["rand"] else None
        if n_rand and n_llm < n_rand:
            notes.append(f"LLM operators ran on {n_llm} documents (other operators: {n_rand}). For full coverage, re-run with "
                         "`LLM_DOCS=eligible` (existing LLM results are reused from the cache).")
        notes.append("Validity of LLM variants is based on automatic checks only (IOCs, sentence count, length). "
                     "Add the NLI check (`python -m retrace.validate nli --run "
                     f"{R['llm']}`) and a human check: `python -m retrace.validate export-human --run "
                     f"{R['llm']} --n 150`, two annotators, then Cohen's κ.")
    notes.append("Every rate is computed on valid variants; 95% CIs use a document-level bootstrap (1000 resamples).")
    L += [f"- {n}" for n in notes]
    summ = _summary(nums, R)
    if summ:
        idx = next(i for i, x in enumerate(L) if x.startswith("## 0."))
        L[idx:idx] = ["## Summary of the main numbers\n"] + [f"- {x}" for x in summ] + [""]
    return "\n".join(L) + "\n", nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="rand,worst,llm")
    args = ap.parse_args()
    md, nums = build_report(args.runs.split(","))
    (ADV_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    (ADV_DIR / "report_numbers.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1, default=str),
                                                 encoding="utf-8")
    log(f"-> {ADV_DIR / 'REPORT.md'} + report_numbers.json")


if __name__ == "__main__":
    main()
