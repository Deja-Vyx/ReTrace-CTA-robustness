#!/usr/bin/env bash
# =============================================================================================
# Run the COMPLETE ReTrace evaluation of the ThreatMAMBA checkpoints with ONE command (GPU machine).
#
#   bash scripts/run_retrace.sh                 # run everything; finished steps are skipped
#   DOCS=pilot bash scripts/run_retrace.sh      # quick run on the 50 pilot documents (~15 min)
#   LLM=0 bash scripts/run_retrace.sh           # skip the LLM operators (default: auto-detect Ollama)
#   FORCE=1 bash scripts/run_retrace.sh         # re-run every step (existing runs are RESUMED, not deleted)
#
# Final result: outputs/retrace/REPORT.md (+ report_numbers.json, runs/*/tables, runs/*/figures)
# Per-step logs: outputs/retrace/logs/<step>.log . If interrupted, run the same command again.
# =============================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate

export PYTHONHASHSEED=0                         # stable IOC order from ioc-finder across runs
export PYTHONUNBUFFERED=1                       # flush logs immediately (no buffering through tee)
if [ -z "${RETRACE_TEST:-}" ]; then             # real run: no fake encoder/SSM/LLM
  unset THREATMAMBA_FAKE_ENCODER THREATMAMBA_SSM RETRACE_FAKE_LLM || true
fi

MODELS="${MODELS:-main,no_cl,no_mamba}"
DOCS="${DOCS:-eligible}"                        # eligible (~144 documents) | pilot (50)
LLM_DOCS="${LLM_DOCS:-pilot}"                   # LLM operators are slow -> 50 pilot documents by default
LLM="${LLM:-auto}"                              # auto | 1 | 0
SUF=""; [ "$DOCS" != "eligible" ] && SUF="_$DOCS"
R_RAND="rand$SUF"; R_WORST="worst$SUF"; R_LLM="llm$SUF"

OPS_RAND="ID,T3,T4,T5,T6,E1,E2,E3,E4,E5,E7,E8,G1,G2,G3,G4,G5"
OPS_GREEDY="E1,E2,E3,G1"                        # item-based operators -> greedy k=1..k_max
OPS_BON="T5,E4,E5,E7,E8,G3,G4"                  # the rest -> best-of-5 (E8: targeted)
OPS_LLM="T1,T2,E2,E6"

ADV=outputs/retrace; LOGD=$ADV/logs; STAMP=$ADV/.done
mkdir -p "$LOGD" "$STAMP"
T0=$(date +%s)

step() {   # step <name> <command...> : run, log, mark as done; skip if already done (unless FORCE=1)
  local name=$1; shift
  if [ -f "$STAMP/$name" ] && [ -z "${FORCE:-}" ]; then
    echo "[skip] $name (already done - FORCE=1 to re-run)"; return 0
  fi
  echo; echo "================ $name ================"
  local t=$(date +%s)
  if "$@" 2>&1 | tee "$LOGD/$name.log"; then
    touch "$STAMP/$name"
    echo "[done] $name in $(( $(date +%s) - t ))s"
  else
    echo "[ERROR] $name failed - see $LOGD/$name.log (re-run this command after fixing)"
    return 1
  fi
}
always() { # always re-run (cheap steps: aggregation, analysis, report)
  local name=$1; shift
  echo; echo "================ $name ================"
  if ! "$@" 2>&1 | tee "$LOGD/$name.log"; then
    echo "[ERROR] $name failed - see $LOGD/$name.log"; return 1
  fi
}

# ---- A. Environment checks and pipeline fidelity ----------------------------------------------
step A1_preflight   python -m retrace.preflight
step A2_unittest    env THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_retrace.py
step A3_parity      python -m retrace.pipeline parity --n 20 --models "$MODELS"

# ---- B. Data for the evaluation ---------------------------------------------------------------
step B1_lexicons    python -m retrace.lexicons export-review
step B2_eligible    python -m retrace.eligible build --models "$MODELS" --tier lenient --cap 25 --pilot 50
step B3_pools       python -m retrace.pools build --what carriers,distractors,false-flag

# ---- C. Generate and score variants -----------------------------------------------------------
step "C1_$R_RAND"   python -m retrace.runner --run "$R_RAND" --docs "$DOCS" --ops "$OPS_RAND" \
                      --strategy random --models "$MODELS"
step "C2_${R_WORST}_greedy" python -m retrace.runner --run "$R_WORST" --docs "$DOCS" --ops "$OPS_GREEDY" \
                      --strategy greedy --models "$MODELS"
step "C3_${R_WORST}_bon"    python -m retrace.runner --run "$R_WORST" --docs "$DOCS" --ops "$OPS_BON" \
                      --strategy best_of_n --n 5 --models "$MODELS"

RUNS="$R_RAND,$R_WORST"
if [ "$LLM" != "0" ]; then
  echo; echo "================ LLM check (Ollama) ================"
  if python -m retrace.llm 2>&1 | tee "$LOGD/llm_check.log"; then
    LLM=1
  elif [ "$LLM" = "1" ]; then
    echo "[LLM] LLM=1 but Ollama/the model is not ready (see the error above and docs/PIPELINE.md) -> STOP."
    exit 1
  else
    LLM=0; echo "[LLM] Skipping T1,T2,E2 neutral,E6 (Ollama/the model is not ready - see the error above)"
  fi
fi
if [ "$LLM" = "1" ]; then
  # the step name includes LLM_DOCS: switching pilot -> eligible resumes (existing variants are skipped, LLM output comes from the cache)
  step "C4_${R_LLM}__docs-$LLM_DOCS"  python -m retrace.runner --run "$R_LLM" --docs "$LLM_DOCS" --ops "$OPS_LLM" \
                      --ops-section ops_llm --strategy best_of_n --n 3 --models "$MODELS"
fi
if [ -f "$ADV/runs/$R_LLM/scores.jsonl.gz" ]; then
  RUNS="$RUNS,$R_LLM"
  # bidirectional NLI for the T1/T2 variants (downloads a DeBERTa NLI model on first use, ~0.7 GB); a failure only warns
  always "C5_nli_$R_LLM" python -m retrace.validate nli --run "$R_LLM" \
    || echo "[warning] NLI could not run - the report will lack the NLI table (other numbers are unaffected)"
fi

# ---- D. Aggregation, analysis, report (always re-run) -----------------------------------------
IFS=',' read -ra RLIST <<< "$RUNS"
for r in "${RLIST[@]}"; do
  always "D1_metrics_$r"  python -m retrace.metrics summarize --run "$r"
done
[ -f "$STAMP/D2_size_probe" ] || step D2_size_probe python -m retrace.analysis size-probe --n 2000
always D3_ttp_fp python -m retrace.analysis ttp-fp
for r in "${RLIST[@]}"; do
  always "D4_analysis_$r" python -m retrace.analysis all --run "$r"
done
if [ -n "$SUF" ]; then
  always D5_report python -m retrace.report --runs "$RUNS"
  # the report reads the standard run names (rand/worst/llm); for other DOCS, write to a separate name
  mv -f $ADV/REPORT.md "$ADV/REPORT$SUF.md"; mv -f $ADV/report_numbers.json "$ADV/report_numbers$SUF.json"
  REP="$ADV/REPORT$SUF.md"
else
  always D5_report python -m retrace.report --runs "$RUNS"
  REP="$ADV/REPORT.md"
fi

echo
echo "=============================================================================="
echo " DONE in $(( ($(date +%s) - T0) / 60 )) min. Report: $REP"
echo " Tables: $ADV/runs/<run>/tables/   Figures: $ADV/runs/<run>/figures/   Logs: $LOGD/"
echo "=============================================================================="
