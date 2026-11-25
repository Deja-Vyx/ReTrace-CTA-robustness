#!/usr/bin/env bash
# =============================================================================================
# Evaluate the case-study systems TTP-Match (Guru et al. 2025) and TTP-Vec (Hussain & Petrova 2026)
# on the ReTrace benchmark (levels 1 and 2) with ONE command. Run AFTER the ReTrace results exist
# (bash scripts/run_retrace.sh).
#
#   bash scripts/run_cta_systems.sh                # run everything; finished steps are skipped
#   ADAPT=0 bash scripts/run_cta_systems.sh        # skip the adaptive attacks (step C2)
#   LIMIT=200 bash scripts/run_cta_systems.sh      # dry run: 200 variants per run, 5 documents for adaptive
#   FORCE=1 bash scripts/run_cta_systems.sh        # re-run every step (existing results are RESUMED)
#
# Only READS outputs/retrace/ and data/; only WRITES outputs/cta_systems/. No ThreatMAMBA file is modified.
# Final result: outputs/cta_systems/REPORT_CTA_SYSTEMS.md (+ _pct.md, tables/, figures/)
# =============================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate

export PYTHONHASHSEED=0                         # as in ReTrace: stable IOC order -> rebuilt graphs match
export PYTHONUNBUFFERED=1
if [ -z "${CTA_SYSTEMS_TEST:-}" ]; then           # real run: no fake encoder/embeddings
  unset THREATMAMBA_FAKE_ENCODER THREATMAMBA_SSM RETRACE_FAKE_LLM CTA_SYSTEMS_FAKE_EMB \
        CTA_SYSTEMS_OUT CTA_SYSTEMS_RETRACE_DIR || true
fi

ADAPT="${ADAPT:-1}"
LIMIT="${LIMIT:-}"
OUT=outputs/cta_systems; LOGD=$OUT/logs; STAMP=$OUT/.done
[ -n "$LIMIT" ] && STAMP=$OUT/.done_limit$LIMIT
mkdir -p "$LOGD" "$STAMP"
T0=$(date +%s)
LIM_ARG=""; [ -n "$LIMIT" ] && LIM_ARG="--limit $LIMIT"
ADAPT_LIM=""; [ -n "$LIMIT" ] && ADAPT_LIM="--limit 5"

step() {   # step <name> <command...>
  local name=$1; shift
  if [ -f "$STAMP/$name" ] && [ -z "${FORCE:-}" ]; then
    echo "[skip] $name (already done - FORCE=1 to re-run)"; return 0
  fi
  echo; echo "================ $name ================"
  local t=$(date +%s)
  if "$@" 2>&1 | tee "$LOGD/$name.log"; then
    touch "$STAMP/$name"; echo "[done] $name in $(( $(date +%s) - t ))s"
  else
    echo "[ERROR] $name failed - see $LOGD/$name.log (re-run this command after fixing)"; return 1
  fi
}
always() {
  local name=$1; shift
  echo; echo "================ $name ================"
  "$@" 2>&1 | tee "$LOGD/$name.log"
}

# ---- A. Input checks --------------------------------------------------------------------------
echo "================ A0 input checks ================"
for f in outputs/ttp_extractor.pt data/processed/docs.jsonl data/attck/enterprise-attack-14.1.json \
         outputs/retrace/eligible_v0.jsonl; do
  [ -e "$f" ] || { echo "[ERROR] missing $f - run the reproduction and ReTrace first"; exit 1; }
done
for r in rand worst llm; do
  if [ -f "outputs/retrace/runs/$r/variants.jsonl.gz" ]; then echo "  run $r: present"; else echo "  run $r: MISSING (skipped)"; fi
done
ls data/cskg/*.pt >/dev/null 2>&1 && echo "  data/cskg: $(ls data/cskg/*.pt | wc -l) graphs" \
  || echo "  [warning] data/cskg is empty - TTP-Vec features will rebuild graphs (slower, needs a GPU)"
step A1_unittest env CTA_SYSTEMS_TEST=1 python tests/test_cta_systems.py

# ---- B. Training on the CTI2Attacker training set ---------------------------------------------
step B1_ttp_vec_train python -m cta_systems.ttp_vec.train
step B2_ttp_match_train    python -m cta_systems.ttp_match.train          # downloads BAAI/bge-large-en-v1.5 on first use (~1.3 GB)

# ---- C. Levels 1 + 2 --------------------------------------------------------------------------
step "C1_rescore${LIMIT:+_limit$LIMIT}" python -m cta_systems.evaluation.rescore $LIM_ARG
if [ "$ADAPT" = "1" ]; then
  # assign first (set -e stops immediately if the python command fails), then loop
  AMODELS=$(python -c "from cta_systems.common import load_cfg; print(' '.join(load_cfg()['adaptive']['models']))")
  for m in $AMODELS; do
    step "C2_adaptive_${m}${LIMIT:+_limit$LIMIT}" python -m cta_systems.evaluation.adaptive --model "$m" $ADAPT_LIM
  done
fi

# ---- D. Report (always re-run) ----------------------------------------------------------------
always D1_report     python -m cta_systems.evaluation.report --fmt dec
always D2_report_pct python -m cta_systems.evaluation.report --fmt pct

echo
echo "=============================================================================="
echo " DONE in $(( ($(date +%s) - T0) / 60 )) min."
echo " Report: $OUT/REPORT_CTA_SYSTEMS.md (decimals) and REPORT_CTA_SYSTEMS_pct.md (%)"
echo " Tables: $OUT/tables/   Figures: $OUT/figures/   Logs: $LOGD/"
echo "=============================================================================="
