#!/usr/bin/env bash
# (Optional) Train extra seeds of main/no_cl/no_mamba (seed variance) and a model with same-size
# contrastive negatives (Eq. 8, --cl-neg matched), then evaluate their robustness. Needs a GPU.
# Requires the core patch in patches/ (adds --cl-neg to src.train).
#   bash scripts/train_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate

for s in 7 13; do
  for flag in "" "--no-cl" "--no-mamba"; do
    name=$([ -z "$flag" ] && echo main || echo "${flag#--no-}"); name="${name/-/_}"
    echo "== seed $s | ${name} =="
    python -m src.train $flag --seed $s --suffix "s$s"
  done
done

echo "== Same-size contrastive negatives (Eq. 8), default seed =="
python -m src.train --cl-neg matched --suffix negm

echo "== Robustness evaluation of the new tags =="
for t in main_s7 main_s13 no_cl_s7 no_cl_s13 no_mamba_s7 no_mamba_s13 main_negm; do
  python -m src.evaluate "$t" --robustness || echo "  (skipping $t)"
done
echo "Done. Compare macro-F1 across seeds in outputs/metrics_*.csv and robustness_*.csv"
