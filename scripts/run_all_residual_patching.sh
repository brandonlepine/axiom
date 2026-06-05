#!/usr/bin/env bash
# Run residual patching across datasets, smallest cohort first. No long command to paste:
#   nohup bash scripts/run_all_residual_patching.sh > resid_patching.log 2>&1 &
#   tail -f resid_patching.log
#
# Optional args:
#   $1 = config path   (default: configs/scoring/winoqueer_llama31_8b.yaml)
#   $2 = datasets       (default: "crows bbq combined_bbq_crows winoqueer")
# Each dataset auto-resolves its latest selected analysis cohort; resume is per-pair.
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${1:-configs/scoring/winoqueer_llama31_8b.yaml}"
DATASETS="${2:-crows bbq combined_bbq_crows winoqueer}"

cd "$(dirname "$0")/.." || exit 1
echo "config=$CONFIG"
echo "datasets=$DATASETS"

for ds in $DATASETS; do
  echo ""
  echo "===================== $ds ====================="
  if python scripts/run_residual_patching.py --config "$CONFIG" --dataset "$ds"; then
    echo "----- $ds: done -----"
  else
    echo "##### $ds: FAILED (continuing to next dataset) #####"
  fi
done

echo ""
echo "===================== ALL DATASETS COMPLETE ====================="
