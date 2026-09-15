#!/bin/bash
# Sweep the context window length t (UA versus context length).
# Usage: bash scripts/sweep_context_duration.sh [IEMOCAP|MELD|SAFE]
set -e

DATASET="${1:-IEMOCAP}"
case "$DATASET" in
    IEMOCAP) SCRIPT="scripts/run_acert_iemocap.sh" ;;
    MELD)    SCRIPT="scripts/run_acert_meld.sh" ;;
    SAFE)    SCRIPT="scripts/run_acert_safe.sh" ;;
    *) echo "usage: $0 [IEMOCAP|MELD|SAFE]" >&2; exit 1 ;;
esac

for t in 5 10 15 20 25 30 35 40; do
    echo "=========== ${DATASET}: context duration = ${t}s ==========="
    CONTEXT_DURATION="$t" bash "$SCRIPT"
done

# Gather the whole sweep into one csv:
#   python utils/collect_results.py --results-dir "results/${DATASET}" \
#       --pattern "*_acert__*" --out "results/uar_${DATASET}.csv"
