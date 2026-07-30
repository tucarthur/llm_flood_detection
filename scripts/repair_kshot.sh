#!/usr/bin/env bash
# Repair placeholder rows in the Gemma K-shot cells.
#
# All K-shot failures were 429s on the input-TOKEN ceiling (16k tokens/min free tier), not
# the request rate -- so the repair rate is set from tokens: ~4.2k prompt tokens at K=4
# allows ~3/min, ~2.6k at K=2 allows ~6/min. Single worker, since concurrency is what
# saturated the ceiling in the first place.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/logs/repair_kshot.log
export PYTHONUNBUFFERED=1
: > "$LOG"
for c in k2_e0_gemma4_31b k2_e0_gemma4_26b_a4b k2_e1_gemma4_31b k2_e1_gemma4_26b_a4b \
         k2_e2_gemma4_31b k2_e2_gemma4_26b_a4b; do
  echo "[$(date -Is)] repairing $c (rpm 6)" | tee -a "$LOG"
  .venv/bin/python -m scripts.rerun_failed --results experiments/results/$c.jsonl --rpm 6 --workers 1 >> "$LOG" 2>&1
done
for c in k4_e0_gemma4_31b k4_e0_gemma4_26b_a4b k4_e1_gemma4_31b k4_e1_gemma4_26b_a4b \
         k4_e2_gemma4_31b k4_e2_gemma4_26b_a4b; do
  echo "[$(date -Is)] repairing $c (rpm 3)" | tee -a "$LOG"
  .venv/bin/python -m scripts.rerun_failed --results experiments/results/$c.jsonl --rpm 3 --workers 1 >> "$LOG" 2>&1
done
echo "[$(date -Is)] REPAIR DONE" | tee -a "$LOG"
