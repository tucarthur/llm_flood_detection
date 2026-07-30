#!/usr/bin/env bash
# Generate minimax's four prompt-sensitivity variants at full n, then repair the placeholders
# that generation leaves behind.
#
# Stage 1 uses run_queue.sh, which is idempotent and resumable: a cell already at 1,592 rows is
# skipped and a partial cell is resumed. That matters here because the whole run is ~42 hours,
# so it must survive an interruption without losing work.
#
# Stage 2 hands the four cells to repair_queue.sh. Generation at any rate leaves some 429
# placeholders, and those deflate flood recall until re-issued.
#
# One lane, one process: nothing else may touch the NVIDIA endpoint while this runs.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=experiments/logs/minimax_sens_full.log
export PYTHONUNBUFFERED=1
: > "$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "STAGE 1: generating v2-v5 at full n via run_queue.sh"
bash scripts/run_queue.sh minimaxSensFull experiments/queue_minimax_sens_full.spec >> "$LOG" 2>&1
log "STAGE 1 done (exit=$?)"

log "STAGE 2: repairing placeholders in the four cells"
bash scripts/repair_queue.sh \
    "sens_v2_3class_minimax_m3|3|2|all" \
    "sens_v3_3class_minimax_m3|3|2|all" \
    "sens_v4_3class_minimax_m3|3|2|all" \
    "sens_v5_3class_minimax_m3|3|2|all" >> "$LOG" 2>&1
log "STAGE 2 done (exit=$?)"

log "MINIMAX SENS FULL DONE"
