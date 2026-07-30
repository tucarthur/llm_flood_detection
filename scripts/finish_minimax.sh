#!/usr/bin/env bash
# Finish the minimax lane: generate the cells that were never completed, then repair whatever
# placeholders that generation leaves behind.
#
# Stage 1 is generation, not repair. k1_e2 is stranded at 164/375 because the lane was stopped
# mid-run, and k2_e0/e1/e2 were never started; run_queue.sh resumes the first and runs the
# other three, skipping every cell that already has its full row count. Stage 2 then hands the
# four cells to repair_queue.sh, because generation at any rate leaves some 429 placeholders
# and those are only fixable by re-issuing the affected frames.
#
# The spec now carries rpm 3 / 2 workers rather than the rpm 10 / 4 workers that produced
# 22-40% placeholder rates on the earlier cells. That is slower per row (~2.5/min) but the
# first-pass residual measured over ~1,900 repaired rows was 1-8%, so it is far less rework.
#
# One lane, one process: nothing else may touch the NVIDIA endpoint while this runs.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=experiments/logs/finish_minimax.log
export PYTHONUNBUFFERED=1
: > "$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "STAGE 1: generating missing cells via run_queue.sh"
bash scripts/run_queue.sh nvidiaMinimax experiments/queue_nvidia_minimax.spec >> "$LOG" 2>&1
log "STAGE 1 done (exit=$?)"

log "STAGE 2: repairing placeholders in the four generated cells"
bash scripts/repair_queue.sh \
    "k1_e2_minimax_m3|3|2|all" \
    "k2_e0_minimax_m3|3|2|all" \
    "k2_e1_minimax_m3|3|2|all" \
    "k2_e2_minimax_m3|3|2|all" >> "$LOG" 2>&1
log "STAGE 2 done (exit=$?)"

log "FINISH MINIMAX DONE"
