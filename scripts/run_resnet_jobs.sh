#!/usr/bin/env bash
# Sequential driver for the ResNet-50 learning-rate band and budget curve.
# Run detached (setsid) -- these take hours and must survive the parent shell exiting.
#
# Precision differs by job on purpose: the lr band keeps bf16 so it stays comparable with
# the early-stopped endpoint run it extends, while the budget curve uses fp32, measured
# 1.53x faster on this Turing card (13.4s vs 20.5s per epoch) at indistinguishable loss.
# Fixed epochs = 6 is the median of the endpoint run's per-fold best epochs (3,5,7,8),
# chosen before any budget-curve result was inspected.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-gpu/bin/python
LOG=experiments/logs/resnet_budget.log
export PYTHONUNBUFFERED=1
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "### JOB 1: lr band (bf16, early stopping)"
$PY -m baselines.resnet50_budget --job lr --lrs 3e-5 1e-4 3e-4 >> "$LOG" 2>&1
log "### JOB 2a: kshot (fp32, 6 fixed epochs)"
$PY -m baselines.resnet50_budget --job kshot --k 1 2 4 6 --episodes 0 1 2 --epochs 6 --precision fp32 >> "$LOG" 2>&1
log "### JOB 2b: balanced"
$PY -m baselines.resnet50_budget --job balanced --budgets 12 40 --draws 0 1 2 --epochs 6 --precision fp32 >> "$LOG" 2>&1
log "### JOB 2c: natural"
$PY -m baselines.resnet50_budget --job natural --budgets 100 400 --draws 0 1 2 --epochs 6 --precision fp32 >> "$LOG" 2>&1
log "### ALL DONE"
