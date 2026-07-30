#!/usr/bin/env bash
# Repair every cell on the NVIDIA lane that still holds placeholder rows.
#
# Supersedes repair_minimax.sh, which had two defects worth recording. It gated only on the
# first cell, and it read each cell's outcome by grepping the shared log for the last
# "repaired N/M" line -- so when a cell died without printing one, the gate silently re-read
# the PREVIOUS cell's success and the queue carried on. Three cells crashed at 30-50% and
# saved nothing while the log looked healthy. This version checks the exit status of each
# run and counts the remaining placeholders on disk, which is the only fact that matters.
#
# One lane, strictly sequential: both models are served by NVIDIA NIM, and two concurrent
# processes against one provider is what produced these placeholders in the first place.
#
# Two failure modes, two treatments:
#   minimax-m3  -- 429s under concurrency. rpm 3 / 2 workers measured at 0% residual.
#   nemotron    -- genuine unparseable completions, not throttling, so the rate is not the
#                  problem; --only parse_failed uses the retry ladder that shows the model
#                  its own malformed output. Runs first: 53 rows against minimax's ~1,900.
#
# Cells are retried up to MAX_PASSES times, because a row that raises a provider error keeps
# its placeholder by design and is only picked up by a later invocation.
set -uo pipefail
cd "$(dirname "$0")/.."

MAX_PASSES=3
LOG=experiments/logs/repair_queue.log
export PYTHONUNBUFFERED=1

# stem|rpm|workers|only -- overridable by passing cell specs as arguments, so a caller that
# has just generated new cells can reuse this retry loop instead of reimplementing it.
CELLS=(
  "sens_v2_3class_nemotron_nano_vl_8b|10|2|parse_failed"
  "sens_v3_3class_nemotron_nano_vl_8b|10|2|parse_failed"
  "sens_v4_sub_nemotron_nano_vl_8b|10|2|parse_failed"
  "sens_v5_sub_nemotron_nano_vl_8b|10|2|parse_failed"
  "zsnaive_3class_nemotron_nano_vl_8b|10|2|parse_failed"
  "zs_3class_nemotron_nano_vl_8b|10|2|parse_failed"
  "zs_3class_minimax_m3|3|2|all"
  "zsnaive_3class_minimax_m3|3|2|all"
  "sens_v2_sub_minimax_m3|3|2|all"
  "sens_v3_sub_minimax_m3|3|2|all"
  "sens_v4_sub_minimax_m3|3|2|all"
  "sens_v5_sub_minimax_m3|3|2|all"
  "k1_e1_minimax_m3|3|2|all"
)

if [ "$#" -gt 0 ]; then
  CELLS=("$@")
  LOG=experiments/logs/repair_queue_$(date +%Y%m%d_%H%M%S).log
fi
: > "$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Placeholders left in a cell, counted from the file rather than from the log.
remaining() {
  .venv/bin/python - "$1" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("api_error") or r.get("parse_failed"):
        n += 1
print(n)
PY
}

log "queue starting: ${#CELLS[@]} cells"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r stem rpm workers only <<<"$spec"
  out="experiments/results/${stem}.jsonl"
  if [ ! -f "$out" ]; then
    log "$stem: MISSING, skipping"
    continue
  fi

  before=$(remaining "$out")
  if [ "$before" -eq 0 ]; then
    log "$stem: already clean, skipping"
    continue
  fi

  for pass in $(seq 1 "$MAX_PASSES"); do
    log "$stem: pass $pass/$MAX_PASSES, $before placeholders (rpm $rpm, workers $workers, only=$only)"
    .venv/bin/python -m scripts.rerun_failed \
        --results "$out" --rpm "$rpm" --workers "$workers" --only "$only" >> "$LOG" 2>&1
    rc=$?
    after=$(remaining "$out")
    log "$stem: pass $pass exit=$rc, $before -> $after placeholders"

    if [ "$after" -eq 0 ]; then
      log "$stem: CLEAN"
      break
    fi
    if [ "$after" -eq "$before" ] && [ "$rc" -eq 0 ]; then
      log "$stem: no progress on a clean exit; the remainder is not repairable by retrying"
      break
    fi
    before=$after
  done
done

log "QUEUE DONE"
.venv/bin/python - <<'PY' | tee -a "$LOG"
import glob, json
print("final placeholder inventory:")
total = 0
for f in sorted(glob.glob("experiments/results/*.jsonl")):
    if f.split("/")[-1].startswith(("probe_", "resnet")):
        continue
    n = sum(1 for line in open(f)
            if (lambda r: r.get("api_error") or r.get("parse_failed"))(json.loads(line)))
    if n:
        print(f"  {f.split('/')[-1]:44s} {n}")
        total += n
print("total:", total)
PY
