#!/usr/bin/env bash
# Run one lane of the experiment matrix sequentially.
#
# A "lane" is one provider. Cells within a lane MUST run one at a time: agent.run
# enforces --rpm per process, so two concurrent runs against the same provider would
# together exceed the rate limit and draw sustained 429s. Different providers are
# independent, so lanes run in parallel with each other.
#
# Usage: scripts/run_queue.sh <lane-name> <spec-file>
#
# Spec file: one cell per line, '#' comments and blank lines ignored, fields pipe-separated
#   provider|model|rpm|out_stem|input_jsonl|extra agent.run flags
#
# input_jsonl may be blank to mean the full evaluation set. It is a per-cell field because
# the K-shot sweep runs on a 375-row subsample while K=0 cells ran at full 1592, and the
# completion check needs the right row count for the cell it is checking.
#
# Idempotent and restartable: a cell whose output already has $EXPECTED lines is skipped,
# and an incomplete one is resumed rather than restarted, so relaunching the lane after a
# crash or a rate-limit stall costs nothing.
set -uo pipefail

LANE="${1:?usage: run_queue.sh <lane-name> <spec-file>}"
SPEC="${2:?usage: run_queue.sh <lane-name> <spec-file>}"

DEFAULT_INPUT="data/processed/test_examples.jsonl"
RESULTS_DIR="experiments/results"
LOG_DIR="experiments/logs"
MAX_ATTEMPTS=4

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LANE_LOG="$LOG_DIR/lane_${LANE}.log"

log() { echo "[$(date -Is)] [$LANE] $*" | tee -a "$LANE_LOG"; }

log "lane starting; spec=$SPEC"

while IFS='|' read -r provider model rpm stem input extra; do
    case "${provider// /}" in ''|'#'*) continue ;; esac
    INPUT="${input:-$DEFAULT_INPUT}"
    [ -z "${INPUT// /}" ] && INPUT="$DEFAULT_INPUT"
    EXPECTED=$(wc -l < "$INPUT")
    out="$RESULTS_DIR/${stem}.jsonl"
    cell_log="$LOG_DIR/${stem}.log"

    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        have=0
        [ -f "$out" ] && have=$(wc -l < "$out")
        if [ "$have" -ge "$EXPECTED" ]; then
            log "$stem: complete ($have/$EXPECTED), skipping"
            break
        fi

        resume=""
        [ "$have" -gt 0 ] && resume="--resume"
        log "$stem: attempt $attempt/$MAX_ATTEMPTS at $have/$EXPECTED rows $resume"

        # shellcheck disable=SC2086 -- $extra and $resume are intentionally word-split
        # --taxonomy is NOT hardcoded here: it is a per-cell factor and must come from the
        # spec's flags column, so a spec file fully determines what it reproduces.
        .venv/bin/python -m agent.run \
            --input "$INPUT" --out "$out" \
            --json-mode schema \
            --provider "$provider" --model "$model" --rpm "$rpm" \
            $resume $extra >> "$cell_log" 2>&1
        rc=$?

        have=0
        [ -f "$out" ] && have=$(wc -l < "$out")
        if [ "$have" -ge "$EXPECTED" ]; then
            log "$stem: DONE ($have/$EXPECTED, exit=$rc)"
            break
        fi
        # Backoff before resuming: a stall is usually a rate-limit window that needs time
        # to clear, and hammering it immediately just burns the next window too.
        log "$stem: incomplete ($have/$EXPECTED, exit=$rc); backing off $((attempt * 120))s"
        sleep $((attempt * 120))
    done
done < "$SPEC"

log "lane finished"
