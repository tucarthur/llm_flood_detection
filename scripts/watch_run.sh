#!/usr/bin/env bash
# Generic unattended watchdog for a long detached run.
#
#   scripts/watch_run.sh <run-log> <done-marker> <driver-pattern> <results-glob>
#
# Supersedes watch_finish_minimax.sh, which measured progress by counting rows and therefore
# reported a false STALLED verdict the moment the run moved from generation into repair:
# repair rewrites rows in place and never appends, so the row count is flat by design. It
# exited 6h45m before the run actually completed, having watched the wrong quantity.
#
# Liveness here is the newest modification time across the results glob, which advances under
# both phases -- generation appends, and repair checkpoints every 25 rows. Row counts are still
# logged, because they are the informative number for a human reading the log; they are just no
# longer the stall signal.
#
# Observes only. Never restarts, kills or edits anything: a second process against the same
# endpoint is what caused the placeholder rates this pipeline spent two days recovering from.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_LOG="${1:?usage: watch_run.sh <run-log> <done-marker> <driver-pattern> <results-glob>}"
MARKER="${2:?}"
DRIVER="${3:?}"
GLOB="${4:?}"

INTERVAL=300
STALL_CHECKS=12          # ~1 hour of no file activity
STATUS="${RUN_LOG%.log}.watch.log"

say() { echo "[$(date -Is)] $*" >> "$STATUS"; }

newest_mtime() { find $GLOB -maxdepth 0 -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1; }
detail() { for f in $GLOB; do [ -f "$f" ] && printf ' %s=%s' "$(basename "$f" .jsonl)" "$(wc -l < "$f")"; done; }

: > "$STATUS"
say "watchdog started: marker='$MARKER' driver='$DRIVER' interval=${INTERVAL}s"
say "baseline:$(detail)"

last=$(newest_mtime); flat=0

while true; do
  sleep "$INTERVAL"

  # The pattern is passed in, so it cannot match this script's own command line.
  alive=$(ps -ef | grep "$DRIVER" | grep -v grep | grep -vc "watch_run")
  done_marker=$(grep -ac "$MARKER" "$RUN_LOG" 2>/dev/null || echo 0)
  now=$(newest_mtime)

  if [ "$done_marker" -gt 0 ]; then
    say "progress:$(detail)"
    say "VERDICT: COMPLETED -- marker '$MARKER' present"
    exit 0
  fi

  if [ "$alive" -eq 0 ]; then
    say "progress:$(detail)"
    say "VERDICT: DIED -- no '$DRIVER' process and no completion marker"
    tail -20 "$RUN_LOG" >> "$STATUS" 2>&1
    exit 1
  fi

  if [ -n "$now" ] && [ -n "$last" ] && [ "$now" -le "$last" ]; then
    flat=$((flat + 1))
    if [ "$flat" -ge "$STALL_CHECKS" ]; then
      say "progress:$(detail)"
      say "VERDICT: STALLED -- no file activity in $((STALL_CHECKS * INTERVAL / 60)) min, process alive"
      exit 2
    fi
  else
    flat=0
    say "progress:$(detail)"
  fi
  last=$now
done
