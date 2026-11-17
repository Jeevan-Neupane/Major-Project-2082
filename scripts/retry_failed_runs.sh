#!/usr/bin/env bash
# Overnight safety net (16 Nov 2026). When the tmux session `fair-queue` has finished, rerun once every
# local fair run whose output has no final "Best Val Dice" summary or contains a Traceback (crash, OOM).
# Needed because run_fair_local.sh / run_fair_queue.sh log "exit=$?" after a command substitution, so the
# logged exit code is always 0 and cannot be trusted.
#   tmux new -d -s fair-retry "bash scripts/retry_failed_runs.sh"
set -u
ROOT=/home/ubuntu/Major-Project-2082
cd "$ROOT"
LOG=$ROOT/campaigns/retry_failed_$(date -u +%Y%m%d_%H%M%S).log
echo "$(date -u '+%F %T') waiting for tmux session 'fair-queue' to finish" | tee -a "$LOG"
while tmux has-session -t fair-queue 2>/dev/null; do sleep 120; done

failed=""
for d in $(ls -td campaigns/fair_dualviewcnn_* | head -1) $(ls -td campaigns/fair_queue_* | head -1); do
  for f in "$d"/*.out; do
    [ -f "$f" ] || continue
    if ! grep -aq "Best Val Dice" "$f" || grep -aq "Traceback" "$f"; then
      n=$(basename "$f" .out)
      case $n in fair_*) cfg=$n ;; *) cfg=fair_dualviewcnn_$n ;; esac
      failed="$failed $cfg"
      echo "$(date -u '+%F %T') not finished properly: $f -> rerun $cfg" | tee -a "$LOG"
    fi
  done
done
if [ -z "$failed" ]; then
  echo "$(date -u '+%F %T') all runs finished properly; nothing to rerun" | tee -a "$LOG"
  exit 0
fi
echo "$(date -u '+%F %T') rerunning:$failed" | tee -a "$LOG"
bash scripts/run_fair_queue.sh none 2 $failed
echo "$(date -u '+%F %T') retry queue done" | tee -a "$LOG"
