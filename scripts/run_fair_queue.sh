#!/usr/bin/env bash
# Run a list of fair configs on the local GPU, SLOTS at a time, in the order given (put long jobs first).
# Optionally waits for a tmux session to finish first, then for a free GPU (medllm-queue stopped and
# < 4 GB GPU memory in use). Start it in tmux so it survives SSH disconnects, e.g.:
#   tmux new -d -s fair-queue "bash scripts/run_fair_queue.sh dvc-train 2 fair_swinx2s_verse19 fair_multiscale_ctpelvic ..."
# Memory per job on a 32 GB GPU: SwinX2S Tiny ~14 GB, MultiScale ~10 GB, DualViewCNN 3-9 GB -> 2 slots.
set -u
WAIT_SESSION=$1; SLOTS=$2; shift 2; CONFIGS="$*"
ROOT=/home/ubuntu/Major-Project-2082
PY=/opt/Major-Project-2082/rootvenv/bin/python
RUN_DIR=$ROOT/campaigns/fair_queue_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/queue.log
cd "$ROOT"

if [ "$WAIT_SESSION" != "none" ]; then
  echo "$(date -u '+%F %T') waiting for tmux session '$WAIT_SESSION' to finish" | tee -a "$LOG"
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do sleep 60; done
fi
echo "$(date -u '+%F %T') waiting for a free GPU" | tee -a "$LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! systemctl --user is-active -q medllm-queue && [ "$used" -lt 4000 ]; then break; fi
  sleep 30
done
echo "$(date -u '+%F %T') GPU free (${used} MiB used); starting $SLOTS at a time: $CONFIGS" | tee -a "$LOG"

run_one() {
  cfg=$1
  echo "$(date -u '+%F %T') start $cfg" >> "$LOG"
  $PY train.py --config "$cfg" --no_wandb --num_workers 2 > "$RUN_DIR/${cfg}.out" 2>&1
  echo "$(date -u '+%F %T') end   $cfg exit=$?" >> "$LOG"
}
export -f run_one
export PY RUN_DIR LOG
printf '%s\n' $CONFIGS | xargs -P "$SLOTS" -I{} bash -c 'run_one {}'
echo "$(date -u '+%F %T') all done" | tee -a "$LOG"
