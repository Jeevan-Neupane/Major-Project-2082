#!/usr/bin/env bash
# Train fair_<model>_<dataset> configs on the local GPU, N at a time, longest jobs first.
# Waits until the GPU is free (medllm-queue stopped and < 4 GB GPU memory in use) before starting.
# Start it in tmux so it survives SSH disconnects, e.g.:
#   tmux new -d -s dvc-train "bash scripts/run_fair_local.sh dualviewcnn 3 ctpelvic rib verse19 hip femur verse20"
set -u
MODEL=$1; SLOTS=$2; shift 2; DATASETS="$*"
ROOT=/home/ubuntu/Major-Project-2082
PY=/opt/Major-Project-2082/rootvenv/bin/python
RUN_DIR=$ROOT/campaigns/fair_${MODEL}_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/queue.log
cd "$ROOT"

echo "$(date -u '+%F %T') waiting for a free GPU (datasets: $DATASETS, $SLOTS at a time)" | tee -a "$LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! systemctl --user is-active -q medllm-queue && [ "$used" -lt 4000 ]; then break; fi
  sleep 30
done
echo "$(date -u '+%F %T') GPU free (${used} MiB used); starting" | tee -a "$LOG"

run_one() {
  ds=$1
  echo "$(date -u '+%F %T') start fair_${MODEL}_${ds}" >> "$LOG"
  $PY train.py --config "fair_${MODEL}_${ds}" --no_wandb --num_workers 2 > "$RUN_DIR/${ds}.out" 2>&1
  echo "$(date -u '+%F %T') end   fair_${MODEL}_${ds} exit=$?" >> "$LOG"
}
export -f run_one
export MODEL PY RUN_DIR LOG
printf '%s\n' $DATASETS | xargs -P "$SLOTS" -I{} bash -c 'run_one {}'
echo "$(date -u '+%F %T') all done" | tee -a "$LOG"
