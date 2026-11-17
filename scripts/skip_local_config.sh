#!/usr/bin/env bash
# Skip one config in the running local queue because it is trained elsewhere (Modal).
# Waits until the queue starts that config, kills only its train.py process, and appends a marker to its
# .out file so scripts/retry_failed_runs.sh does not rerun it. The queue continues with the next config.
#   tmux new -d -s skip-rib "bash scripts/skip_local_config.sh fair_multiscale_rib"
set -u
CFG=$1
ROOT=/home/ubuntu/Major-Project-2082
Q=$(ls -td $ROOT/campaigns/fair_queue_* | head -1)
echo "$(date -u '+%F %T') waiting for '$CFG' to start in $Q"
while ! grep -q " start $CFG\$" "$Q/queue.log"; do sleep 20; done
sleep 5
pkill -9 -f "train.py --config $CFG --no_wandb" && echo "$(date -u '+%F %T') killed local $CFG"
echo "Best Val Dice : n/a (skipped locally on $(date -u '+%F %T'); trained on Modal instead)" >> "$Q/$CFG.out"
echo "$(date -u '+%F %T') marker written to $Q/$CFG.out"
