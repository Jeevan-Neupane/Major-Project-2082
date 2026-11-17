#!/usr/bin/env bash
# Wait until the given Modal apps have no running containers, then download all logs and best weights
# from the profile's private Volume "biplanar-data" (results/) to a local folder.
# Usage (runs on its own; start it in tmux so it survives SSH disconnects):
#   tmux new -d -s dl-second "bash scripts/modal_wait_and_download.sh second-account modal_results_second ap-XXXX"
# First use (16 Nov 2026, account jeevan-neupane) had the apps and destination hard-coded; same logic.
set -u
PROFILE=${1:?profile}; DEST_REL=${2:?destination folder}; shift 2; APPS="$*"
export MODAL_PROFILE=$PROFILE
MODAL=~/modal-env/bin/modal
ROOT=/home/ubuntu/Major-Project-2082
DEST=$ROOT/$DEST_REL
LOG=$ROOT/${DEST_REL}_download.log

echo "$(date -u '+%F %T') [$PROFILE] waiting for apps: $APPS" | tee -a "$LOG"
while true; do
  n=$($MODAL container list --json 2>/dev/null | APPS="$APPS" python3 -c '
import json, os, sys
try:
    apps = set(os.environ["APPS"].split())
    print(sum(1 for c in json.load(sys.stdin) if c.get("app_id") in apps))
except Exception:
    print("err")')
  if [ "$n" = "0" ]; then break; fi
  echo "$(date -u '+%F %T') running containers: $n" >> "$LOG"
  sleep 600
done

echo "$(date -u '+%F %T') all jobs finished; downloading results/ to $DEST" | tee -a "$LOG"
mkdir -p "$DEST"
for attempt in 1 2 3; do
  if $MODAL volume get --force biplanar-data results/ "$DEST" >> "$LOG" 2>&1; then
    echo "$(date -u '+%F %T') download complete:" | tee -a "$LOG"
    find "$DEST" -name "best_model_weights.pth" | sort | tee -a "$LOG"
    exit 0
  fi
  echo "$(date -u '+%F %T') download attempt $attempt failed; retrying in 60 s" | tee -a "$LOG"
  sleep 60
done
exit 1
