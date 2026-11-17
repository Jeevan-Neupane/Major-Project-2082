#!/usr/bin/env bash
# Overnight dashboard: every 10 minutes print the status of all fair runs, and append it to campaigns/monitor.log.
#   tmux new -d -s train-monitor "bash scripts/monitor_dashboard.sh"      view with: tmux attach -t train-monitor
while true; do
  out=$(bash /home/ubuntu/Major-Project-2082/scripts/training_status.sh 2>&1)
  clear
  echo "$out"
  { echo "=================================================="; echo "$out"; } >> /home/ubuntu/Major-Project-2082/campaigns/monitor.log
  echo; echo "(refreshes every 10 min; detach with Ctrl+b then d)"
  sleep 600
done
