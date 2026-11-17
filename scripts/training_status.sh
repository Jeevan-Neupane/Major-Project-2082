#!/usr/bin/env bash
# One-line-per-run status of the fair training runs (16 Nov 2026): Modal (both accounts) and local.
T=$(mktemp -d)
date -u '+now: %H:%M UTC'
# first account (jeevan-neupane) stopped at its budget on 16 Nov; its partial runs are not used
for prof in second-account; do
  export MODAL_PROFILE=$prof
  for f in $(timeout 120 ~/modal-env/bin/modal volume ls biplanar-data results/logs 2>/dev/null \
             | grep -oE "fair_swinx2s_[a-z0-9]+_20261116_1[1-9][0-9]{4}\.log" | sort -u); do
    timeout 60 ~/modal-env/bin/modal volume get biplanar-data "results/logs/$f" "$T/$f" >/dev/null 2>&1
    n=${f%_2026*}
    ep=$(grep -a "\] Train Loss" "$T/$f" | tail -1 | grep -oE "\[Epoch [0-9]+/200\]")
    best=$(grep -a "\*BEST\*" "$T/$f" | tail -1 | grep -oE "Val Dice: [0-9.]+" | sed 's/Val Dice: //')
    fin=$(grep -a -cE "Best Val Dice" "$T/$f")
    started=$(echo "$f" | grep -oE "_[0-9]{6}\.log" | cut -c2-5)
    note=""; [ "$fin" = "0" ] && grep -aq "preemption" "$T/$f" && note=" (preempted)"
    echo "[modal $prof] ${n#fair_} (started ${started:0:2}:${started:2:2}) | $ep | best val Dice $best | finished: $fin$note"
  done
  n=$(timeout 60 ~/modal-env/bin/modal container list --json 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)))")
  echo "[modal $prof] running containers: $n"
done
unset MODAL_PROFILE
for D in $(ls -td /home/ubuntu/Major-Project-2082/campaigns/fair_dualviewcnn_* 2>/dev/null | head -1) \
         $(ls -td /home/ubuntu/Major-Project-2082/campaigns/fair_queue_* 2>/dev/null | head -1); do
  for f in "$D"/*.out; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .out)
    ep=$(grep -a "\] Train Loss" "$f" | tail -1 | grep -oE "\[Epoch [0-9]+/200\]")
    best=$(grep -a "\*BEST\*" "$f" | tail -1 | grep -oE "Val Dice: [0-9.]+" | sed 's/Val Dice: //')
    fin=$(grep -a -cE "Best Val Dice" "$f")
    err=$(grep -a -cE "Traceback|CUDA out of memory" "$f")
    echo "[local $(basename "$D" | cut -d_ -f2)] $n | $ep | best val Dice $best | finished: $fin | errors: $err"
  done
  tail -1 "$D/queue.log"
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/local GPU: /'
rm -rf "$T"
