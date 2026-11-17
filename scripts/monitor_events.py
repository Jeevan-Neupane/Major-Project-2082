#!/usr/bin/env python3
"""Print one line per important training event (start, early stop, finish, crash, Modal job end,
download), plus a short status of running local jobs every 30 minutes. Meant for a live monitor.

State (bytes already read per file) is kept in a JSON file, so restarting the monitor does not
repeat old events. Usage:  python3 scripts/monitor_events.py [--state FILE] [--interval 60]
"""
import argparse
import glob
import json
import os
import re
import time

ROOT = "/home/ubuntu/Major-Project-2082"
MODAL_LOGS = {  # local copies of the Modal launch output (container logs stream into these)
    "modal-second": "/tmp/modal_data/launch_second_swin.log",
}
DOWNLOAD_LOGS = [f"{ROOT}/modal_results_second_download.log"]

EARLY = re.compile(r"Early stopping at epoch (\d+).*Best Dice: ([0-9.]+) at epoch (\d+)")
BEST = re.compile(r"Best Val Dice\s*:\s*([0-9.]+)")
EPOCH = re.compile(r"\[Epoch (\d+)/(\d+)\].*Val Dice: ([0-9.]+)")
ERROR = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|Killed|RuntimeError|Error:")
SPLIT = re.compile(r"Split manifest: splits/(\S+?)_seed")


def watched_files():
    files = {}
    for d in sorted(glob.glob(f"{ROOT}/campaigns/fair_dualviewcnn_*") + glob.glob(f"{ROOT}/campaigns/fair_queue_*")):
        files[f"{d}/queue.log"] = ("queue", os.path.basename(d))
        for f in glob.glob(f"{d}/*.out"):
            name = os.path.basename(f)[:-4]
            files[f] = ("run", name if name.startswith("fair_") else f"fair_dualviewcnn_{name}")
    for f in glob.glob(f"{ROOT}/campaigns/retry_failed_*.log"):
        files[f] = ("retry", "retry")
    for tag, f in MODAL_LOGS.items():
        files[f] = ("modal", tag)
    for f in DOWNLOAD_LOGS:
        files[f] = ("download", "download")
    return {f: v for f, v in files.items() if os.path.exists(f)}


def new_lines(path, state):
    size = os.path.getsize(path)
    start = state.get(path, None)
    if start is None or start > size:   # first time seen: start at the beginning for new run files only
        start = 0
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read()
    state[path] = start + len(data)
    # keep only complete lines; tqdm progress bars use \r, so split on both
    return [l for l in re.split(r"[\r\n]+", data.decode("utf-8", "ignore")) if l.strip()]


def events_for(kind, name, lines):
    out = []
    for l in lines:
        if kind == "queue":
            if " start " in l or " end " in l or "GPU free" in l or "all done" in l:
                out.append(f"[local queue] {l.strip()}")
        elif kind == "retry":
            out.append(f"[retry] {l.strip()}")
        elif kind == "download":
            if "download complete" in l or "failed" in l or "all jobs finished" in l:
                out.append(f"[download] {l.strip()}")
        elif kind == "run":
            m = EARLY.search(l)
            if m:
                out.append(f"[local] {name}: EARLY STOP at epoch {m[1]} - best val Dice {float(m[2]):.4f} (epoch {m[3]})")
            elif BEST.search(l):
                out.append(f"[local] {name}: FINISHED - best val Dice {float(BEST.search(l)[1]):.4f}")
            elif ERROR.search(l) and "it/s" not in l:
                out.append(f"[local] {name}: ERROR - {l.strip()[:160]}")
        elif kind == "modal":
            m = EARLY.search(l)
            if m:
                out.append(f"[{name}] EARLY STOP at epoch {m[1]} - best val Dice {float(m[2]):.4f} (epoch {m[3]})")
            elif l.startswith("{'config'"):
                out.append(f"[{name}] JOB ENDED {l.strip()[:200]}")
            elif ERROR.search(l) and "it/s" not in l and "KeyboardInterrupt" not in l:
                out.append(f"[{name}] ERROR - {l.strip()[:160]}")
    return out


def status_line():
    parts = []
    for d in glob.glob(f"{ROOT}/campaigns/fair_queue_*") + glob.glob(f"{ROOT}/campaigns/fair_dualviewcnn_*"):
        for f in glob.glob(f"{d}/*.out"):
            txt = open(f, "rb").read().decode("utf-8", "ignore")
            if BEST.search(txt):
                continue   # finished
            eps = EPOCH.findall(txt)
            if eps:
                name = os.path.basename(f)[:-4].replace("fair_", "")
                best = max(float(e[2]) for e in eps)
                parts.append(f"{name} ep {eps[-1][0]}/{eps[-1][1]} best {best:.3f}")
    gpu = os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").read().strip()
    return f"[status {time.strftime('%H:%M', time.gmtime())} UTC] running locally: " + ("; ".join(parts) or "none") + f" | GPU {gpu}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="/tmp/monitor_state.json")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--status-every", type=int, default=1800)
    args = ap.parse_args()
    state = json.load(open(args.state)) if os.path.exists(args.state) else None
    first_run = state is None
    state = state or {}
    if first_run:   # do not replay history: start at the current end of files that already exist
        for f in watched_files():
            state[f] = os.path.getsize(f)
    last_status = 0.0
    while True:
        for f, (kind, name) in watched_files().items():
            for e in events_for(kind, name, new_lines(f, state)):
                print(e, flush=True)
        json.dump(state, open(args.state, "w"))
        if time.time() - last_status >= args.status_every:
            print(status_line(), flush=True)
            last_status = time.time()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
