#!/usr/bin/env python3
"""Run grouped training experiments sequentially, with three models in parallel.

Each anatomy group launches DualViewCNN, MultiScaleDualViewCNN, and Swin-X2S
together, then waits for all three before the next anatomy begins.  A JSON
status file records process IDs, start/end times, and exit codes.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PREFIXES = ("dualviewcnn", "multiscale_dualviewcnn", "swinx2s")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def latest_run_id(config_name: str, started_at: float) -> str | None:
    candidates = [p for p in (ROOT / "logs").glob(f"{config_name}_*.csv") if p.stat().st_mtime >= started_at - 2]
    return max(candidates, key=lambda path: path.stat().st_mtime).stem if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue grouped training experiments")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--after-pids", nargs="*", type=int, default=[])
    parser.add_argument("--skip-configs", nargs="*", default=[])
    parser.add_argument("--campaign-id", default=f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    status_path = ROOT / "campaigns" / f"{args.campaign_id}.json"
    status = {"campaign_id": args.campaign_id, "created_at": now(), "state": "waiting", "after_pids": args.after_pids, "groups": []}
    write_status(status_path, status)

    while any(pid_running(pid) for pid in args.after_pids):
        time.sleep(30)

    for dataset in args.datasets:
        group = {"dataset": dataset, "started_at": now(), "state": "running", "jobs": []}
        status["state"] = "running"
        status["groups"].append(group)
        write_status(status_path, status)
        processes = []
        for prefix in MODEL_PREFIXES:
            config_name = f"{prefix}_{dataset}"
            if config_name in args.skip_configs:
                continue
            launcher_log = ROOT / "campaigns" / f"{args.campaign_id}_{config_name}.launcher.log"
            started_at = time.time()
            handle = launcher_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "train.py", "--config", config_name],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            job = {"config": config_name, "pid": process.pid, "started_at": now(), "launcher_log": str(launcher_log.relative_to(ROOT))}
            group["jobs"].append(job)
            processes.append((process, handle, job, started_at))
        write_status(status_path, status)

        while any(process.poll() is None for process, _, _, _ in processes):
            for process, _, job, _ in processes:
                job["state"] = "running" if process.poll() is None else "exited"
                job["exit_code"] = process.poll()
            status["last_checked_at"] = now()
            write_status(status_path, status)
            time.sleep(15)

        for process, handle, job, started_at in processes:
            job["exit_code"] = process.returncode
            handle.close()
            job["finished_at"] = now()
            run_id = latest_run_id(job["config"], started_at)
            if run_id:
                job["run_id"] = run_id
                subprocess.run([sys.executable, "scripts/generate_experiment_report.py", "--run-id", run_id], cwd=ROOT, check=False)
        group["state"] = "completed" if all(job["exit_code"] == 0 for job in group["jobs"]) else "completed_with_failures"
        group["finished_at"] = now()
        write_status(status_path, status)

    status["state"] = "completed"
    status["finished_at"] = now()
    write_status(status_path, status)
    print(status_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
