"""Train any fair_<model>_<dataset> config on Modal GPUs (generalised from scripts/modal_swinx2s.py).

Remote: train.py, unchanged, with --no_wandb. Code comes from configs/, models/, utils/, splits/ and train.py
only (never .env). Data: the tar of the six datasets' train/ folders in the private Volume "biplanar-data"
(data/biplanar_train.tar) of the active Modal profile. Logs and best weights go to results/ in the same
Volume, committed every 5 minutes. Each job has a hard time limit (hours, given on the command line) that
caps its cost; at the limit training stops and the best weights so far are kept.

Usage (repository root; choose the account with MODAL_PROFILE and the GPU with FAIR_GPU, default L40S):
  MODAL_PROFILE=second-account ~/modal-env/bin/modal run scripts/modal_fair.py::timing --config fair_multiscale_hip
  MODAL_PROFILE=second-account ~/modal-env/bin/modal run --detach scripts/modal_fair.py::launch \
      --jobs fair_swinx2s_ctpelvic,fair_multiscale_femur --hours 5.3,3.0
  MODAL_PROFILE=second-account ~/modal-env/bin/modal volume get --force biplanar-data results/ modal_results_second/
"""
import os
import subprocess
import threading
import time

import modal

GPU = os.environ.get("FAIR_GPU", "L40S")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = "/root/project"
VOL = "/vol"

app = modal.App("fair-benchmark")
volume = modal.Volume.from_name("biplanar-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "monai==1.6.1", "timm==1.0.30",
                 "nibabel==5.4.2", "numpy", "pillow", "tqdm", "scipy")
    .add_local_file(os.path.join(ROOT, "train.py"), f"{PROJECT}/train.py")
    .add_local_dir(os.path.join(ROOT, "configs"), f"{PROJECT}/configs", ignore=["**/__pycache__/**"])
    .add_local_dir(os.path.join(ROOT, "models"), f"{PROJECT}/models", ignore=["**/__pycache__/**"])
    .add_local_dir(os.path.join(ROOT, "utils"), f"{PROJECT}/utils", ignore=["**/__pycache__/**"])
    .add_local_dir(os.path.join(ROOT, "splits"), f"{PROJECT}/splits")
)


def _commit_every(seconds: int, stop: threading.Event):
    while not stop.wait(seconds):
        try:
            volume.commit()
        except Exception as e:  # keep training even if one commit fails
            print(f"[modal] volume commit failed: {e}", flush=True)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _run_training(config_name: str, extra: list[str], limit_s: int) -> dict:
    t0 = time.time()
    subprocess.run(["tar", "-xf", f"{VOL}/data/biplanar_train.tar", "-C", PROJECT], check=True)
    results = f"{VOL}/results"
    os.makedirs(f"{results}/logs", exist_ok=True)
    os.makedirs(f"{results}/saved_models", exist_ok=True)
    if not os.path.islink(f"{PROJECT}/logs"):
        os.symlink(f"{results}/logs", f"{PROJECT}/logs")
    print(f"[modal] {config_name}: data ready in {time.time() - t0:.0f} s on {_gpu_name()}", flush=True)

    stop = threading.Event()
    threading.Thread(target=_commit_every, args=(300, stop), daemon=True).start()
    peak = {"mib": 0}

    def _watch_memory():
        while not stop.wait(5):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.strip()
                peak["mib"] = max(peak["mib"], int(out.splitlines()[0]))
            except Exception:
                pass
    threading.Thread(target=_watch_memory, daemon=True).start()
    cmd = ["python", "train.py", "--config", config_name, "--no_wandb",
           "--save_dir", f"{results}/saved_models", "--num_workers", "3", *extra]
    print("[modal] " + " ".join(cmd), flush=True)
    try:
        rc = subprocess.run(cmd, cwd=PROJECT, timeout=limit_s).returncode
    except subprocess.TimeoutExpired:
        rc = "time limit reached (best weights so far are kept)"
    stop.set()
    volume.commit()
    return {"config": config_name, "returncode": rc, "hours": round((time.time() - t0) / 3600, 3),
            "gpu": _gpu_name(), "peak_gpu_memory_gib": round(peak["mib"] / 1024, 1)}


@app.function(image=image, gpu=GPU, cpu=4.0, memory=16384, timeout=13 * 3600, volumes={VOL: volume})
def train_job(config_name: str, extra: list[str], limit_s: int) -> dict:
    return _run_training(config_name, extra, limit_s)


@app.local_entrypoint()
def timing(config: str = "fair_multiscale_hip", epochs: int = 1):
    """Short timing test: a few epochs of one config."""
    print(train_job.remote(config, ["--epochs", str(epochs)], 1800), flush=True)


@app.local_entrypoint()
def launch(jobs: str, hours: str):
    """Full runs in parallel. jobs: comma-separated config names; hours: matching hard time limits."""
    names = [j.strip() for j in jobs.split(",") if j.strip()]
    limits = [float(h) for h in hours.split(",") if h.strip()]
    assert len(names) == len(limits), "give one time limit (hours) per job"
    assert all(n.startswith("fair_") for n in names), names
    calls = {n: train_job.spawn(n, [], int(h * 3600)) for n, h in zip(names, limits)}
    for n, c in calls.items():
        print(f"{n}: started call {c.object_id} (limit {dict(zip(names, limits))[n]} h)", flush=True)
    for n, c in calls.items():
        print(c.get(), flush=True)
