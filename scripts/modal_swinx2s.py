"""Train the fair SwinX2S models (configs fair_swinx2s_<dataset>) on Modal GPUs. Only SwinX2S.

What runs remotely: train.py, unchanged, with --no_wandb. Code is copied from configs/, models/,
utils/, splits/ and train.py only (never .env). Data come from a tar of the six datasets' train/
folders kept in the private Modal Volume "biplanar-data". Logs and best weights are written to the
same Volume under results/, which is committed every few minutes so progress survives a crash.

Usage (from the repository root, with the Modal client in ~/modal-env):
  # 1. upload data once
  tar -cf /tmp/biplanar_train.tar "datasets/Total Segmentator/Femur/train" ... (see README in reports)
  ~/modal-env/bin/modal volume put biplanar-data /tmp/biplanar_train.tar data/biplanar_train.tar
  # 2. short timing test (2 epochs on VerSe20, 1 epoch on hip)
  ~/modal-env/bin/modal run scripts/modal_swinx2s.py::smoke
  # 3. all six datasets in parallel, detached (keeps running if this machine disconnects)
  ~/modal-env/bin/modal run --detach scripts/modal_swinx2s.py::launch
  # 4. download results
  ~/modal-env/bin/modal volume get biplanar-data results/ modal_results/
"""
import os
import subprocess
import threading
import time

import modal

GPU = os.environ.get("SWINX2S_GPU", "A100-80GB")
DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]
# Hard time limit per job (seconds). This caps the cost: 200 epochs are expected to need about
# femur 3.7 h, hip 3.4 h, rib 3.7 h, VerSe20 0.6 h, VerSe19 4.7 h, CTPelvic 4.9 h on an L40S.
# If a job reaches its limit, training stops and the best weights so far are kept.
TIMEOUTS = {"femur": int(3.9 * 3600), "hip": int(3.6 * 3600), "rib": int(3.9 * 3600),
            "verse20": int(0.75 * 3600), "verse19": int(5.0 * 3600), "ctpelvic": int(5.3 * 3600)}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = "/root/project"
VOL = "/vol"

app = modal.App("swinx2s-fair")
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


def _run_training(config_name: str, extra: list[str], limit_s: int) -> dict:
    t0 = time.time()
    # data on local disk (fast), results on the Volume
    subprocess.run(["tar", "-xf", f"{VOL}/data/biplanar_train.tar", "-C", PROJECT], check=True)
    results = f"{VOL}/results"
    os.makedirs(f"{results}/logs", exist_ok=True)
    os.makedirs(f"{results}/saved_models", exist_ok=True)
    if not os.path.islink(f"{PROJECT}/logs"):
        os.symlink(f"{results}/logs", f"{PROJECT}/logs")
    print(f"[modal] data ready in {time.time() - t0:.0f} s; GPU {GPU}", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

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
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip(),
            "peak_gpu_memory_gib": round(peak["mib"] / 1024, 1)}


@app.function(image=image, gpu=GPU, cpu=4.0, memory=16384, timeout=13 * 3600, volumes={VOL: volume})
def train_job(config_name: str, extra: list[str], limit_s: int) -> dict:
    return _run_training(config_name, extra, limit_s)


@app.local_entrypoint()
def smoke():
    """Timing test: VerSe20 (96x96x128 grid) for 2 epochs and hip (128x128x160) for 1 epoch."""
    calls = [train_job.spawn("fair_swinx2s_verse20", ["--epochs", "2"], 1800),
             train_job.spawn("fair_swinx2s_hip", ["--epochs", "1"], 1800)]
    for c in calls:
        print(c.get(), flush=True)


@app.local_entrypoint()
def timing_hip():
    """One hip epoch (largest grid) to compare GPU types; set SWINX2S_GPU."""
    print(train_job.remote("fair_swinx2s_hip", ["--epochs", "1"], 1800), flush=True)


@app.local_entrypoint()
def launch(datasets: str = "femur,hip,rib,verse20"):
    """SwinX2S fair runs in parallel (comma-separated datasets), each with its own time limit."""
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    assert all(d in DATASETS for d in names), names
    calls = {ds: train_job.spawn(f"fair_swinx2s_{ds}", [], TIMEOUTS[ds]) for ds in names}
    for ds, c in calls.items():
        print(f"{ds}: started call {c.object_id}", flush=True)
    for ds, c in calls.items():
        print(c.get(), flush=True)
