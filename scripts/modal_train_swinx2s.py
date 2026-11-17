"""Train the fixed SwinX2S models (configs fair_swinx2s_<dataset>) on Modal GPUs.

Only SwinX2S runs here. Each dataset is one Modal function call on its own GPU. The code in
configs/, models/, utils/, splits/ and train.py is copied into the image; the training folders come
from the private Modal volume "bone-data" (bone_data_train.tar); logs and best weights go to the
private volume "bone-results".

Usage (from the repository root, with the Modal client in ~/modal-env):
  # short timing test: 2 epochs on VerSe20
  ~/modal-env/bin/modal run scripts/modal_train_swinx2s.py --datasets verse20 --epochs 2 --gpu A100-80GB
  # full runs, detached so they keep going after this terminal closes
  ~/modal-env/bin/modal run --detach scripts/modal_train_swinx2s.py --datasets femur,hip,rib,verse19,verse20,ctpelvic
  # download results afterwards
  ~/modal-env/bin/modal volume get bone-results / ./modal_results
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"

app = modal.App("bone-swinx2s-fair")
data_vol = modal.Volume.from_name("bone-data")
results_vol = modal.Volume.from_name("bone-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "numpy", "nibabel==5.4.2", "monai==1.6.1",
                 "timm==1.0.30", "Pillow", "tqdm", "scipy")
    .add_local_dir(PROJECT / "configs", f"{REMOTE}/configs", ignore=["__pycache__"])
    .add_local_dir(PROJECT / "models", f"{REMOTE}/models", ignore=["__pycache__"])
    .add_local_dir(PROJECT / "utils", f"{REMOTE}/utils", ignore=["__pycache__"])
    .add_local_dir(PROJECT / "splits", f"{REMOTE}/splits")
    .add_local_file(PROJECT / "train.py", f"{REMOTE}/train.py")
)


def _commit_every(vol, stop, seconds=600):
    while not stop.wait(seconds):
        try:
            vol.commit()
        except Exception as e:  # keep training even if one commit fails
            print(f"[modal] volume commit failed: {e}", flush=True)


def _train(dataset: str, epochs: int | None, tag: str) -> dict:
    t0 = time.time()
    # 1. training data onto the container's local disk
    subprocess.run(["tar", "-xf", "/data/bone_data_train.tar", "-C", REMOTE], check=True)
    # 2. logs and checkpoints straight into the results volume
    run_root = Path("/results") / tag
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "saved_models").mkdir(parents=True, exist_ok=True)
    logs_link = Path(REMOTE) / "logs"
    if not logs_link.exists():
        logs_link.symlink_to(run_root / "logs")
    cmd = ["python", "train.py", "--config", f"fair_swinx2s_{dataset}", "--no_wandb",
           "--save_dir", str(run_root / "saved_models"), "--num_workers", "4"]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    print("[modal] " + " ".join(cmd), flush=True)
    stop = threading.Event()
    committer = threading.Thread(target=_commit_every, args=(results_vol, stop), daemon=True)
    committer.start()
    env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_MININTERVAL="30")
    proc = subprocess.run(cmd, cwd=REMOTE, env=env)
    stop.set()
    results_vol.commit()
    return {"dataset": dataset, "returncode": proc.returncode, "minutes": round((time.time() - t0) / 60, 1),
            "results_dir": str(run_root)}


common = dict(image=image, volumes={"/data": data_vol, "/results": results_vol}, cpu=4.0, memory=24 * 1024,
              timeout=12 * 3600)


@app.function(gpu="A100-80GB", **common)
def train_a100(dataset: str, epochs: int | None = None, tag: str = "run") -> dict:
    return _train(dataset, epochs, tag)


@app.function(gpu="L40S", **common)
def train_l40s(dataset: str, epochs: int | None = None, tag: str = "run") -> dict:
    return _train(dataset, epochs, tag)


@app.function(gpu="H100", **common)
def train_h100(dataset: str, epochs: int | None = None, tag: str = "run") -> dict:
    return _train(dataset, epochs, tag)


@app.local_entrypoint()
def main(datasets: str = "verse20", epochs: int = 0, gpu: str = "A100-80GB", tag: str = ""):
    fn = {"A100-80GB": train_a100, "L40S": train_l40s, "H100": train_h100}[gpu]
    tag = tag or time.strftime("fair_swinx2s_%Y%m%d_%H%M%S")
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    calls = [(d, fn.spawn(d, epochs or None, f"{tag}/{d}")) for d in names]
    print(f"[local] started {len(calls)} run(s) on {gpu}, results under bone-results:/{tag}/", flush=True)
    for d, call in calls:
        try:
            print(f"[local] {d}: {call.get()}", flush=True)
        except Exception as e:
            print(f"[local] {d}: FAILED {type(e).__name__}: {e}", flush=True)
