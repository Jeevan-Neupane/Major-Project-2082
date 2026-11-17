#!/usr/bin/env python3
"""Stage the 18 fair-benchmark models (paper version 2) for upload to the Hugging Face model repo.

Writes a folder that mirrors what will be added to `jeevan1234/biplanar-bone-reconstruction-models`:
  fair_v2/<model>/<dataset>/best_model_weights.pth   (state dict, copied unchanged)
  fair_v2/<model>/<dataset>/run_info.json            (config, run, best epoch, settings, val/test summary)
  fair_v2/code/...                                   (exact code to build and load the models)
  fair_v2/results_fair.json                          (test statistics of the paper)
  README.md                                          (model card of these models only)
  checkpoint_inventory.json                          (list of the uploaded files)
The upload (scripts/upload_hf_fair_v2.py) removes the files of the earlier models from the repo. No data are included.

Usage: python scripts/stage_hf_fair_v2.py --stage /tmp/hf_stage
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.evaluate_fair import RUNS, DATASETS, MODELS  # noqa: E402

CODE = ["train.py", "requirements.txt", "configs/__init__.py", "configs/config.py", "configs/args_parser.py",
        "models/__init__.py", "models/transvert.py", "models/multiscale_dualview_cnn.py", "models/swinx2s.py",
        "models/swinX2S/__init__.py", "models/swinX2S/swin_transformer.py", "models/swinX2S/transfusion.py",
        "models/swinX2S/unetr_blocks.py", "utils/__init__.py", "utils/data/__init__.py", "utils/data/dataset.py",
        "utils/training/__init__.py", "utils/training/checkpoint.py", "utils/training/logger.py",
        "utils/training/losses.py", "utils/training/metrics.py", "utils/training/trainer.py",
        "scripts/evaluate_fair.py", "scripts/evaluate_checkpoints.py", "scripts/check_geometry.py",
        "scripts/__init__.py"]
LABEL = {"femur": "Femur", "hip": "Hip", "rib": "Rib", "verse19": "VerSe19", "verse20": "VerSe20", "ctpelvic": "CTPelvic"}


def model_card(results):
    s = results["summary"]
    rows = "\n".join(
        f"| {LABEL[ds]} | {s[f'dualviewcnn/{ds}']['n']} | " + " | ".join(
            f"{100 * s[f'{m}/{ds}']['dice']:.1f}" for m in MODELS) + " |" for ds in DATASETS)
    return f"""---
library_name: pytorch
tags:
- 3d-reconstruction
- medical-imaging
- biplanar-x-ray
---
# Biplanar bone reconstruction models

The 18 trained models (DualViewCNN, MultiScale and SwinX2S on femur, hip, rib, VerSe19, VerSe20 and CTPelvic) of the paper
"Reconstructing 3D Bone Shapes from Biplanar X-rays: Three Deep Learning Models Tested on Six Datasets".

Code, data splits and per-case results: https://github.com/Jeevan-Neupane/biplanar-bone-reconstruction (release v1.0).

How they were trained:
- Both X-ray views are placed in 3D according to the X-ray geometry, and the AP image is mirrored on loading.
- All models share the same training settings: up to 200 epochs, batch size 2, early stopping after 40 epochs without a
  better validation Dice, and the checkpoint with the best validation Dice.
- SwinX2S (Tiny) is adapted from the official Swin-X2S code and uses its averaging of the two outputs.
- The CTPelvic split keeps one copy of each scan.

Test Dice (%) on the held-out splits, scored once on 17 November 2026:

| Dataset | n | DualViewCNN | MultiScale | SwinX2S |
|---|---|---|---|---|
{rows}

Layout:
- `fair_v2/<model>/<dataset>/best_model_weights.pth`: PyTorch state dict of the checkpoint with the best validation Dice.
- `fair_v2/<model>/<dataset>/run_info.json`: config name, training run, best epoch, settings, validation/test summary.
- `fair_v2/code/`: the exact code used (configs `fair_<model>_<dataset>`, e.g. `fair_multiscale_hip`).
- `fair_v2/results_fair.json`: test statistics reported in the paper.

Loading a model (custom architectures, not Transformers AutoModel packages):

```python
import sys, torch
sys.path.insert(0, "fair_v2/code")
from configs.config import CONFIG_REGISTRY
from train import create_model
cfg = CONFIG_REGISTRY["fair_multiscale_hip"]
model = create_model(cfg, torch.device("cpu"))
model.load_state_dict(torch.load("fair_v2/multiscale/hip/best_model_weights.pth", map_location="cpu"))
model.eval()
```

Inputs must be prepared as in training: `utils/data/dataset.py`, `BoneDataset(..., flip_ap_lr=cfg.FLIP_AP_LR)` with the
image and grid sizes of the config.

Limitations:
- Trained and tested only on simulated X-ray images (DRRs) from public CT datasets.
- Not clinically validated.
- Splits are by sample, so vertebrae or femurs from the same scan can fall into different splits.
- Each model was trained once (one seed). The test splits had been scored once before, for an earlier version of these models.

Licences:
- Training data: TotalSegmentator (CC BY 4.0), VerSe (CC BY-SA 4.0) and CTPelvic1K (CC BY-NC-SA). Models trained on CTPelvic1K
  must not be used commercially.
- The SwinX2S code is adapted from the official Swin-X2S repository (https://github.com/liukuan5625/Swin-X2S, MIT licence);
  see `fair_v2/code/THIRD_PARTY_NOTICES.md`.
- No other licence is granted. No dataset files are included.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    st = args.stage
    shutil.rmtree(st, ignore_errors=True)
    os.makedirs(st)
    ev = os.path.join(ROOT, "paper", "evidence_fair")
    results = json.load(open(os.path.join(ev, "results_fair.json")))
    new_files = []
    for (model, ds), run_dir in RUNS.items():
        dst = os.path.join(st, "fair_v2", model, ds)
        os.makedirs(dst)
        shutil.copy2(os.path.join(ROOT, run_dir, "best_model_weights.pth"), os.path.join(dst, "best_model_weights.pth"))
        t = json.load(open(os.path.join(ev, "test", f"fair_{model}_{ds}.json")))
        v = json.load(open(os.path.join(ev, "val", f"fair_{model}_{ds}.json")))
        info = {k: t[k] for k in ["config", "model", "dataset", "best_epoch", "logged_val_dice", "params", "grid",
                                  "flip_ap_lr", "view_geometry", "swin_impl", "split_manifest"]}
        info.update(run=os.path.basename(run_dir), val_summary=v["summary"], test_summary=t["summary"],
                    test_scored_at=t["evaluated_at"],
                    trained_on="NVIDIA L40S (Modal)" if run_dir.startswith("modal_results") else "NVIDIA RTX 5090")
        json.dump(info, open(os.path.join(dst, "run_info.json"), "w"), indent=1)
        new_files += [f"fair_v2/{model}/{ds}/best_model_weights.pth", f"fair_v2/{model}/{ds}/run_info.json"]
    for rel in CODE:
        src = os.path.join(ROOT, rel)
        if rel == "scripts/__init__.py" and not os.path.exists(src):
            os.makedirs(os.path.join(st, "fair_v2", "code", "scripts"), exist_ok=True)
            open(os.path.join(st, "fair_v2", "code", rel), "w").close()
        else:
            os.makedirs(os.path.dirname(os.path.join(st, "fair_v2", "code", rel)), exist_ok=True)
            shutil.copy2(src, os.path.join(st, "fair_v2", "code", rel))
        new_files.append(f"fair_v2/code/{rel}")
    shutil.copytree(os.path.join(ROOT, "splits"), os.path.join(st, "fair_v2", "code", "splits"))
    new_files += [f"fair_v2/code/splits/{f}" for f in sorted(os.listdir(os.path.join(ROOT, "splits")))]
    mit = open(os.path.join(ROOT, "paper", "arxiv_v2", "anc_README.md")).read()
    mit = mit[mit.index("## Licence of the adapted Swin-X2S code"):]
    open(os.path.join(st, "fair_v2", "code", "THIRD_PARTY_NOTICES.md"), "w").write(
        "# Third-party notices\n\n`models/swinx2s.py` and `models/swinX2S/` are adapted from Swin-X2S.\n\n" + mit)
    new_files.append("fair_v2/code/THIRD_PARTY_NOTICES.md")
    shutil.copy2(os.path.join(ev, "results_fair.json"), os.path.join(st, "fair_v2", "results_fair.json"))
    new_files.append("fair_v2/results_fair.json")
    open(os.path.join(st, "README.md"), "w").write(model_card(results))
    inv = [{"path": f, "bytes": os.path.getsize(os.path.join(st, f)), "uploaded": True} for f in new_files]
    json.dump(inv, open(os.path.join(st, "checkpoint_inventory.json"), "w"), indent=1)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(st) for f in fs)
    print(f"staged {len(new_files)} new files + README + inventory in {st} ({total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
