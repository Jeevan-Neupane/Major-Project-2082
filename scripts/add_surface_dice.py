"""Add NSD@1.5mm (and the voxel spacing) to existing test results without re-running
the models. Uses the binary test predictions saved by evaluate_checkpoints.py
(--pred-dir) and recomputes each case's voxel spacing exactly as the data loader does.
As a check, it recomputes HD95 and requires it to match the stored value.

Usage: python scripts/add_surface_dice.py --pred-dir <dir> [--results paper/evidence/test]
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import get_config  # noqa: E402
from scripts.evaluate_checkpoints import summarize, surface_dice, surface_distances  # noqa: E402


def loader_spacing(mask_path, grid):
    """Voxel spacing after resampling to `grid`, as in BoneDataset._load_volume_resized."""
    nii = nib.load(mask_path)
    zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
    shape = np.array(nii.shape[:3], dtype=np.float32)
    return (zooms * (shape / np.array(grid, dtype=np.float32))).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--results", default=os.path.join(ROOT, "paper", "evidence", "test"))
    args = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        res = json.load(open(path))
        cfg = get_config(f"{res['model']}_{res['dataset']}")
        z = np.load(os.path.join(args.pred_dir, f"{res['run_id']}.npz"))
        shape = tuple(int(x) for x in z["shape"])
        n = int(np.prod(shape))
        index = {str(p): i for i, p in enumerate(z["pids"])}
        worst = 0.0
        for case in res["test_cases"]:
            i = index[case["pid"]]
            pred = np.unpackbits(z["pred"][i])[:n].reshape(shape).astype(bool)
            gt = np.unpackbits(z["gt"][i])[:n].reshape(shape).astype(bool)
            masks = glob.glob(os.path.join(cfg.DATA_ROOT, "train", case["pid"], "ct-scan", "*.nii*"))
            assert len(masks) == 1, masks
            sp = loader_spacing(masks[0], res["grid"])
            hd95, _ = surface_distances(pred, gt, sp)
            worst = max(worst, abs(hd95 - case["hd95_mm"]))
            case["nsd_1.5mm"] = surface_dice(pred, gt, sp)
            case["voxel_spacing_mm"] = [float(x) for x in sp]
        assert worst < 1e-3, (res["run_id"], worst)  # spacing reconstruction reproduces HD95
        res["test"] = summarize(res["test_cases"])
        json.dump(res, open(path, "w"), indent=1)
        t = res["test"]
        print(f"{res['run_id']:48s} NSD@1.5mm {100*t['nsd_1.5mm']:6.2f}  HD95 mean {t['hd95_mm']:6.2f} median {t['hd95_mm_median']:6.2f}  (HD95 check max diff {worst:.1e})", flush=True)


if __name__ == "__main__":
    main()
