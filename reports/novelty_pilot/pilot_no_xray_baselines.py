"""CPU pilot: how far do baselines that barely use the X-rays get, on the VALIDATION split?

1. Mean shape: average of all training masks, thresholded (threshold picked on train only).
   It ignores the X-rays completely.
2. Retrieval: for each val case, find the training case whose two X-rays look most similar
   (pixel distance on 64x64 images) and copy its mask. No learning at all.

Masks are preprocessed exactly like utils/data/dataset.py (min-max, trilinear resize to the
MultiScale VOLUME_SIZE, threshold 0.5). Test data are not touched. For CTPelvic the cleaned split
(splits/ctpelvic_seed42_dedup.json) is used so val has no copies of training cases.
"""
import csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/home/ubuntu/Major-Project-2082"
DS = {  # name: (data root, volume size used by MultiScale, split file)
    "femur": ("datasets/Total Segmentator/Femur", (96, 96, 128), "femur_seed42.json"),
    "hip": ("datasets/Total Segmentator/Hip", (128, 128, 160), "hip_seed42.json"),
    "rib": ("datasets/Total Segmentator/Rib", (128, 128, 160), "rib_seed42.json"),
    "verse19": ("datasets/Verse/verse19", (96, 96, 128), "verse19_seed42.json"),
    "verse20": ("datasets/Verse/verse20", (96, 96, 128), "verse20_seed42.json"),
    "ctpelvic": ("datasets/ctpelvik", (128, 128, 160), "ctpelvic_seed42_dedup.json"),
}


def load_mask(args):
    path, size = args
    v = nib.load(path).get_fdata(dtype=np.float32)
    lo, hi = float(v.min()), float(v.max())
    v = (v - lo) / (hi - lo + 1e-8) if hi - lo > 1e-8 else np.zeros_like(v)
    t = F.interpolate(torch.from_numpy(v)[None, None], size=size, mode="trilinear", align_corners=False)
    return (t[0, 0].numpy() > 0.5)


def load_pair(sample_dir):
    out = []
    for pat in ("*ap*.png", "*lat*.png"):
        im = Image.open(sorted(glob.glob(os.path.join(sample_dir, "x-ray", pat)))[0]).convert("L").resize((64, 64))
        a = np.asarray(im, dtype=np.float32)
        out.append((a - a.mean()) / (a.std() + 1e-6))
    return np.concatenate([o.ravel() for o in out])


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def best_val_dice(model, name):
    logs = sorted(glob.glob(f"{ROOT}/logs/{model}_{name}_2026*.csv"))
    if not logs:
        return float("nan")
    rows = list(csv.DictReader(open(logs[-1])))
    return max(float(r["val_dice"]) for r in rows)


def main():
    pool = ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    results = {}
    for name, (rel, size, split_file) in DS.items():
        sp = json.load(open(f"{ROOT}/splits/{split_file}"))["splits"]
        store = f"{ROOT}/{rel}/train"
        mpath = lambda p: sorted(glob.glob(f"{store}/{p}/ct-scan/*.nii.gz"))[0]
        tr, va = sp["train"], sp["val"]
        tr_masks = list(pool.map(load_mask, [(mpath(p), size) for p in tr], chunksize=4))
        va_masks = list(pool.map(load_mask, [(mpath(p), size) for p in va], chunksize=4))
        # 1. mean shape, threshold chosen on TRAIN only
        mean = np.zeros(tr_masks[0].shape, np.float32)
        for m in tr_masks:
            mean += m
        mean /= len(tr_masks)
        ths = [0.2, 0.3, 0.4, 0.5, 0.6]
        t_best = max(ths, key=lambda t: np.mean([dice(mean > t, m) for m in tr_masks[:200]]))
        mean_val = [dice(mean > t_best, m) for m in va_masks]
        # 2. retrieval by X-ray similarity
        tr_x = np.stack([load_pair(f"{store}/{p}") for p in tr])
        va_x = np.stack([load_pair(f"{store}/{p}") for p in va])
        d = (va_x ** 2).sum(1)[:, None] + (tr_x ** 2).sum(1)[None, :] - 2.0 * va_x @ tr_x.T
        nn = d.argmin(1)
        ret_val = [dice(tr_masks[j], m) for j, m in zip(nn, va_masks)]
        results[name] = {
            "n_train": len(tr), "n_val": len(va), "threshold": t_best,
            "mean_shape_dice": float(np.mean(mean_val)), "retrieval_dice": float(np.mean(ret_val)),
            "multiscale_best_val_dice": best_val_dice("multiscale_dualviewcnn", name),
            "dualviewcnn_best_val_dice": best_val_dice("dualviewcnn", name),
            "swinx2s_best_val_dice": best_val_dice("swinx2s", name),
        }
        r = results[name]
        print(f"{name:9s} val={len(va):3d}  mean-shape {100*r['mean_shape_dice']:5.1f}  retrieval {100*r['retrieval_dice']:5.1f}  "
              f"| models (best val): MultiScale {100*r['multiscale_best_val_dice']:5.1f}  "
              f"DualViewCNN {100*r['dualviewcnn_best_val_dice']:5.1f}  SwinX2S {100*r['swinx2s_best_val_dice']:5.1f}", flush=True)
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
