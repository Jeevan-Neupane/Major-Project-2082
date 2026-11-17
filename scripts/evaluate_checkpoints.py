"""Evaluate the selected (best-validation) checkpoint of each final run.

For every run this re-scores the validation split (a consistency check against
the training log) and scores the held-out test split once.  Per-case metrics
use the same thresholded overlap code as training, plus HD95 and ASSD in mm on
the resampled output grid.  Nothing is trained and no checkpoint is modified.

Usage:
  python scripts/evaluate_checkpoints.py --out paper/evidence/test \
      --pred-dir /path/to/scratch/preds
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train import create_model, load_split_manifest  # noqa: E402
from configs import get_config  # noqa: E402
from utils.data.dataset import BoneDataset, collate_bone  # noqa: E402
from utils.training.metrics import compute_segmentation_metrics  # noqa: E402
from utils.training.trainer import match_target_to_output  # noqa: E402

DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]
MODELS = ["dualviewcnn", "multiscale_dualviewcnn", "swinx2s"]
RUN_STAMP = {
    ("femur", "dualviewcnn"): "20261103_012936",
    ("femur", "multiscale_dualviewcnn"): "20261103_162259",
    ("femur", "swinx2s"): "20261103_162259",
    ("hip", None): "20261103_210359",
    ("rib", None): "20261104_003130",
    ("verse19", None): "20261104_043330",
    ("verse20", None): "20261104_110801",
    ("ctpelvic", None): "20261110_081701",
}


def run_id(ds, model):
    stamp = RUN_STAMP.get((ds, model)) or RUN_STAMP[(ds, None)]
    return f"{model}_{ds}_{stamp}"


def sizes(config):
    if config.MODEL_TYPE == "swinx2s":
        return config.IMG_SIZE[1:], tuple(config.IMG_SIZE)
    if config.MODEL_TYPE == "multiscale_dualviewcnn":
        return tuple(config.IMG_SIZE), tuple(config.VOLUME_SIZE)
    return tuple(config.IMG_SIZE), (config.INTERMEDIATE_DEPTH * 4, *config.IMG_SIZE)


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


NSD_TOLERANCE_MM = 1.5


def surface_dice(pred, gt, spacing, tol=NSD_TOLERANCE_MM):
    """Normalized surface Dice (Nikolov et al. 2021): share of the boundary voxels of
    both masks lying within `tol` mm of the other mask's boundary. None if a mask is empty."""
    if not pred.any() or not gt.any():
        return None
    sp, sg = surface(pred), surface(gt)
    d_pg = ndimage.distance_transform_edt(~sg, sampling=spacing)[sp]
    d_gp = ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(((d_pg <= tol).sum() + (d_gp <= tol).sum()) / (len(d_pg) + len(d_gp)))


def surface_distances(pred, gt, spacing):
    """HD95 and ASSD (mm). None when either mask is empty."""
    if not pred.any() or not gt.any():
        return None, None
    sp, sg = surface(pred), surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dt_g[sp], dt_p[sg]
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    return float(hd95), float(assd)


@torch.no_grad()
def score(model, dataset, config, device, pred_dir=None, tag=None):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_bone)
    cases, preds, gts, pids = [], [], [], []
    for batch in loader:
        xa = batch["xa"].to(device)
        xb = batch["xb"].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            logits = model(xa, xb)
        logits = logits.float()
        target = match_target_to_output(batch["vol"].to(device), logits)
        m = {k: float(v[0]) for k, v in compute_segmentation_metrics(logits, target).items()}
        pred = (torch.sigmoid(logits) > 0.5)[0, 0].cpu().numpy()
        gt = (target > 0.5)[0, 0].cpu().numpy()
        spacing = np.asarray(batch["voxel_spacing"][0], dtype=np.float64)
        m["hd95_mm"], m["assd_mm"] = surface_distances(pred, gt, spacing)
        m["nsd_1.5mm"] = surface_dice(pred, gt, spacing)
        m["voxel_spacing_mm"] = [float(x) for x in spacing]
        m["pid"] = batch["pid"][0]
        m["pred_voxels"], m["gt_voxels"] = int(pred.sum()), int(gt.sum())
        cases.append(m)
        if pred_dir:
            preds.append(np.packbits(pred))
            gts.append(np.packbits(gt))
            pids.append(m["pid"])
    if pred_dir:
        np.savez_compressed(os.path.join(pred_dir, f"{tag}.npz"), pred=np.stack(preds),
                            gt=np.stack(gts), pids=np.array(pids), shape=np.array(pred.shape))
    return cases


def summarize(cases):
    out = {"n": len(cases)}
    for k in ["dice", "iou", "precision", "recall"]:
        v = np.array([c[k] for c in cases])
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    for k in ["hd95_mm", "assd_mm", "nsd_1.5mm"]:
        v = np.array([c[k] for c in cases if c[k] is not None])
        out[k] = float(v.mean()) if len(v) else None
        out[k + "_median"] = float(np.median(v)) if len(v) else None
        out[k + "_n"] = int(len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated run ids")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.pred_dir:
        os.makedirs(args.pred_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds in DATASETS:
        for model_name in MODELS:
            rid = run_id(ds, model_name)
            if args.only and rid not in args.only.split(","):
                continue
            t0 = time.time()
            config = get_config(f"{model_name}_{ds}")
            ckpt_dirs = glob.glob(os.path.join(ROOT, "saved_models", model_name, "**", rid), recursive=True)
            assert len(ckpt_dirs) == 1, (rid, ckpt_dirs)
            ckpt_dir = ckpt_dirs[0]
            best = glob.glob(os.path.join(ckpt_dir, "best_epoch*.pth"))
            assert len(best) == 1, best
            meta = torch.load(best[0], map_location="cpu", weights_only=False)

            model = create_model(config, device)
            state = torch.load(os.path.join(ckpt_dir, "best_model_weights.pth"),
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            _, splits = load_split_manifest(config)
            size2d, size3d = sizes(config)
            mk = lambda pids: BoneDataset(root_dir=os.path.join(config.DATA_ROOT, "train"),
                                          img_size_2d=size2d, img_size_3d=size3d,
                                          return_volume=True, include_pids=pids,
                                          flip_ap_lr=getattr(config, "FLIP_AP_LR", False))
            val_cases = score(model, mk(splits["val"]), config, device)
            test_cases = score(model, mk(splits["test"]), config, device,
                               pred_dir=args.pred_dir, tag=rid)
            result = {
                "run_id": rid, "model": model_name, "dataset": ds,
                "model_size": getattr(config, "MODEL_SIZE", None),
                "checkpoint": os.path.relpath(best[0], ROOT),
                "best_epoch": int(meta["epoch"]), "logged_val_dice": float(meta["val_dice"]),
                "params": n_params, "grid": list(size3d),
                "val": summarize(val_cases), "test": summarize(test_cases),
                "test_cases": test_cases, "val_cases": val_cases,
            }
            with open(os.path.join(args.out, f"{rid}.json"), "w") as f:
                json.dump(result, f, indent=1)
            v, t = result["val"], result["test"]
            print(f"{rid:48s} val {100*v['dice']:6.2f} (logged {100*result['logged_val_dice']:6.2f}) "
                  f"test {100*t['dice']:6.2f}±{100*t['dice_sd']:5.2f} n={t['n']:3d} "
                  f"HD95 {t['hd95_mm'] if t['hd95_mm'] is None else round(t['hd95_mm'], 2)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del model, state
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
