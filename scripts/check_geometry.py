#!/usr/bin/env python3
"""Check that the X-ray views and the models agree with the 3D geometry. No GPU needed.

Volume axes are (D, H, W) = (anterior-posterior, superior-inferior, left-right).
  AP image:      rows = H, columns = W, rays travel along D
  lateral image: rows = H, columns = D, rays travel along W

Part 1 - data. For every sample, project the mask along D (AP) and along W (lateral) and correlate
each projection with the X-ray as loaded for training. The image should match the projection better
than its left-right mirror. With --flip-ap (what the fair configs use) the AP image is mirrored on
loading, which is the fix for the radiographic display convention of the AP PNGs.

Part 2 - models. With random weights, put a small bright square into one view at several column
positions and record where the output changes (centre of mass of |change| along D and W). For the
AP view the W position must follow the square; for the lateral view the D position must. The test
runs each model with its legacy placement and with geometry="correct".

Usage:
    python scripts/check_geometry.py --part data --flip-ap          # all samples, all datasets
    python scripts/check_geometry.py --part models
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config import CONFIG_REGISTRY  # noqa: E402
from utils.data.dataset import BoneDataset  # noqa: E402

DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double() - a.double().mean()
    b = b.flatten().double() - b.double().mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


# ----------------------------------------------------------------------------- part 1: data
def _first(batch):
    return batch[0]


def check_data(flip_ap: bool, max_per_dataset: int | None, workers: int) -> dict:
    report = {}
    for ds in DATASETS:
        cfg = CONFIG_REGISTRY[f"fair_multiscale_{ds}"]
        manifest = json.load(open(cfg.SPLIT_MANIFEST if hasattr(cfg, "SPLIT_MANIFEST")
                                  else os.path.join(ROOT, "splits", f"{ds}_seed42.json")))["splits"]
        pids = sorted(p for s in manifest.values() for p in s)
        if max_per_dataset:
            pids = pids[:max_per_dataset]
        data = BoneDataset(root_dir=os.path.join(cfg.DATA_ROOT, "train"), img_size_2d=tuple(cfg.IMG_SIZE),
                           img_size_3d=tuple(cfg.VOLUME_SIZE), return_volume=True, include_pids=pids,
                           flip_ap_lr=flip_ap)
        loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=workers,
                                             collate_fn=_first)
        ap_ok = lat_ok = n = 0
        bad = []
        for item in loader:
            vol = (item["vol"][0] > 0.5).float()                       # [D, H, W]
            xa, xb = item["xa"][0], item["xb"][0]                       # [H, W] each
            if vol.sum() == 0:
                continue
            ap_proj = vol.sum(0)                                        # [H, W]
            lat_proj = vol.sum(2).t()[None, None]                       # [1, 1, H, D]
            lat_proj = F.interpolate(lat_proj, size=xb.shape, mode="bilinear", align_corners=False)[0, 0]
            ap_good = _corr(xa, ap_proj) > _corr(xa, ap_proj.flip(-1))
            lat_good = _corr(xb, lat_proj) > _corr(xb, lat_proj.flip(-1))
            ap_ok += ap_good
            lat_ok += lat_good
            n += 1
            if not (ap_good and lat_good):
                bad.append({"pid": item["pid"], "ap_ok": bool(ap_good), "lat_ok": bool(lat_good)})
        report[ds] = {"n": n, "ap_match": ap_ok, "lat_match": lat_ok, "mismatches": bad}
        print(f"{ds:9s} n={n:4d}  AP matches unmirrored projection: {ap_ok:4d} ({100 * ap_ok / max(n, 1):5.1f}%)  "
              f"lateral matches: {lat_ok:4d} ({100 * lat_ok / max(n, 1):5.1f}%)", flush=True)
    return report


# ----------------------------------------------------------------------------- part 2: models
def _build(model_type: str, geometry: str, ds: str):
    from models import DualViewCNN, MultiScaleDualViewCNN, SwinX2S
    cfg = CONFIG_REGISTRY[f"fair_{'multiscale' if model_type == 'multiscale_dualviewcnn' else model_type}_{ds}"]
    if model_type == "dualviewcnn":
        m = DualViewCNN(mid_channels=cfg.MID_CHANNELS, intermediate_depth=cfg.INTERMEDIATE_DEPTH,
                        geometry=geometry, input_width=cfg.IMG_SIZE[1], norm=cfg.DVC_NORM)
        size2d = tuple(cfg.IMG_SIZE)
    elif model_type == "swinx2s":
        m = SwinX2S(img_size=cfg.IMG_SIZE, in_channels=1, out_channels=1, feature_size=cfg.FEATURE_SIZE,
                    depths=cfg.DEPTHS, num_heads=cfg.NUM_HEADS, window_size=cfg.WINDOW_SIZE,
                    norm_name=cfg.NORM_NAME, fusion_depths=cfg.FUSION_DEPTHS, spatial_dims=3,
                    geometry=geometry, output_mode=cfg.SWIN_OUTPUT_MODE,
                    swin_impl=getattr(cfg, "SWIN_IMPL", "legacy"))
        size2d = tuple(cfg.IMG_SIZE[1:])
    else:
        m = MultiScaleDualViewCNN(output_size=cfg.VOLUME_SIZE, base_channels=cfg.BASE_CHANNELS,
                                  num_scales=cfg.NUM_SCALES)
        size2d = tuple(cfg.IMG_SIZE)
    return m.eval(), size2d


def _centroids(delta: torch.Tensor) -> tuple[float, float, float]:
    d = delta.abs()[0, 0].double()                                       # [D, H, W]
    out = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        prof = d.sum(other)
        pos = torch.linspace(0, 1, prof.numel(), dtype=torch.float64)
        out.append(float((prof * pos).sum() / prof.sum()))
    return tuple(out)


def check_models(ds: str, positions: int) -> dict:
    torch.manual_seed(0)
    results = {}
    cols = np.linspace(0.15, 0.85, positions)
    for model_type, modes in (("dualviewcnn", ("legacy", "correct")), ("swinx2s", ("legacy", "correct")),
                              ("multiscale_dualviewcnn", ("n/a",))):
        for mode in modes:
            torch.manual_seed(0)
            model, (h, w) = _build(model_type, mode if mode != "n/a" else "legacy", ds)
            base = torch.full((1, 1, h, w), 0.2)
            with torch.no_grad():
                y0 = model(base, base)
                row = {}
                for view in ("ap", "lat"):
                    d_pos, w_pos = [], []
                    for c in cols:
                        img = base.clone()
                        r0, c0 = int(0.5 * h) - h // 16, int(c * w) - w // 16
                        img[..., r0:r0 + h // 8, max(c0, 0):c0 + w // 8] = 1.0
                        y = model(img, base) if view == "ap" else model(base, img)
                        cd, _, cw = _centroids(y - y0)
                        d_pos.append(cd)
                        w_pos.append(cw)
                    corr_d = float(np.corrcoef(cols, d_pos)[0, 1])
                    corr_w = float(np.corrcoef(cols, w_pos)[0, 1])
                    spread_d, spread_w = float(np.ptp(d_pos)), float(np.ptp(w_pos))
                    # the column must move the output along the correct axis (strongly) and
                    # clearly more than along the other axis
                    if view == "ap":
                        ok = corr_w > 0.9 and spread_w > 2 * spread_d
                    else:
                        ok = corr_d > 0.9 and spread_d > 2 * spread_w
                    row[view] = {"corr_D": round(corr_d, 3), "corr_W": round(corr_w, 3),
                                 "shift_D": round(spread_d, 3), "shift_W": round(spread_w, 3), "ok": ok}
            name = model_type if mode == "n/a" else f"{model_type} ({mode})"
            results[name] = row
            print(f"{name:28s} AP view -> follows W: {'PASS' if row['ap']['ok'] else 'FAIL'} "
                  f"(shift W {row['ap']['shift_W']:.2f} vs D {row['ap']['shift_D']:.2f})   "
                  f"lateral view -> follows D: {'PASS' if row['lat']['ok'] else 'FAIL'} "
                  f"(shift D {row['lat']['shift_D']:.2f} vs W {row['lat']['shift_W']:.2f})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "models", "all"], default="all")
    ap.add_argument("--flip-ap", action="store_true", help="mirror AP images on loading (fair configs)")
    ap.add_argument("--max-per-dataset", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-dataset", default="femur", help="dataset whose sizes the model test uses")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    args = ap.parse_args()
    out = {}
    if args.part in ("data", "all"):
        print(f"== data (AP flipped on loading: {args.flip_ap})")
        out["data"] = check_data(args.flip_ap, args.max_per_dataset, args.workers)
    if args.part in ("models", "all"):
        print(f"== models (random weights, {args.model_dataset} sizes)")
        out["models"] = check_models(args.model_dataset, args.positions)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
