#!/usr/bin/env python3
"""Evaluate the 18 fair-benchmark runs (16-17 Nov 2026) with the same metric code as the paper.

Each run is evaluated with ITS OWN fair config (fair_<model>_<dataset>), so preprocessing, AP flip, view
geometry, Swin implementation, output averaging and the split manifest (deduplicated CTPelvic) are exactly
those used in training. Metric functions are imported from scripts/evaluate_checkpoints.py, unchanged.

Two stages, in this order:
  1. --split val : rescores the validation split and checks that the mean equals the best validation Dice
                   stored in the checkpoint (consistency check). Writes <out>/val/verification.json.
  2. --split test: scores the held-out test split ONCE. Refuses to run unless all 18 validation checks passed,
                   and refuses to overwrite existing test results.
Nothing is trained and no checkpoint is modified.

Usage:
  python scripts/evaluate_fair.py --split val
  python scripts/evaluate_fair.py --split test --pred-dir paper/evidence_fair/test_preds
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config import CONFIG_REGISTRY  # noqa: E402
from train import create_model, load_split_manifest  # noqa: E402
from scripts.evaluate_checkpoints import score, sizes, summarize  # noqa: E402
from utils.data.dataset import BoneDataset  # noqa: E402

DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]
MODELS = ["dualviewcnn", "multiscale", "swinx2s"]
# The run folder of every fair run (relative to the repository root). Excluded on purpose:
#   modal_results_second/.../fair_swinx2s_ctpelvic_20261116_115216 (preempted at epoch 135, rerun),
#   modal_results_second/.../fair_multiscale_hip_20261116_111154 (2-epoch timing test),
#   modal_results/ (morning SwinX2S runs with the old, broken Swin windows).
RUNS = {
    ("dualviewcnn", "femur"): "saved_models/dualviewcnn/femur/fair_dualviewcnn_femur_20261116_124821",
    ("dualviewcnn", "hip"): "saved_models/dualviewcnn/hip/fair_dualviewcnn_hip_20261116_120926",
    ("dualviewcnn", "rib"): "saved_models/dualviewcnn/rib/fair_dualviewcnn_rib_20261116_104529",
    ("dualviewcnn", "verse19"): "saved_models/dualviewcnn/verse19/fair_dualviewcnn_verse19_20261116_104529",
    ("dualviewcnn", "verse20"): "saved_models/dualviewcnn/verse20/fair_dualviewcnn_verse20_20261116_131218",
    ("dualviewcnn", "ctpelvic"): "saved_models/dualviewcnn/ctpelvic/fair_dualviewcnn_ctpelvic_20261116_104529",
    ("multiscale", "femur"): "saved_models/multiscale_dualviewcnn/femur/fair_multiscale_femur_20261117_020007",
    ("multiscale", "hip"): "saved_models/multiscale_dualviewcnn/hip/fair_multiscale_hip_20261117_022014",
    ("multiscale", "rib"): "saved_models/multiscale_dualviewcnn/rib/fair_multiscale_rib_20261117_005149",
    ("multiscale", "verse19"): "saved_models/multiscale_dualviewcnn/verse19/fair_multiscale_verse19_20261116_224640",
    ("multiscale", "verse20"): "saved_models/multiscale_dualviewcnn/verse20/fair_multiscale_verse20_20261117_040355",
    ("multiscale", "ctpelvic"): "saved_models/multiscale_dualviewcnn/ctpelvic/fair_multiscale_ctpelvic_20261116_212955",
    ("swinx2s", "femur"): "saved_models/swinx2s/tiny/femur/fair_swinx2s_femur_20261116_141548",
    ("swinx2s", "hip"): "modal_results_second/results/saved_models/swinx2s/tiny/hip/fair_swinx2s_hip_20261116_115216",
    ("swinx2s", "rib"): "modal_results_second/results/saved_models/swinx2s/tiny/rib/fair_swinx2s_rib_20261116_115214",
    ("swinx2s", "verse19"): "saved_models/swinx2s/tiny/verse19/fair_swinx2s_verse19_20261116_141548",
    ("swinx2s", "verse20"): "modal_results_second/results/saved_models/swinx2s/tiny/verse20/fair_swinx2s_verse20_20261116_115500",
    ("swinx2s", "ctpelvic"): "modal_results_second/results/saved_models/swinx2s/tiny/ctpelvic/fair_swinx2s_ctpelvic_20261116_153412",
}
VAL_TOLERANCE = 5e-4   # max |rescored mean val Dice - Dice stored in the checkpoint|


def load_run(model, ds, device):
    cfg = CONFIG_REGISTRY[f"fair_{model}_{ds}"]
    run_dir = os.path.join(ROOT, RUNS[(model, ds)])
    best = glob.glob(os.path.join(run_dir, "best_epoch*.pth"))
    assert len(best) == 1, f"expected one best checkpoint in {run_dir}, found {best}"
    meta = torch.load(best[0], map_location="cpu", weights_only=False)
    net = create_model(cfg, device)
    net.load_state_dict(torch.load(os.path.join(run_dir, "best_model_weights.pth"), map_location="cpu"),
                        strict=True)
    net.eval()
    info = {
        "config": f"fair_{model}_{ds}", "model": model, "dataset": ds,
        "run_dir": RUNS[(model, ds)], "best_epoch": int(meta["epoch"]),
        "logged_val_dice": float(meta["val_dice"]),
        "params": int(sum(p.numel() for p in net.parameters())),
        "grid": list(sizes(cfg)[1]), "flip_ap_lr": cfg.FLIP_AP_LR,
        "view_geometry": getattr(cfg, "VIEW_GEOMETRY", None), "swin_impl": getattr(cfg, "SWIN_IMPL", None),
        "split_manifest": os.path.relpath(load_split_manifest(cfg)[0], ROOT),
    }
    return cfg, net, info


def dataset_for(cfg, pids):
    s2, s3 = sizes(cfg)
    return BoneDataset(root_dir=os.path.join(cfg.DATA_ROOT, "train"), img_size_2d=s2, img_size_3d=s3,
                       return_volume=True, include_pids=pids, flip_ap_lr=cfg.FLIP_AP_LR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "evidence_fair"))
    ap.add_argument("--pred-dir", default=None, help="save thresholded predictions (test only)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == "test":
        ver = os.path.join(args.out, "val", "verification.json")
        assert os.path.exists(ver), "run --split val first"
        v = json.load(open(ver))
        assert v["all_passed"] and len(v["runs"]) == 18, "validation check did not pass for all 18 runs"
        existing = glob.glob(os.path.join(out_dir, "fair_*.json"))
        assert not existing, f"test results already exist ({len(existing)} files): the test split is scored once"
        if args.pred_dir:
            os.makedirs(args.pred_dir, exist_ok=True)

    verification = []
    started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    for ds in DATASETS:
        for model in MODELS:
            t0 = time.time()
            cfg, net, info = load_run(model, ds, device)
            _, splits = load_split_manifest(cfg)
            cases = score(net, dataset_for(cfg, splits[args.split]), cfg, device,
                          pred_dir=args.pred_dir if args.split == "test" else None, tag=info["config"])
            summary = summarize(cases)
            result = dict(info, split=args.split, n_cases=len(cases), summary=summary, cases=cases,
                          device=str(device), evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
            json.dump(result, open(os.path.join(out_dir, f"{info['config']}.json"), "w"), indent=1)
            line = f"{info['config']:26s} n={len(cases):3d} Dice {summary['dice']:.4f}"
            if args.split == "val":
                diff = abs(summary["dice"] - info["logged_val_dice"])
                ok = diff <= VAL_TOLERANCE
                verification.append({"config": info["config"], "rescored": summary["dice"],
                                     "logged": info["logged_val_dice"], "abs_diff": diff, "passed": ok})
                line += f" | logged {info['logged_val_dice']:.4f} | diff {diff:.1e} {'PASS' if ok else 'FAIL'}"
            else:
                line += (f" | HD95 {summary['hd95_mm']:.2f} mm (n={summary['hd95_mm_n']})"
                         f" | NSD {summary['nsd_1.5mm']:.3f}")
            print(line + f" | {time.time() - t0:.0f} s", flush=True)
            del net
            torch.cuda.empty_cache()

    if args.split == "val":
        all_ok = all(v["passed"] for v in verification)
        json.dump({"all_passed": all_ok, "tolerance": VAL_TOLERANCE, "started": started, "runs": verification},
                  open(os.path.join(out_dir, "verification.json"), "w"), indent=1)
        print("VALIDATION CHECK:", "ALL 18 PASSED" if all_ok else "SOME FAILED", flush=True)
    else:
        json.dump({"test_scored_once_at": started, "runs": len(MODELS) * len(DATASETS),
                   "note": "fair runs; test split scored once after the validation check passed"},
                  open(os.path.join(out_dir, "test_run_info.json"), "w"), indent=1)
        print("TEST EVALUATION DONE", flush=True)


if __name__ == "__main__":
    main()
