"""
Clinical metrics for vertebra bone reconstruction.

Measurements (via PCA axis spans):
  - **Height** (mm): vertebral body height (smallest PCA axis)
  - **Width** (mm): vertebral body width (largest PCA axis)
  - **Depth** (mm): anterior-posterior depth (second PCA axis)

All measurements use actual voxel spacing from the NIfTI header.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional


def compute_verse_metrics(
    mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute clinical metrics for a single vertebra mask.

    Args:
        mask: Binary 3D numpy array.
        spacing: Voxel spacing (z, y, x) in mm.

    Returns:
        (height, width, depth) — any value may be *None* if the mask
        has fewer than 500 non-zero voxels.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < 500:
        return None, None, None

    coords_mm = coords * np.asarray(spacing, dtype=np.float64)

    pca = PCA(n_components=3)
    pca.fit(coords_mm)
    projected = pca.transform(coords_mm)
    spans = projected.max(axis=0) - projected.min(axis=0)

    # PC1 = largest extent → width
    # PC2 = second extent  → depth (anterior-posterior)
    # PC3 = smallest extent → height (superior-inferior body height)
    width = float(spans[0])
    depth = float(spans[1])
    height = float(spans[2])

    return height, width, depth


def evaluate_batch_clinical_metrics(
    pred_volumes: torch.Tensor,
    gt_volumes: torch.Tensor,
    voxel_spacings: Optional[List[np.ndarray]] = None,
    pids: Optional[List[str]] = None,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Evaluate vertebra clinical metrics for a batch.

    Args:
        pred_volumes: [B, 1, D, H, W] tensor (probabilities 0–1).
        gt_volumes:   [B, 1, D, H, W] tensor.
        voxel_spacings: list of per-sample spacing arrays (preferred).
        pids: list of patient IDs.
        spacing: fallback spacing when *voxel_spacings* is not given.

    Returns:
        dict with MAE values + ``_per_sample`` list for detailed logging.
    """
    pred_np = pred_volumes.detach().cpu().numpy()
    gt_np = gt_volumes.detach().cpu().numpy()
    B = pred_np.shape[0]

    height_errors, width_errors, depth_errors = [], [], []
    per_sample = []

    for i in range(B):
        sp = np.asarray(
            voxel_spacings[i] if voxel_spacings and i < len(voxel_spacings)
            else spacing,
            dtype=np.float64,
        )
        pid = pids[i] if pids and i < len(pids) else f"sample_{i}"

        pred_mask = (pred_np[i, 0] > 0.5).astype(np.uint8)
        gt_mask = (gt_np[i, 0] > 0.5).astype(np.uint8)

        pred_h, pred_w, pred_d = compute_verse_metrics(pred_mask, sp)
        gt_h, gt_w, gt_d = compute_verse_metrics(gt_mask, sp)

        if pred_h is not None and gt_h is not None:
            height_errors.append(abs(pred_h - gt_h))
        if pred_w is not None and gt_w is not None:
            width_errors.append(abs(pred_w - gt_w))
        if pred_d is not None and gt_d is not None:
            depth_errors.append(abs(pred_d - gt_d))

        per_sample.append({
            "pid": pid,
            "pred": {"height": pred_h, "width": pred_w, "depth": pred_d},
            "gt": {"height": gt_h, "width": gt_w, "depth": gt_d},
        })

    return {
        "height_mae": float(np.mean(height_errors)) if height_errors else None,
        "width_mae": float(np.mean(width_errors)) if width_errors else None,
        "depth_mae": float(np.mean(depth_errors)) if depth_errors else None,
        "_per_sample": per_sample,
    }
