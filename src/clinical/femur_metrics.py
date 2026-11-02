"""
Clinical metrics for femur bone reconstruction.

Measurements:
  - **Femoral head diameter** (mm): sphere-fit on the proximal region
  - **Neck-shaft angle** (degrees): angle between neck and shaft PCA axes
  - **Femur length** (mm): extent along the principal axis

All measurements use actual voxel spacing from the NIfTI header.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional


def compute_femur_metrics(
    mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute clinical metrics for a single femur mask.

    Args:
        mask: Binary 3D numpy array.
        spacing: Voxel spacing (z, y, x) in mm.

    Returns:
        (diameter, nsa, length) — any value may be *None* if the
        mask is too small or a sub-computation fails.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < 2000:
        return None, None, None

    coords_mm = coords * np.asarray(spacing, dtype=np.float64)

    # Global PCA for principal axis
    pca_global = PCA(n_components=3)
    pca_global.fit(coords_mm)
    main_axis = pca_global.components_[0]

    projections = coords_mm @ main_axis
    min_proj, max_proj = projections.min(), projections.max()
    length = float(max_proj - min_proj)

    # 1. Femoral head diameter — sphere fit on proximal 25 mm
    head_points = coords_mm[projections > (max_proj - 25)]
    if len(head_points) < 1000:
        diameter = None
    else:
        A = np.hstack((2 * head_points, np.ones((len(head_points), 1))))
        b = np.sum(head_points ** 2, axis=1).reshape(-1, 1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        center = x[:3].flatten()
        c = float(x[3])
        r2 = c + np.sum(center ** 2)
        diameter = float(2 * np.sqrt(r2)) if r2 > 0 else None

    # 2. Neck-shaft angle
    femur_len = max_proj - min_proj
    shaft_region = coords_mm[projections < (min_proj + 0.4 * femur_len)]
    neck_region = coords_mm[projections > (min_proj + 0.6 * femur_len)]

    if len(shaft_region) < 500 or len(neck_region) < 500:
        nsa = None
    else:
        shaft_axis = PCA(n_components=3).fit(shaft_region).components_[0]
        neck_axis = PCA(n_components=3).fit(neck_region).components_[0]
        dot = np.clip(abs(np.dot(shaft_axis, neck_axis)), -1.0, 1.0)
        nsa = float(180 - np.degrees(np.arccos(dot)))

    return diameter, nsa, length


def evaluate_batch_clinical_metrics(
    pred_volumes: torch.Tensor,
    gt_volumes: torch.Tensor,
    voxel_spacings: Optional[List[np.ndarray]] = None,
    pids: Optional[List[str]] = None,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Evaluate femur clinical metrics for a batch.

    Args:
        pred_volumes: [B, 1, D, H, W] tensor (probabilities 0–1).
        gt_volumes:   [B, 1, D, H, W] tensor.
        voxel_spacings: list of per-sample spacing arrays (preferred).
        pids: list of patient IDs.
        spacing: fallback spacing when *voxel_spacings* is not given.

    Returns:
        dict with ``diameter_mae``, ``nsa_mae``, ``length_mae`` (floats or None)
        plus ``_per_sample`` list of dicts for detailed logging.
    """
    pred_np = pred_volumes.detach().cpu().numpy()
    gt_np = gt_volumes.detach().cpu().numpy()
    B = pred_np.shape[0]

    diameter_errors, nsa_errors, length_errors = [], [], []
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

        pred_d, pred_nsa, pred_l = compute_femur_metrics(pred_mask, sp)
        gt_d, gt_nsa, gt_l = compute_femur_metrics(gt_mask, sp)

        if pred_d is not None and gt_d is not None:
            diameter_errors.append(abs(pred_d - gt_d))
        if pred_nsa is not None and gt_nsa is not None:
            nsa_errors.append(abs(pred_nsa - gt_nsa))
        if pred_l is not None and gt_l is not None:
            length_errors.append(abs(pred_l - gt_l))

        per_sample.append({
            "pid": pid,
            "pred": {"diameter": pred_d, "nsa": pred_nsa, "length": pred_l},
            "gt": {"diameter": gt_d, "nsa": gt_nsa, "length": gt_l},
        })

    return {
        "diameter_mae": float(np.mean(diameter_errors)) if diameter_errors else None,
        "nsa_mae": float(np.mean(nsa_errors)) if nsa_errors else None,
        "length_mae": float(np.mean(length_errors)) if length_errors else None,
        "_per_sample": per_sample,
    }
