"""
Clinical metrics for hip / pelvis bone reconstruction.

Measurements:
  - **Pelvic width** (mm): maximum bone span along the principal axis
  - **Acetabular diameter** (mm): sphere-fit diameter of the acetabular cup
  - **Acetabular depth** (mm): physical depth of the acetabular cup measured
    as the smallest PCA span of the acetabular region

All measurements use actual voxel spacing from the NIfTI header.

Changes from the original implementation:
  - Removed unused imports (ConvexHull, cdist, binary_erosion).
  - ``acetabular_depth`` now computes a physical depth in mm using the
    third PCA component span of the acetabular region, instead of the
    angle between lateral / medial PCA axes (which was mislabelled as
    a depth but actually computed an angle in degrees).
  - Per-sample voxel spacing is used instead of hardcoded (1, 1, 1).
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional


def compute_hip_metrics(
    mask: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute clinical metrics for a single hip / pelvis mask.

    Args:
        mask: Binary 3D numpy array.
        spacing: Voxel spacing (z, y, x) in mm.

    Returns:
        (pelvic_width, acetabular_diameter, acetabular_depth)
        — any value may be *None* if the mask is too small.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < 2000:
        return None, None, None

    coords_mm = coords * np.asarray(spacing, dtype=np.float64)

    # Global PCA for main axis
    pca_global = PCA(n_components=3)
    pca_global.fit(coords_mm)
    main_axis = pca_global.components_[0]

    projections = coords_mm @ main_axis
    min_proj, max_proj = projections.min(), projections.max()
    pelvic_width = float(max_proj - min_proj)

    # -----------------------------------------------------------------
    # 1. Acetabular diameter — sphere fit on the lateral-most 25 mm
    # -----------------------------------------------------------------
    acetab_points = coords_mm[projections > (max_proj - 25)]

    if len(acetab_points) < 1000:
        diameter = None
    else:
        A = np.hstack((2 * acetab_points, np.ones((len(acetab_points), 1))))
        b = np.sum(acetab_points ** 2, axis=1).reshape(-1, 1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        center = x[:3].flatten()
        c = float(x[3])
        r2 = c + np.sum(center ** 2)
        diameter = float(2 * np.sqrt(r2)) if r2 > 0 else None

    # -----------------------------------------------------------------
    # 2. Acetabular depth — physical cup depth in mm
    #
    #    Take the lateral region (acetabular cup), run PCA on it, and
    #    measure the span along the smallest principal component.  For a
    #    concave cup the smallest extent corresponds to the depth of the
    #    concavity.
    # -----------------------------------------------------------------
    pelvis_len = max_proj - min_proj
    lateral_region = coords_mm[projections > (min_proj + 0.7 * pelvis_len)]

    if len(lateral_region) < 500:
        depth = None
    else:
        pca_local = PCA(n_components=3)
        pca_local.fit(lateral_region)
        projected = pca_local.transform(lateral_region)
        spans = projected.max(axis=0) - projected.min(axis=0)
        # The smallest PCA span of the cup ≈ depth
        depth = float(spans.min())

    return pelvic_width, diameter, depth


def evaluate_batch_clinical_metrics(
    pred_volumes: torch.Tensor,
    gt_volumes: torch.Tensor,
    voxel_spacings: Optional[List[np.ndarray]] = None,
    pids: Optional[List[str]] = None,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Evaluate hip clinical metrics for a batch.

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

    width_errors, diameter_errors, depth_errors = [], [], []
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

        pred_w, pred_d, pred_dep = compute_hip_metrics(pred_mask, sp)
        gt_w, gt_d, gt_dep = compute_hip_metrics(gt_mask, sp)

        if pred_w is not None and gt_w is not None:
            width_errors.append(abs(pred_w - gt_w))
        if pred_d is not None and gt_d is not None:
            diameter_errors.append(abs(pred_d - gt_d))
        if pred_dep is not None and gt_dep is not None:
            depth_errors.append(abs(pred_dep - gt_dep))

        per_sample.append({
            "pid": pid,
            "pred": {"pelvic_width": pred_w, "acetabular_diameter": pred_d, "acetabular_depth": pred_dep},
            "gt": {"pelvic_width": gt_w, "acetabular_diameter": gt_d, "acetabular_depth": gt_dep},
        })

    return {
        "pelvic_width_mae": float(np.mean(width_errors)) if width_errors else None,
        "acetabular_diameter_mae": float(np.mean(diameter_errors)) if diameter_errors else None,
        "acetabular_depth_mae": float(np.mean(depth_errors)) if depth_errors else None,
        "_per_sample": per_sample,
    }
