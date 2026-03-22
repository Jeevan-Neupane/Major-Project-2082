"""Clinical metrics for anatomical measurements.

Each module provides:
  - compute_*_metrics(mask, spacing)  → per-sample measurements
  - evaluate_batch_clinical_metrics(pred, gt, spacing)  → batch MAEs + per-sample details
"""

from utils.clinical.femur_metrics import (
    compute_femur_metrics,
    evaluate_batch_clinical_metrics as evaluate_femur_batch,
)
from utils.clinical.hip_metrics import (
    compute_hip_metrics,
    evaluate_batch_clinical_metrics as evaluate_hip_batch,
)
from utils.clinical.verse_metrics import (
    compute_verse_metrics,
    evaluate_batch_clinical_metrics as evaluate_verse_batch,
)

__all__ = [
    "compute_femur_metrics",
    "evaluate_femur_batch",
    "compute_hip_metrics",
    "evaluate_hip_batch",
    "compute_verse_metrics",
    "evaluate_verse_batch",
]
