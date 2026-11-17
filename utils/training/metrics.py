"""Standard, anatomy-agnostic metrics for binary segmentation."""
import torch


@torch.no_grad()
def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    smooth: float = 0.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return per-sample thresholded segmentation metrics.

    Both predictions and targets are binarised so these are evaluation
    metrics, not differentiable training objectives.  Returning one value per
    sample lets callers compute a sample-weighted epoch average.
    """
    targets = targets.float()
    if from_logits:
        preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    dims = list(range(2, preds.ndim))
    tp = torch.sum(preds * targets, dim=dims)
    fp = torch.sum(preds * (1.0 - targets), dim=dims)
    fn = torch.sum((1.0 - preds) * targets, dim=dims)
    tn = torch.sum((1.0 - preds) * (1.0 - targets), dim=dims)

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Use a perfect score for a metric with no relevant voxels."""
        return torch.where(
            denominator > 0,
            (numerator + smooth) / (denominator + smooth + eps),
            torch.ones_like(denominator),
        )

    return {
        "dice": ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


@torch.no_grad()
def compute_dice(*args, **kwargs) -> float:
    """Backward-compatible batch-mean Dice helper."""
    return compute_segmentation_metrics(*args, **kwargs)["dice"].mean().item()
