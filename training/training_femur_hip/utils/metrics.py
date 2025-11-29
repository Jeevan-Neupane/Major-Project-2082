import torch

@torch.no_grad()
def compute_dice(preds: torch.Tensor,
                 targets: torch.Tensor,
                 from_logits: bool = True,
                 threshold: float = 0.5,
                 smooth: float = 1.0,
                 eps: float = 1e-6) -> float:
    targets = targets.float()
    if from_logits:
        preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    dims = list(range(2, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    denom = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    dice_per = (2.0 * intersection + smooth) / (denom + smooth + eps)
    return dice_per.mean().item() 