"""
Combined Dice + BCE loss for binary segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Weighted sum of Binary Cross-Entropy and Dice losses."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        from_logits: bool = True,
        pos_weight: torch.Tensor = None,
        smooth: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.from_logits = from_logits
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape != targets.shape:
            raise ValueError(
                f"DiceBCELoss: preds shape {tuple(preds.shape)} "
                f"!= targets shape {tuple(targets.shape)}"
            )

        targets = targets.float()

        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(
                preds.float(), targets, pos_weight=self.pos_weight, reduction="mean"
            )
            probs = torch.sigmoid(preds.float())
        else:
            bce = F.binary_cross_entropy(
                preds.float().clamp_(0, 1), targets, reduction="mean"
            )
            probs = preds.float().clamp_(0, 1)

        dims = list(range(2, preds.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice_per = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        dice_loss = 1.0 - dice_per.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss

    # ------------------------------------------------------------------
    # Component accessors  (used by trainer for separate logging)
    # ------------------------------------------------------------------

    def bce_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate only the BCE component."""
        targets = targets.float()
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(
                preds.float(), targets, pos_weight=self.pos_weight, reduction="mean"
            )
        return F.binary_cross_entropy(
            preds.float().clamp_(0, 1), targets, reduction="mean"
        )

    def dice_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate only the Dice component."""
        targets = targets.float()
        if self.from_logits:
            probs = torch.sigmoid(preds.float())
        else:
            probs = preds.float().clamp_(0, 1)

        dims = list(range(2, preds.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice_per = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice_per.mean()


def _two_class_logits(z: torch.Tensor) -> torch.Tensor:
    """Binary logit z -> two-class logits [0, z], so softmax gives [1 - p, p]."""
    z = z.float()
    return torch.cat([torch.zeros_like(z), z], dim=1)


def two_view_loss(criterion, view_logits, target):
    """Official Swin-X2S objective for a model with one output per view.

    Follows trainer.py of https://github.com/liukuan5625/Swin-X2S (MIT licence): for each view
    output i, loss_i = criterion(out_i, target) + sum_{j != i} KL(softmax(out_j) || softmax(out_i)),
    and the total is the mean over views. The official code uses 2+ softmax channels; for our
    binary sigmoid output the logit z is written as two-class logits [0, z]. KL uses
    reduction="mean" over all elements, as in the official KLDivLoss(reduction="mean").
    """
    n = len(view_logits)
    total = 0.0
    for i, zi in enumerate(view_logits):
        mutual = 0.0
        for j, zj in enumerate(view_logits):
            if i != j:
                # KL(p_j || p_i) averaged over all elements, i.e. KLDivLoss(reduction="mean") as in the
                # official code, written out so a future PyTorch change of "mean" cannot alter it
                log_q = F.log_softmax(_two_class_logits(zi), dim=1)
                log_p = F.log_softmax(_two_class_logits(zj), dim=1)
                mutual = mutual + (log_p.exp() * (log_p - log_q)).mean()
        total = total + (criterion(zi, target) + mutual) / n
    return total
