"""Training and validation loops with standard segmentation metrics."""
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.training.losses import two_view_loss


def _objective(model, criterion, logits, target):
    """Loss on the prediction, or the official two-view loss when the model keeps one output per view."""
    view_logits = getattr(model, "view_logits", None)
    if view_logits is not None:
        return two_view_loss(criterion, view_logits, target)
    return criterion(logits, target)


# ------------------------------------------------------------------
# Target alignment
# ------------------------------------------------------------------

def match_target_to_output(target, out):
    """Resize / squeeze *target* so its spatial dims match *out*."""
    target = target.float()

    if out.ndim == 5:  # 3D  [B, C, D, H, W]
        B, C, D, H, W = out.shape
        if target.ndim == 4:
            target = target.unsqueeze(2)
        if target.shape[-3:] != (D, H, W):
            target = F.interpolate(target, size=(D, H, W), mode="trilinear", align_corners=False)
    elif out.ndim == 4:  # 2D  [B, C, H, W]
        B, C, H, W = out.shape
        if target.ndim == 5:
            target = target.max(dim=2, keepdim=False).values
        if target.shape[-2:] != (H, W):
            target = F.interpolate(target, size=(H, W), mode="bilinear", align_corners=False)

    return target.clamp_(0, 1)


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, config, epoch, total_epochs, use_wandb=False):
    """
    Train for one epoch.

    Returns:
        (avg_loss, avg_bce, avg_dice_loss, avg_grad_norm, epoch_time)
    """
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    total_grad_norm = 0.0

    epoch_start = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")

    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xb = batch["xb"].to(device, non_blocking=True)
        target_raw = batch[config.TARGET_KEY].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            logits = model(xa, xb)
            target = match_target_to_output(target_raw, logits)
            loss = _objective(model, criterion, logits, target)

            bce_loss = criterion.bce_loss(logits, target) if hasattr(criterion, "bce_loss") else loss
            dice_loss = criterion.dice_loss(logits, target) if hasattr(criterion, "dice_loss") else torch.tensor(0.0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_bce += bce_loss.item()
        running_dice += dice_loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"})

    epoch_time = time.time() - epoch_start
    n = len(train_loader)
    return running_loss / n, running_bce / n, running_dice / n, total_grad_norm / n, epoch_time


# ------------------------------------------------------------------
# Validate
# ------------------------------------------------------------------

def validate_one_epoch(model, val_loader, criterion, device, config,
                       epoch, total_epochs):
    """
    Validate for one epoch.

    Args:
        model, val_loader, criterion, device, config: standard training args.
        epoch: 0-indexed epoch number.
        total_epochs: total number of epochs.
    Returns:
        (avg_loss, avg_dice, avg_bce, avg_dice_loss, segmentation_metrics)
    """
    from utils.training.metrics import compute_segmentation_metrics

    model.eval()
    val_loss = 0.0
    val_bce = 0.0
    val_dice_loss = 0.0
    metric_totals = {name: 0.0 for name in (
        "dice", "iou", "precision", "recall", "specificity", "accuracy"
    )}
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        for batch in pbar:
            xa = batch["xa"].to(device, non_blocking=True)
            xb = batch["xb"].to(device, non_blocking=True)
            target_raw = batch[config.TARGET_KEY].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                logits = model(xa, xb)
                target = match_target_to_output(target_raw, logits)
                loss = _objective(model, criterion, logits, target)
                bce_loss = criterion.bce_loss(logits, target) if hasattr(criterion, "bce_loss") else loss
                dice_loss_val = criterion.dice_loss(logits, target) if hasattr(criterion, "dice_loss") else torch.tensor(0.0)

            val_loss += loss.item()
            val_bce += bce_loss.item()
            val_dice_loss += dice_loss_val.item()
            batch_metrics = compute_segmentation_metrics(logits.float(), target.float())
            batch_size = target.shape[0]
            sample_count += batch_size
            for name, values in batch_metrics.items():
                metric_totals[name] += values.sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_metrics['dice'].mean().item():.4f}",
            })

    n = len(val_loader)
    avg_loss = val_loss / n
    segmentation_metrics = {
        name: value / sample_count for name, value in metric_totals.items()
    }
    avg_dice = segmentation_metrics["dice"]
    avg_bce = val_bce / n
    avg_dice_loss = val_dice_loss / n

    return avg_loss, avg_dice, avg_bce, avg_dice_loss, segmentation_metrics
