"""
Training and validation loops with clinical metric evaluation.

Key improvements over the original:
  - Uses actual voxel spacing from the batch (not hardcoded 1.0)
  - Collects per-sample clinical metric predictions for detailed logging
  - Returns per-sample details so the caller can log them
"""
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


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
            loss = criterion(logits, target)

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

# Metric names keyed by task
_METRIC_NAMES = {
    "femur": ["diameter", "nsa", "length"],
    "hip": ["pelvic_width", "acetabular_diameter", "acetabular_depth"],
    "verse": ["height", "width", "depth"],
}


def _get_clinical_eval_fn(clinical_task):
    """Return (eval_fn, metric_names) or (None, [])."""
    if clinical_task == "femur":
        try:
            from src.clinical.femur_metrics import evaluate_batch_clinical_metrics
            return evaluate_batch_clinical_metrics, _METRIC_NAMES["femur"]
        except ImportError:
            pass
    elif clinical_task == "hip":
        try:
            from src.clinical.hip_metrics import evaluate_batch_clinical_metrics
            return evaluate_batch_clinical_metrics, _METRIC_NAMES["hip"]
        except ImportError:
            pass
    elif clinical_task == "verse":
        try:
            from src.clinical.verse_metrics import evaluate_batch_clinical_metrics
            return evaluate_batch_clinical_metrics, _METRIC_NAMES["verse"]
        except ImportError:
            pass
    return None, []


def validate_one_epoch(model, val_loader, criterion, device, config,
                       epoch, total_epochs, use_wandb=False,
                       clinical_task=None, logger=None):
    """
    Validate for one epoch.

    Args:
        model, val_loader, criterion, device, config: standard training args.
        epoch: 0-indexed epoch number.
        total_epochs: total number of epochs.
        use_wandb: whether W&B is active.
        clinical_task: ``'femur'``, ``'hip'``, ``'verse'``, or None.
        logger: optional :class:`TrainingLogger` for per-sample clinical output.

    Returns:
        (avg_loss, avg_dice, avg_bce, avg_dice_loss, clinical_metrics_dict)
    """
    from src.utils.training.metrics import compute_dice

    eval_fn, metric_names = _get_clinical_eval_fn(clinical_task)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_bce = 0.0
    val_dice_loss = 0.0

    # Accumulators (keyed by metric name)
    clinical_accum: dict[str, list] = {}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        for batch in pbar:
            xa = batch["xa"].to(device, non_blocking=True)
            xb = batch["xb"].to(device, non_blocking=True)
            target_raw = batch[config.TARGET_KEY].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                logits = model(xa, xb)
                target = match_target_to_output(target_raw, logits)
                loss = criterion(logits, target)
                bce_loss = criterion.bce_loss(logits, target) if hasattr(criterion, "bce_loss") else loss
                dice_loss_val = criterion.dice_loss(logits, target) if hasattr(criterion, "dice_loss") else torch.tensor(0.0)

            val_loss += loss.item()
            val_bce += bce_loss.item()
            val_dice_loss += dice_loss_val.item()
            val_dice += compute_dice(logits.float(), target.float())

            # -- Clinical metrics for this batch --
            if eval_fn is not None:
                pids = batch.get("pid", [])
                voxel_spacings = batch.get("voxel_spacing", [])

                batch_result = eval_fn(
                    torch.sigmoid(logits.float()),
                    target.float(),
                    voxel_spacings=voxel_spacings,
                    pids=pids,
                )

                # batch_result is a dict with:
                #   *_mae  → float (batch average)
                #   _per_sample → list[dict] with per-sample details
                per_sample = batch_result.pop("_per_sample", [])

                for k, v in batch_result.items():
                    if v is not None:
                        clinical_accum.setdefault(k, []).append(v)

                # Log per-sample details (CSV + limited console output)
                if logger is not None and per_sample:
                    for sample_info in per_sample:
                        logger.log_sample_clinical(
                            epoch=epoch + 1,
                            pid=sample_info["pid"],
                            pred_metrics=sample_info["pred"],
                            gt_metrics=sample_info["gt"],
                            metric_names=metric_names,
                        )
                        logger.log_sample_clinical_console(
                            epoch=epoch + 1,
                            pid=sample_info["pid"],
                            pred_metrics=sample_info["pred"],
                            gt_metrics=sample_info["gt"],
                            max_samples=3,
                        )

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{compute_dice(logits.float(), target.float()):.4f}",
            })

    n = len(val_loader)
    avg_loss = val_loss / n
    avg_dice = val_dice / n
    avg_bce = val_bce / n
    avg_dice_loss = val_dice_loss / n

    clinical_metrics = None
    if clinical_accum:
        clinical_metrics = {k: float(np.mean(v)) for k, v in clinical_accum.items()}
        clinical_metrics["_task"] = clinical_task

    return avg_loss, avg_dice, avg_bce, avg_dice_loss, clinical_metrics
