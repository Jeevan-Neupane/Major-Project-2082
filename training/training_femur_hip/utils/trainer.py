import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


def match_target_to_output(target, out):
    """
    Align target to output shape.
    
    Args:
        target: Ground truth tensor
        out: Model output tensor
        
    Returns:
        Aligned target tensor in [0,1]
    """
    target = target.float()
    
    if out.ndim == 5:  # 3D output [B,C,D,H,W]
        B, C, D, H, W = out.shape
        if target.ndim == 4:
            target = target.unsqueeze(2)
        if target.shape[-3:] != (D, H, W):
            target = F.interpolate(target, size=(D, H, W), mode="trilinear", align_corners=False)
    elif out.ndim == 4:  # 2D output [B,C,H,W]
        B, C, H, W = out.shape
        if target.ndim == 5:
            target = target.max(dim=2, keepdim=False).values
        if target.shape[-2:] != (H, W):
            target = F.interpolate(target, size=(H, W), mode="bilinear", align_corners=False)
    
    return target.clamp_(0, 1)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, 
                    config, epoch, total_epochs, use_wandb=False):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: torch.device
        config: Configuration object
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        use_wandb: Whether to log to W&B
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xb = batch["xb"].to(device, non_blocking=True)
        target_raw = batch[config.TARGET_KEY].to(device, non_blocking=True)
        
        # Use half precision only on GPU with AMP
        if device.type == 'cuda' and config.USE_AMP:
            xa = xa.half()
            xb = xb.half()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            logits = model(xa, xb)
            target = match_target_to_output(target_raw, logits)
            loss = criterion(logits, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if use_wandb:
            wandb.log({"train/batch_loss": loss.item()})
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate_one_epoch(model, val_loader, criterion, device, config, 
                       epoch, total_epochs, use_wandb=False):
    """
    Validate for one epoch.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: torch.device
        config: Configuration object
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        use_wandb: Whether to log to W&B
        
    Returns:
        Tuple of (average validation loss, average dice score)
    """
    from utils.metrics import compute_dice
    
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        for batch in pbar:
            xa = batch["xa"].to(device, non_blocking=True)
            xb = batch["xb"].to(device, non_blocking=True)
            target_raw = batch[config.TARGET_KEY].to(device, non_blocking=True)
            
            if device.type == 'cuda' and config.USE_AMP:
                xa = xa.half()
                xb = xb.half()
            
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                logits = model(xa, xb)
                target = match_target_to_output(target_raw, logits)
                loss = criterion(logits, target)
            
            val_loss += loss.item()
            val_dice += compute_dice(logits.float(), target.float())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if use_wandb:
                wandb.log({"val/batch_loss": loss.item()})
    
    avg_loss = val_loss / len(val_loader)
    avg_dice = val_dice / len(val_loader)
    
    return avg_loss, avg_dice