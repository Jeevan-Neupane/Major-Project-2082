import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"✅ Saved best model to {best_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Neural network model
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load model to
        
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    
    return epoch