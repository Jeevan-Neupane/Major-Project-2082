"""
Checkpoint management for training runs.

Provides:
  - CheckpointManager  – save/replace best + periodic checkpoints
  - save_checkpoint()   – legacy functional helper
  - load_checkpoint()   – load any checkpoint file
"""
import os
import glob
import torch


class CheckpointManager:
    """
    Manages checkpoint saving with organized folder structure.

    Layout::

        {save_dir}/{model_type}[/{size}]/{dataset}/
            best_epoch{N}_dice{D:.4f}_loss{L:.4f}.pth
            best_model_weights.pth
            periodic/
                periodic_epoch{N}_dice{D:.4f}_loss{L:.4f}.pth

    Only ONE best and ONE periodic file are kept at any time.
    """

    def __init__(self, save_dir: str, dataset: str, model_type: str,
                 model_size: str = "default"):
        if model_type == "swinx2s":
            self.best_dir = os.path.join(save_dir, model_type, model_size, dataset)
            self.periodic_dir = os.path.join(save_dir, model_type, model_size, dataset, "periodic")
        else:
            self.best_dir = os.path.join(save_dir, model_type, dataset)
            self.periodic_dir = os.path.join(save_dir, model_type, dataset, "periodic")

        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.periodic_dir, exist_ok=True)

        self._last_best: str | None = None
        self._last_periodic: str | None = None

        # Recover pre-existing files so we can clean them on save
        self._last_best = self._find_existing("best_epoch*.pth", self.best_dir)
        self._last_periodic = self._find_existing("periodic_epoch*.pth", self.periodic_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_best(self, model, optimizer, scheduler, epoch: int,
                  val_loss: float, val_dice: float) -> str:
        filename = f"best_epoch{epoch:03d}_dice{val_dice:.4f}_loss{val_loss:.4f}.pth"
        save_path = os.path.join(self.best_dir, filename)

        self._write_checkpoint(model, optimizer, scheduler, epoch,
                               val_loss, val_dice, save_path, is_best=True)

        if self._last_best and os.path.isfile(self._last_best) and self._last_best != save_path:
            os.remove(self._last_best)
        self._last_best = save_path
        return save_path

    def save_periodic(self, model, optimizer, scheduler, epoch: int,
                      val_loss: float, val_dice: float) -> str:
        filename = f"periodic_epoch{epoch:03d}_dice{val_dice:.4f}_loss{val_loss:.4f}.pth"
        save_path = os.path.join(self.periodic_dir, filename)

        self._write_checkpoint(model, optimizer, scheduler, epoch,
                               val_loss, val_dice, save_path, is_best=False)

        if self._last_periodic and os.path.isfile(self._last_periodic) and self._last_periodic != save_path:
            os.remove(self._last_periodic)
        self._last_periodic = save_path
        return save_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_checkpoint(self, model, optimizer, scheduler, epoch: int,
                          val_loss: float, val_dice: float, save_path: str,
                          is_best: bool):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
        torch.save(checkpoint, save_path)

        if is_best:
            weights_path = os.path.join(self.best_dir, "best_model_weights.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"  Checkpoint (best)     -> {os.path.relpath(save_path)}")
        else:
            print(f"  Checkpoint (periodic) -> {os.path.relpath(save_path)}")

    @staticmethod
    def _find_existing(pattern: str, directory: str):
        matches = glob.glob(os.path.join(directory, pattern))
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Legacy functional API
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, is_best=False):
    """Legacy helper — prefer CheckpointManager for new code."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
        "loss": loss,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model_weights.pth")
        torch.save(model.state_dict(), best_path)
        print(f"  Best model weights -> {best_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device="cpu"):
    """Load a checkpoint saved by CheckpointManager or save_checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch
