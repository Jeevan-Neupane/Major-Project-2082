"""
Utilities for training and evaluation.

Sub-packages:

  - utils.data      — dataset loading
  - utils.training  — training loop, losses, metrics, checkpoints, logging
  - utils.clinical  — anatomy-specific clinical metric computation
"""

# Re-export core functions/classes for easy imports
from utils.data.dataset import BoneDataset, collate_bone
from utils.training.losses import DiceBCELoss
from utils.training.metrics import compute_dice
from utils.training.trainer import train_one_epoch, validate_one_epoch
from utils.training.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)
from utils.training.logger import TrainingLogger

__all__ = [
    "BoneDataset",
    "collate_bone",
    "DiceBCELoss",
    "compute_dice",
    "train_one_epoch",
    "validate_one_epoch",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "TrainingLogger",
]
