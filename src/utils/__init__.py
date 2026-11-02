"""
Utilities for training and evaluation.

Sub-packages:

  - src.datasets     — dataset loading (re-exported here)
  - utils.training   — training loop, losses, metrics, checkpoints, logging
"""

# Re-export core functions/classes for easy imports
from src.datasets import BoneDataset, collate_bone
from src.utils.training.losses import DiceBCELoss
from src.utils.training.metrics import compute_dice
from src.utils.training.trainer import train_one_epoch, validate_one_epoch
from src.utils.training.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)
from src.utils.training.logger import TrainingLogger

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
