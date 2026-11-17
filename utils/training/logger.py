"""
Professional training logger with dual output (file + console) and CSV export.

Produces per run a human-readable ``.log`` and epoch-level ``.csv`` file.
"""
import os
import csv
import logging
from datetime import datetime
from typing import Optional, Dict


class TrainingLogger:
    """
    Dual-output logger: human-readable .log file + machine-readable .csv files.

    Usage::

        logger = TrainingLogger(log_dir="logs", run_name="swinx2s_femur")
        logger.log_epoch(epoch=1, total=100, train_loss=0.5, val_loss=0.4,
                         val_dice=0.85, lr=1e-4, patience=0, patience_max=10)
        logger.log_best(epoch=1, val_dice=0.85, val_loss=0.4)
        logger.close()
    """

    # Main CSV columns (one row per epoch)
    CSV_COLUMNS = [
        "epoch", "train_loss", "train_bce", "train_dice_loss", "grad_norm",
        "val_loss", "val_bce", "val_dice_loss", "val_dice",
        "val_iou", "val_precision", "val_recall", "val_specificity", "val_accuracy",
        "lr", "patience", "epoch_time_sec", "is_best",
    ]

    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        base = f"{run_name}_{timestamp}"

        self.log_path = os.path.join(log_dir, f"{base}.log")
        self.csv_path = os.path.join(log_dir, f"{base}.csv")

        # ── Python logger (file + console) ──
        self._logger = logging.getLogger(f"train.{run_name}.{timestamp}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        # ── Epoch CSV ──
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_COLUMNS)


    # ------------------------------------------------------------------
    # Public API — general
    # ------------------------------------------------------------------

    def info(self, msg: str):
        """Log an informational message (console + file)."""
        self._logger.info(msg)

    def debug(self, msg: str):
        """Log a debug message (file only)."""
        self._logger.debug(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def log_config(self, config):
        """Log key configuration fields."""
        self._logger.info("=" * 70)
        self._logger.info(f"  Run: {self.run_name}")
        self._logger.info("=" * 70)
        for attr in [
            "RUN_ID", "MODEL_TYPE", "DATASET_NAME", "DATA_ROOT", "SAVE_DIR",
            "IMG_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE",
            "OPTIMIZER", "WEIGHT_DECAY", "LR_SCHEDULER", "WARMUP_EPOCHS",
            "RANDOM_SEED", "NUM_WORKERS", "DEVICE", "USE_AMP", "PATIENCE", "AUGMENT_TRAIN",
            # settings that change geometry, architecture or data (added 16 Nov 2026)
            "FLIP_AP_LR", "VIEW_GEOMETRY", "DVC_NORM", "SWIN_IMPL", "SWIN_OUTPUT_MODE", "NORM_NAME",
            "MODEL_SIZE", "FEATURE_SIZE", "DEPTHS", "NUM_HEADS", "INTERMEDIATE_DEPTH", "MID_CHANNELS",
            "VOLUME_SIZE", "SPLIT_MANIFEST",
        ]:
            val = getattr(config, attr, None)
            if val is not None:
                self._logger.info(f"  {attr:<20s}: {val}")
        self._logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Public API — epoch level
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        total: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        lr: float,
        patience: int,
        patience_max: int,
        train_bce: float = 0.0,
        train_dice_loss: float = 0.0,
        grad_norm: float = 0.0,
        val_bce: float = 0.0,
        val_dice_loss: float = 0.0,
        segmentation_metrics: Optional[Dict[str, float]] = None,
        epoch_time: float = 0.0,
        is_best: bool = False,
    ):
        """Write one epoch's metrics to log and CSV."""
        line = (
            f"[Epoch {epoch}/{total}] "
            f"Train Loss: {train_loss:.4f} (BCE {train_bce:.4f} + Dice {train_dice_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} (BCE {val_bce:.4f} + Dice {val_dice_loss:.4f}) | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {lr:.2e} | "
            f"Patience: {patience}/{patience_max}"
        )
        if epoch_time > 0:
            line += f" | Time: {epoch_time:.1f}s"
        if segmentation_metrics:
            line += (
                f" | IoU: {segmentation_metrics['iou']:.4f}"
                f" | Precision: {segmentation_metrics['precision']:.4f}"
                f" | Recall: {segmentation_metrics['recall']:.4f}"
            )
        if is_best:
            line += " *BEST*"
        self._logger.info(line)

        metric_values = [
            (segmentation_metrics or {}).get(name, 0.0)
            for name in ("iou", "precision", "recall", "specificity", "accuracy")
        ]
        row = [
            epoch, train_loss, train_bce, train_dice_loss, grad_norm,
            val_loss, val_bce, val_dice_loss, val_dice,
            *metric_values,
            lr, patience, epoch_time, int(is_best),
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ------------------------------------------------------------------
    # Public API — events
    # ------------------------------------------------------------------

    def log_best(self, epoch: int, val_dice: float, val_loss: float):
        self._logger.info(
            f"  ** New best Dice: {val_dice:.4f} (epoch {epoch}, loss {val_loss:.4f})"
        )

    def log_early_stop(self, epoch: int, patience: int, best_dice: float, best_epoch: int):
        self._logger.info(
            f"  Early stopping at epoch {epoch} "
            f"(no improvement for {patience} epochs). "
            f"Best Dice: {best_dice:.4f} at epoch {best_epoch}"
        )

    def log_summary(self, best_dice: float, best_loss: float, best_epoch: int, save_dir: str):
        self._logger.info("=" * 70)
        self._logger.info("  Training Complete")
        self._logger.info("=" * 70)
        self._logger.info(f"  Best Val Dice : {best_dice:.4f}")
        self._logger.info(f"  Best Val Loss : {best_loss:.4f}")
        self._logger.info(f"  Best Epoch    : {best_epoch}")
        self._logger.info(f"  Model Dir     : {save_dir}")
        self._logger.info(f"  Log File      : {self.log_path}")
        self._logger.info(f"  CSV File      : {self.csv_path}")
        self._logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Flush and close all handlers and files."""
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        try:
            self._csv_file.close()
        except Exception:
            pass
