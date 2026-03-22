"""
Professional training logger with dual output (file + console) and CSV export.

Produces per run:
  - ``{run_name}_{timestamp}.log``  — human-readable log file
  - ``{run_name}_{timestamp}.csv``  — epoch-level metrics for plotting
  - ``{run_name}_{timestamp}_clinical.csv`` — per-sample clinical metrics
"""
import os
import csv
import logging
from datetime import datetime
from typing import Optional, List, Dict


class TrainingLogger:
    """
    Dual-output logger: human-readable .log file + machine-readable .csv files.

    Usage::

        logger = TrainingLogger(log_dir="logs", run_name="swinx2s_femur")
        logger.log_epoch(epoch=1, total=100, train_loss=0.5, val_loss=0.4,
                         val_dice=0.85, lr=1e-4, patience=0, patience_max=10)
        logger.log_clinical({"diameter_mae": 2.3, "nsa_mae": 3.1})
        logger.log_sample_clinical(epoch=1, pid="p_0001",
                                   pred={"diameter": 45.2}, gt={"diameter": 44.1})
        logger.log_best(epoch=1, val_dice=0.85, val_loss=0.4)
        logger.close()
    """

    # Main CSV columns (one row per epoch)
    CSV_COLUMNS = [
        "epoch", "train_loss", "train_bce", "train_dice_loss", "grad_norm",
        "val_loss", "val_bce", "val_dice_loss", "val_dice",
        "lr", "patience", "epoch_time_sec", "is_best",
    ]

    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        base = f"{run_name}_{timestamp}"

        self.log_path = os.path.join(log_dir, f"{base}.log")
        self.csv_path = os.path.join(log_dir, f"{base}.csv")
        self.clinical_csv_path = os.path.join(log_dir, f"{base}_clinical.csv")

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

        # ── Per-sample clinical CSV (created lazily on first write) ──
        self._clinical_csv_file = None
        self._clinical_csv_writer = None
        self._clinical_columns_written = False

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
            "MODEL_TYPE", "DATASET_NAME", "DATA_ROOT", "SAVE_DIR",
            "IMG_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE",
            "DEVICE", "USE_AMP", "PATIENCE",
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
        if is_best:
            line += " *BEST*"
        self._logger.info(line)

        row = [
            epoch, train_loss, train_bce, train_dice_loss, grad_norm,
            val_loss, val_bce, val_dice_loss, val_dice,
            lr, patience, epoch_time, int(is_best),
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ------------------------------------------------------------------
    # Public API — clinical metrics (aggregate)
    # ------------------------------------------------------------------

    def log_clinical(self, metrics: dict):
        """Log batch-averaged clinical metrics."""
        if not metrics:
            return
        task = metrics.get("_task", "?")
        parts = [
            f"{k.replace('_mae', '').replace('_', ' ').title()} MAE: {v:.2f}"
            for k, v in metrics.items()
            if k != "_task" and v is not None
        ]
        if parts:
            self._logger.info(f"  Clinical ({task}): " + " | ".join(parts))

    # ------------------------------------------------------------------
    # Public API — per-sample clinical metrics
    # ------------------------------------------------------------------

    def log_sample_clinical(
        self,
        epoch: int,
        pid: str,
        pred_metrics: dict,
        gt_metrics: dict,
        metric_names: List[str],
    ):
        """
        Log per-sample clinical measurements to the clinical CSV.

        First call lazily creates the file and writes the header.

        Args:
            epoch: Current epoch number.
            pid: Patient ID.
            pred_metrics: dict mapping metric_name → predicted value.
            gt_metrics: dict mapping metric_name → ground-truth value.
            metric_names: ordered list of metric names (used for header).
        """
        # Lazy-create clinical CSV
        if self._clinical_csv_file is None:
            self._clinical_csv_file = open(
                self.clinical_csv_path, "w", newline="", encoding="utf-8"
            )
            self._clinical_csv_writer = csv.writer(self._clinical_csv_file)
            # Build header: epoch, pid, pred_<metric>, gt_<metric>, error_<metric>, ...
            header = ["epoch", "pid"]
            for name in metric_names:
                header += [f"pred_{name}", f"gt_{name}", f"error_{name}"]
            self._clinical_csv_writer.writerow(header)
            self._clinical_columns_written = True

        row = [epoch, pid]
        for name in metric_names:
            pv = pred_metrics.get(name)
            gv = gt_metrics.get(name)
            err = abs(pv - gv) if (pv is not None and gv is not None) else None
            row += [
                f"{pv:.3f}" if pv is not None else "",
                f"{gv:.3f}" if gv is not None else "",
                f"{err:.3f}" if err is not None else "",
            ]
        self._clinical_csv_writer.writerow(row)
        self._clinical_csv_file.flush()

    def log_sample_clinical_console(
        self,
        epoch: int,
        pid: str,
        pred_metrics: dict,
        gt_metrics: dict,
        max_samples: int = 3,
        _counter: dict = {},
    ):
        """
        Print per-sample clinical metrics to the console (first N samples
        per epoch only, to avoid flooding the terminal).
        """
        key = f"epoch_{epoch}"
        _counter.setdefault(key, 0)
        if _counter[key] >= max_samples:
            return
        _counter[key] += 1

        parts = []
        for k in sorted(pred_metrics.keys()):
            pv = pred_metrics.get(k)
            gv = gt_metrics.get(k)
            if pv is not None and gv is not None:
                parts.append(f"{k}: pred={pv:.2f} gt={gv:.2f} err={abs(pv-gv):.2f}")
        if parts:
            self._logger.info(f"    Sample {pid}: " + " | ".join(parts))

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
        self._logger.info(f"  Clinical CSV  : {self.clinical_csv_path}")
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
        if self._clinical_csv_file is not None:
            try:
                self._clinical_csv_file.close()
            except Exception:
                pass
