"""
X-Ray to 3D Bone Reconstruction - Training Script

Usage:
  python -m src.train --config swinx2s_femur                          # single run
  python -m src.train --config swinx2s_femur --model_size small       # with model size
  python -m src.train --config swinx2s_femur --no_wandb               # without W&B
  python -m src.train --list_configs                                  # list configs
  python -m src.train --batch                                         # batch mode
"""
import os
import sys

# Ensure the repository root is on sys.path so `python src/train.py` works
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.configs import parse_args, apply_args_to_config, get_config
from src.datasets import BoneDataset, collate_bone
from src.utils.training.losses import DiceBCELoss
from src.utils.training.trainer import train_one_epoch, validate_one_epoch
from src.utils.training.checkpoint import CheckpointManager
from src.utils.training.logger import TrainingLogger
from src.models.swinx2s import SwinX2S
from src.models.transvert import TransVert

# Optional batch-mode support
try:
    from src.training_jobs import TRAINING_JOBS
    BATCH_MODE_AVAILABLE = True
except ImportError:
    BATCH_MODE_AVAILABLE = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_clinical_wandb_dict(clinical_metrics: dict) -> dict:
    """Convert a clinical_metrics dict to prefixed wandb keys, skipping '_task'."""
    return {
        f"clinical/{k}": v
        for k, v in clinical_metrics.items()
        if k != "_task" and v is not None
    }


def create_model(config, device):
    """Instantiate model on *device* based on config."""
    if config.MODEL_TYPE == "swinx2s":
        model = SwinX2S(
            img_size=config.IMG_SIZE,
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            feature_size=config.FEATURE_SIZE,
            depths=config.DEPTHS,
            num_heads=config.NUM_HEADS,
            window_size=config.WINDOW_SIZE,
            norm_name=config.NORM_NAME,
            fusion_depths=config.FUSION_DEPTHS,
            spatial_dims=config.SPATIAL_DIMS,
        ).to(device)
    elif config.MODEL_TYPE == "transvert":
        model = TransVert(
            in_channels=config.IN_CHANNELS,
            mid_channels=config.MID_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            intermediate_depth=config.INTERMEDIATE_DEPTH,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    return model


# ------------------------------------------------------------------
# Core training loop
# ------------------------------------------------------------------

def train_single_config(config_name, epochs=None):
    """Train a single configuration. Returns True on success."""

    # -- Config --
    config_class = get_config(config_name)
    if epochs is not None:
        config_class.NUM_EPOCHS = epochs
    config = config_class

    device = torch.device(config.DEVICE)
    use_amp = config.USE_AMP

    # -- Directories --
    config.create_dirs()
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # -- Logger (file + console + CSV) --
    log_dir = getattr(config, "LOG_DIR", os.path.join(config.SAVE_DIR, "logs"))
    os.makedirs(log_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=log_dir, run_name=config_name)
    logger.log_config(config)
    config.print_config()

    # -- Validate data path --
    if not os.path.exists(config.DATA_ROOT):
        logger.warning(f"Data directory not found: {config.DATA_ROOT}")
        logger.close()
        return False

    # -- Datasets --
    try:
        if config.MODEL_TYPE == "swinx2s":
            img_size_2d = config.IMG_SIZE[1:]
            img_size_3d = tuple(config.IMG_SIZE)
        else:
            img_size_2d = tuple(config.IMG_SIZE)
            img_size_3d = (
                config.INTERMEDIATE_DEPTH * 4,
                config.IMG_SIZE[0],
                config.IMG_SIZE[1],
            )

        train_set = BoneDataset(
            root_dir=os.path.join(config.DATA_ROOT, "train"),
            img_size_2d=img_size_2d,
            img_size_3d=img_size_3d,
            return_volume=True,
        )
        val_set = BoneDataset(
            root_dir=os.path.join(config.DATA_ROOT, "val"),
            img_size_2d=img_size_2d,
            img_size_3d=img_size_3d,
            return_volume=True,
        )

        # Merge test split into training for more data
        test_path = os.path.join(config.DATA_ROOT, "test")
        if os.path.exists(test_path):
            test_set = BoneDataset(
                root_dir=test_path,
                img_size_2d=img_size_2d,
                img_size_3d=img_size_3d,
                return_volume=True,
            )
            from torch.utils.data import ConcatDataset
            train_set = ConcatDataset([train_set, test_set])
            logger.info("Merged test split into training data")

    except Exception as e:
        logger.warning(f"Error loading dataset: {e}")
        logger.close()
        return False

    if len(train_set) == 0 or len(val_set) == 0:
        logger.warning("Train or validation set is empty!")
        logger.close()
        return False

    logger.info(f"Dataset -- Train: {len(train_set)} | Val: {len(val_set)}")

    # -- Data loaders --
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_bone,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_bone,
    )

    # -- Model --
    model = create_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.MODEL_TYPE} -- {num_params:,} params ({num_params / 1e6:.2f}M)")

    # -- Loss / Optimiser / Scheduler --
    criterion = DiceBCELoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # -- Weights & Biases --
    wandb_enabled = False
    if config.USE_WANDB:
        try:
            model_size = getattr(config, "MODEL_SIZE", "small").upper()
            lr_str = f"{config.LEARNING_RATE:.0e}".replace("e-0", "e-")
            if config.MODEL_TYPE == "swinx2s":
                run_name = (
                    f"{config.DATASET_NAME}/SwinX2S-{model_size}/"
                    f"ep{config.NUM_EPOCHS}_lr{lr_str}_bs{config.BATCH_SIZE}"
                )
            else:
                run_name = (
                    f"{config.DATASET_NAME}/TransVert/"
                    f"ep{config.NUM_EPOCHS}_lr{lr_str}_bs{config.BATCH_SIZE}"
                )

            wandb.init(
                project=config.WANDB_PROJECT,
                name=run_name,
                config={
                    "model": config.MODEL_TYPE,
                    "model_size": model_size,
                    "dataset": config.DATASET_NAME,
                    "batch_size": config.BATCH_SIZE,
                    "epochs": config.NUM_EPOCHS,
                    "learning_rate": config.LEARNING_RATE,
                    "device": device.type,
                    "use_amp": use_amp,
                },
                tags=[config.DATASET_NAME, config.MODEL_TYPE, model_size],
            )
            wandb_enabled = True
            logger.info("W&B logging enabled")
        except Exception as e:
            logger.warning(f"W&B init failed ({e}); continuing without W&B")
            wandb_enabled = False

    # -- Checkpoint manager --
    model_size = getattr(config, "MODEL_SIZE", "default")
    ckpt_manager = CheckpointManager(
        save_dir=config.SAVE_DIR,
        dataset=config.DATASET_NAME,
        model_type=config.MODEL_TYPE,
        model_size=model_size,
    )

    # -- Training loop --
    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_epoch = 0
    early_stop_counter = 0

    logger.info("Starting training...")

    try:
        for epoch in range(config.NUM_EPOCHS):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, config, epoch, config.NUM_EPOCHS, wandb_enabled,
            )
            avg_train_loss, avg_train_bce, avg_train_dice_loss, avg_grad_norm, epoch_time = train_metrics

            # Validate
            _ds = config.DATASET_NAME.lower()
            clinical_task = (
                "femur" if "femur" in _ds
                else "hip" if "hip" in _ds or "pelv" in _ds
                else "verse" if "verse" in _ds
                else None
            )
            val_metrics = validate_one_epoch(
                model, val_loader, criterion, device, config,
                epoch, config.NUM_EPOCHS, wandb_enabled, clinical_task,
                logger=logger,
            )
            avg_val_loss, avg_val_dice, avg_val_bce, avg_val_dice_loss, clinical_metrics = val_metrics

            # Scheduler
            scheduler.step(avg_val_loss)

            # Best model check
            is_best = avg_val_dice > best_val_dice
            if is_best:
                best_val_loss = avg_val_loss
                best_val_dice = avg_val_dice
                best_epoch = epoch + 1
                early_stop_counter = 0
                ckpt_manager.save_best(
                    model, optimizer, scheduler,
                    epoch=epoch + 1,
                    val_loss=avg_val_loss,
                    val_dice=avg_val_dice,
                )
                logger.log_best(epoch + 1, avg_val_dice, avg_val_loss)
            else:
                early_stop_counter += 1

            # Periodic checkpoint
            save_interval = getattr(config, "SAVE_INTERVAL", 0)
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                ckpt_manager.save_periodic(
                    model, optimizer, scheduler,
                    epoch=epoch + 1,
                    val_loss=avg_val_loss,
                    val_dice=avg_val_dice,
                )

            # W&B logging
            cur_lr = optimizer.param_groups[0]["lr"]
            if wandb_enabled:
                try:
                    gpu_mem_gb = (
                        torch.cuda.memory_allocated() / 1e9
                        if torch.cuda.is_available() else 0
                    )
                    samples_per_sec = (
                        len(train_loader) * config.BATCH_SIZE / epoch_time
                        if epoch_time > 0 else 0
                    )
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": avg_train_loss,
                        "train/bce_loss": avg_train_bce,
                        "train/dice_loss": avg_train_dice_loss,
                        "train/grad_norm": avg_grad_norm,
                        "val/loss": avg_val_loss,
                        "val/bce_loss": avg_val_bce,
                        "val/dice_loss": avg_val_dice_loss,
                        "val/dice": avg_val_dice,
                        **(
                            _build_clinical_wandb_dict(clinical_metrics)
                            if clinical_metrics is not None
                            else {}
                        ),
                        "best/val_loss": best_val_loss,
                        "best/val_dice": best_val_dice,
                        "best/epoch": best_epoch,
                        "learning_rate": cur_lr,
                        "train/early_stop_counter": early_stop_counter,
                        "performance/epoch_time_sec": epoch_time,
                        "performance/samples_per_sec": samples_per_sec,
                        "performance/gpu_memory_gb": gpu_mem_gb,
                    })
                except Exception:
                    pass

            # Logger (file + console + CSV)
            logger.log_epoch(
                epoch=epoch + 1,
                total=config.NUM_EPOCHS,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_dice=avg_val_dice,
                lr=cur_lr,
                patience=early_stop_counter,
                patience_max=config.PATIENCE,
                train_bce=avg_train_bce,
                train_dice_loss=avg_train_dice_loss,
                grad_norm=avg_grad_norm,
                val_bce=avg_val_bce,
                val_dice_loss=avg_val_dice_loss,
                epoch_time=epoch_time,
                is_best=is_best,
            )
            logger.log_clinical(clinical_metrics)

            # Early stopping
            if early_stop_counter >= config.PATIENCE:
                logger.log_early_stop(
                    epoch + 1, config.PATIENCE, best_val_dice, best_epoch,
                )
                break

    finally:
        if wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass

        model_size = getattr(config, "MODEL_SIZE", "default")
        if config.MODEL_TYPE == "swinx2s":
            final_dir = os.path.join(
                config.SAVE_DIR, "swinx2s", model_size, config.DATASET_NAME,
            )
        else:
            final_dir = os.path.join(
                config.SAVE_DIR, "transvert", config.DATASET_NAME,
            )
        logger.log_summary(best_val_dice, best_val_loss, best_epoch, final_dir)
        logger.close()

    return True


# ------------------------------------------------------------------
# Batch training
# ------------------------------------------------------------------

def batch_train():
    """Train multiple configurations sequentially."""
    from datetime import datetime

    print("=" * 80)
    print("Batch Training Mode")
    print("=" * 80)
    print(f"\nTotal jobs: {len(TRAINING_JOBS)}")
    for i, (cfg, ep, desc) in enumerate(TRAINING_JOBS, 1):
        print(f"  {i}. {desc} ({cfg}, {ep} epochs)")

    resp = input("\nProceed? (yes/no): ").strip().lower()
    if resp != "yes":
        print("Cancelled.")
        return

    results = []
    start = datetime.now()

    for i, (config_name, ep, desc) in enumerate(TRAINING_JOBS, 1):
        print(f"\n{'#' * 80}")
        print(f"# Job {i}/{len(TRAINING_JOBS)}: {desc}")
        print(f"{'#' * 80}\n")
        success = train_single_config(config_name, ep)
        results.append((desc, success))
        if not success:
            r = input("Job failed. Continue? (yes/no): ").strip().lower()
            if r != "yes":
                break

    end = datetime.now()
    ok = sum(s for _, s in results)
    print(f"\n{'=' * 80}")
    print(f"Batch complete: {ok}/{len(results)} succeeded | Duration: {end - start}")
    print("=" * 80)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    if "--batch" in sys.argv:
        if not BATCH_MODE_AVAILABLE:
            print("Error: training_jobs.py not found.")
            return
        batch_train()
    else:
        args = parse_args()
        config_class = get_config(args.config)
        config = apply_args_to_config(config_class, args)
        train_single_config(args.config, config.NUM_EPOCHS)


if __name__ == "__main__":
    main()
