"""
X-Ray to 3D Bone Reconstruction - Training Script

Usage:
  python train.py --config swinx2s_femur                          # single run
  python train.py --config swinx2s_femur --model_size small       # with model size
  python train.py --config swinx2s_femur --no_wandb               # without W&B
  python train.py --list_configs                                  # list configs
  python train.py --batch                                         # batch mode
"""
import os
import sys
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def load_local_env(path: str) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load local, Git-ignored experiment credentials before W&B is initialized.
load_local_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

try:
    import wandb
except ImportError:
    wandb = None

from configs import parse_args, apply_args_to_config, get_config
from utils.data.dataset import BoneDataset, collate_bone
from utils.training.losses import DiceBCELoss
from utils.training.trainer import train_one_epoch, validate_one_epoch
from utils.training.checkpoint import CheckpointManager
from utils.training.logger import TrainingLogger
from models.swinx2s import SwinX2S
from models.transvert import DualViewCNN
from models.multiscale_dualview_cnn import MultiScaleDualViewCNN

# Optional batch-mode support
try:
    from training_jobs import TRAINING_JOBS
    BATCH_MODE_AVAILABLE = True
except ImportError:
    BATCH_MODE_AVAILABLE = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all random number generators used by this training process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def validate_config(config) -> None:
    """Fail early for invalid settings that would otherwise waste a run."""
    if config.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1")
    if getattr(config, "NUM_WORKERS", 0) < 0:
        raise ValueError("NUM_WORKERS must be non-negative")
    if config.NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be at least 1")
    if config.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive")
    if config.BCE_WEIGHT < 0 or config.DICE_WEIGHT < 0:
        raise ValueError("loss weights must be non-negative")
    if config.BCE_WEIGHT + config.DICE_WEIGHT == 0:
        raise ValueError("at least one loss weight must be positive")
    if config.OPTIMIZER.lower() not in {"adam", "adamw"}:
        raise ValueError("OPTIMIZER must be 'adam' or 'adamw'")
    if config.LR_SCHEDULER.lower() not in {"plateau", "warmup_cosine", "none"}:
        raise ValueError("LR_SCHEDULER must be 'plateau', 'warmup_cosine', or 'none'")
    canonical_path = os.path.join(config.DATA_ROOT, "train")
    if not os.path.isdir(canonical_path):
        raise FileNotFoundError(f"Canonical sample store not found: {canonical_path}")


def load_split_manifest(config):
    """Load a reproducible manifest with mutually exclusive train/val/test PIDs."""
    default_path = os.path.join(
        config.PROJECT_ROOT, "splits", f"{config.DATASET_NAME}_seed{config.RANDOM_SEED}.json",
    )
    path = getattr(config, "SPLIT_MANIFEST", default_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run scripts/create_patient_splits.py first."
        )
    with open(path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("dataset") not in (None, config.DATASET_NAME):
        raise ValueError(f"Split manifest {path} is for dataset {manifest.get('dataset')!r}, "
                         f"not {config.DATASET_NAME!r}")
    splits = manifest.get("splits", {})
    required = {"train", "val", "test"}
    if set(splits) != required or any(not isinstance(splits[name], list) for name in required):
        raise ValueError(f"Invalid split manifest format: {path}")
    split_sets = {name: set(splits[name]) for name in required}
    if any(len(split_sets[name]) != len(splits[name]) for name in required):
        raise ValueError(f"Duplicate PID in split manifest: {path}")
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise ValueError(f"Overlapping PID(s) in split manifest: {path}")
    return path, splits

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
            geometry=getattr(config, "VIEW_GEOMETRY", "legacy"),
            output_mode=getattr(config, "SWIN_OUTPUT_MODE", "fuse"),
            swin_impl=getattr(config, "SWIN_IMPL", "legacy"),
        ).to(device)
    elif config.MODEL_TYPE == "dualviewcnn":
        model = DualViewCNN(
            in_channels=config.IN_CHANNELS,
            mid_channels=config.MID_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            intermediate_depth=config.INTERMEDIATE_DEPTH,
            geometry=getattr(config, "VIEW_GEOMETRY", "legacy"),
            input_width=config.IMG_SIZE[1],
            norm=getattr(config, "DVC_NORM", "batch"),
        ).to(device)
    elif config.MODEL_TYPE == "multiscale_dualviewcnn":
        model = MultiScaleDualViewCNN(
            output_size=config.VOLUME_SIZE,
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            num_scales=config.NUM_SCALES,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    return model


def create_optimizer_and_scheduler(model, config):
    """Build the optimizer and schedule selected by the experiment config."""
    optimizer_cls = optim.AdamW if config.OPTIMIZER.lower() == "adamw" else optim.Adam
    optimizer = optimizer_cls(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )

    scheduler_name = config.LR_SCHEDULER.lower()
    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )
    elif scheduler_name == "warmup_cosine":
        warmup_epochs = min(config.WARMUP_EPOCHS, max(0, config.NUM_EPOCHS - 1))
        cosine_epochs = max(1, config.NUM_EPOCHS - warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=config.MIN_LR,
        )
        if warmup_epochs:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine
    else:
        scheduler = None
    return optimizer, scheduler


# ------------------------------------------------------------------
# Core training loop
# ------------------------------------------------------------------

def train_single_config(config_name, epochs=None, config=None, dry_run=False):
    """Train a single configuration. Returns True on success."""

    # -- Config --
    config_class = config or get_config(config_name)
    if epochs is not None:
        config_class.NUM_EPOCHS = epochs
    config = config_class
    config.RUN_ID = getattr(
        config, "RUN_ID", f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    validate_config(config)
    seed_everything(config.RANDOM_SEED, getattr(config, "DETERMINISTIC", True))

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

    # -- Datasets --
    try:
        if config.MODEL_TYPE == "swinx2s":
            img_size_2d = config.IMG_SIZE[1:]
            img_size_3d = tuple(config.IMG_SIZE)
        elif config.MODEL_TYPE == "multiscale_dualviewcnn":
            img_size_2d = tuple(config.IMG_SIZE)
            img_size_3d = tuple(config.VOLUME_SIZE)
        else:
            img_size_2d = tuple(config.IMG_SIZE)
            img_size_3d = (
                config.INTERMEDIATE_DEPTH * 4,
                config.IMG_SIZE[0],
                config.IMG_SIZE[1],
            )

        manifest_path, split_pids = load_split_manifest(config)
        # train/ is the complete canonical sample store.  The manifest selects
        # mutually exclusive train, validation, and held-out test PIDs from it.
        canonical_root = os.path.join(config.DATA_ROOT, "train")
        train_set = BoneDataset(
            root_dir=canonical_root,
            img_size_2d=img_size_2d,
            img_size_3d=img_size_3d,
            return_volume=True,
            include_pids=split_pids["train"],
            augment=getattr(config, "AUGMENT_TRAIN", False),
            flip_ap_lr=getattr(config, "FLIP_AP_LR", False),
        )
        val_set = BoneDataset(
            root_dir=canonical_root,
            img_size_2d=img_size_2d,
            img_size_3d=img_size_3d,
            return_volume=True,
            include_pids=split_pids["val"],
            flip_ap_lr=getattr(config, "FLIP_AP_LR", False),
        )
        test_set = BoneDataset(
            root_dir=canonical_root,
            img_size_2d=img_size_2d,
            img_size_3d=img_size_3d,
            return_volume=False,
            include_pids=split_pids["test"],
            flip_ap_lr=getattr(config, "FLIP_AP_LR", False),
        )

    except Exception as e:
        logger.warning(f"Error loading dataset: {e}")
        logger.close()
        return False

    if len(train_set) == 0 or len(val_set) == 0:
        logger.warning("Train or validation set is empty!")
        logger.close()
        return False

    logger.info(
        f"Split manifest: {os.path.relpath(manifest_path, config.PROJECT_ROOT)} | "
        f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}"
    )

    if dry_run:
        # Force decoding/resampling now, so corrupt images or volumes fail before
        # a long run begins.  No checkpoint or W&B run is created.
        train_sample = train_set[0]
        val_sample = val_set[0]
        logger.info(
            "Dry run passed: "
            f"train {train_sample['pid']} xa={tuple(train_sample['xa'].shape)} "
            f"vol={tuple(train_sample['vol'].shape)}; "
            f"val {val_sample['pid']}"
        )
        logger.close()
        return True

    # -- Data loaders --
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.RANDOM_SEED)
    num_workers = getattr(config, "NUM_WORKERS", 2)
    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": getattr(config, "PIN_MEMORY", device.type == "cuda"),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_bone,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_bone,
        **common_loader_args,
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

    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # -- Weights & Biases --
    wandb_enabled = False
    if config.USE_WANDB and wandb is not None:
        try:
            model_size = getattr(config, "MODEL_SIZE", "small").upper()
            lr_str = f"{config.LEARNING_RATE:.0e}".replace("e-0", "e-")
            if config.MODEL_TYPE == "swinx2s":
                run_name = (
                    f"{config.DATASET_NAME}/SwinX2S-{model_size}/"
                    f"ep{config.NUM_EPOCHS}_lr{lr_str}_bs{config.BATCH_SIZE}"
                )
            elif config.MODEL_TYPE == "dualviewcnn":
                run_name = (
                    f"{config.DATASET_NAME}/DualViewCNN/"
                    f"ep{config.NUM_EPOCHS}_lr{lr_str}_bs{config.BATCH_SIZE}"
                )
            else:
                run_name = (
                    f"{config.DATASET_NAME}/MultiScaleDualViewCNN/"
                    f"ep{config.NUM_EPOCHS}_lr{lr_str}_bs{config.BATCH_SIZE}"
                )

            wandb.init(
                project=config.WANDB_PROJECT,
                name=f"{run_name}/{config.RUN_ID}",
                config={
                    "model": config.MODEL_TYPE,
                    "model_size": model_size,
                    "dataset": config.DATASET_NAME,
                    "batch_size": config.BATCH_SIZE,
                    "epochs": config.NUM_EPOCHS,
                    "learning_rate": config.LEARNING_RATE,
                    "optimizer": config.OPTIMIZER,
                    "weight_decay": config.WEIGHT_DECAY,
                    "lr_scheduler": config.LR_SCHEDULER,
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
    elif config.USE_WANDB:
        logger.warning("W&B is not installed; continuing without W&B logging")

    # -- Checkpoint manager --
    model_size = getattr(config, "MODEL_SIZE", "default")
    ckpt_manager = CheckpointManager(
        save_dir=config.SAVE_DIR,
        dataset=config.DATASET_NAME,
        model_type=config.MODEL_TYPE,
        model_size=model_size,
        run_name=config.RUN_ID,
        save_optimizer=getattr(config, "SAVE_OPTIMIZER_STATE", True),
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
            val_metrics = validate_one_epoch(
                model, val_loader, criterion, device, config,
                epoch, config.NUM_EPOCHS,
            )
            avg_val_loss, avg_val_dice, avg_val_bce, avg_val_dice_loss, segmentation_metrics = val_metrics

            # Scheduler
            if scheduler is not None:
                if config.LR_SCHEDULER.lower() == "plateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

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
                        **{f"val/{name}": value for name, value in segmentation_metrics.items()},
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
                segmentation_metrics=segmentation_metrics,
                epoch_time=epoch_time,
                is_best=is_best,
            )

            # Early stopping
            if early_stop_counter >= config.PATIENCE:
                logger.log_early_stop(
                    epoch + 1, config.PATIENCE, best_val_dice, best_epoch,
                )
                break

    except BaseException:
        logger.warning("Training interrupted or failed; see the traceback. Run is incomplete.")
        raise
    else:
        logger.log_summary(best_val_dice, best_val_loss, best_epoch, ckpt_manager.best_dir)
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
        elif config.MODEL_TYPE == "dualviewcnn":
            final_dir = os.path.join(
                config.SAVE_DIR, "dualviewcnn", config.DATASET_NAME,
            )
        else:
            final_dir = os.path.join(
                config.SAVE_DIR, "multiscale_dualviewcnn", config.DATASET_NAME,
            )
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
        config_class = type(
            f"{get_config(args.config).__name__}RunConfig",
            (get_config(args.config),),
            {},
        )
        config = apply_args_to_config(config_class, args)
        ok = train_single_config(
            args.config, config.NUM_EPOCHS, config=config, dry_run=args.dry_run,
        )
        if ok is False:
            sys.exit(1)  # so launch scripts and Modal see a failed run as failed


if __name__ == "__main__":
    main()
