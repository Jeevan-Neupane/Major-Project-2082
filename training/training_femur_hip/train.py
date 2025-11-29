import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import wandb

from configs.config import Config
from utils.dataset import SmallBoneDataset, collate_small
from utils.losses import DiceBCELoss
from utils.trainer import train_one_epoch, validate_one_epoch
from utils.checkpoint import save_checkpoint
from models.swinx2s import SwinX2S
from models.transvert_model import TransVert


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SwinX2S or TransVert model for 3D reconstruction from X-rays',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='swinx2s',
                        choices=['swinx2s', 'transvert'],
                        help='Model architecture to use')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default=Config.DATA_ROOT,
                        help='Path to dataset directory')
    parser.add_argument('--save-dir', type=str, default=Config.SAVE_DIR,
                        help='Directory to save model checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--train-ratio', type=float, default=Config.TRAIN_RATIO,
                        help='Ratio of training data (0-1)')
    
    # Logging arguments
    parser.add_argument('--wandb', action='store_true', default=Config.USE_WANDB,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb',
                        help='Disable Weights & Biases logging')
    
    return parser.parse_args()


def create_model(model_name, device):
    """Create model based on name."""
    if model_name == 'swinx2s':
        print(f"🏗️  Creating SwinX2S model...")
        model = SwinX2S(
            img_size=Config.IMG_SIZE,
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            feature_size=Config.FEATURE_SIZE,
            depths=Config.DEPTHS,
            num_heads=Config.NUM_HEADS,
            window_size=Config.WINDOW_SIZE,
            norm_name=Config.NORM_NAME,
        ).to(device)
        
    elif model_name == 'transvert':
        print(f"🏗️  Creating TransVert model...")
        model = TransVert(
            in_channels=Config.IN_CHANNELS,
            mid_channels=32,
            out_channels=Config.OUT_CHANNELS,
            intermediate_depth=16,
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(Config.DEVICE)
    use_amp = Config.USE_AMP
    
    # Print device info
    Config.print_device_info()
    
    # Create directories
    Config.create_dirs()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print(f"{'='*60}")
    print(f"🖥️  TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:          {args.model.upper()}")
    print(f"Dataset:        {args.data_dir}")
    print(f"Device:         {device.type.upper()}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Train Ratio:    {args.train_ratio}")
    print(f"Val Ratio:      {Config.VAL_RATIO}")
    print(f"Learning Rate:  {Config.LEARNING_RATE}")
    print(f"Save Dir:       {args.save_dir}")
    print(f"W&B Logging:    {'✅ Enabled' if args.wandb else '❌ Disabled'}")
    print(f"{'='*60}\n")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' does not exist!")
        print(f"   Please create it or specify correct path with --data-dir")
        return
    
    # Dataset
    try:
        dataset = SmallBoneDataset(
            root_dir=args.data_dir,
            img_size_2d=Config.IMG_SIZE[1:],
            img_size_3d=tuple(Config.IMG_SIZE),
            return_volume=True
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Please check your data directory: {args.data_dir}")
        return
    
    if len(dataset) == 0:
        print(f"❌ Error: Dataset is empty!")
        print(f"   Please check your data directory: {args.data_dir}")
        return
    
    # Split
    total_size = len(dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(Config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    if train_size == 0 or val_size == 0:
        print(f"❌ Error: Train or validation set is empty!")
        print(f"   Total samples: {total_size}")
        print(f"   Try adjusting --train-ratio or add more data")
        return
    
    print(f"📊 Dataset Split:")
    print(f"   Total:  {total_size}")
    print(f"   Train:  {train_size} ({args.train_ratio*100:.0f}%)")
    print(f"   Val:    {val_size} ({Config.VAL_RATIO*100:.0f}%)")
    print(f"   Test:   {test_size}\n")
    
    generator = torch.Generator().manual_seed(Config.RANDOM_SEED)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Loaders
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type=='cuda'), 
        collate_fn=collate_small
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type=='cuda'), 
        collate_fn=collate_small
    )
    
    # Create model
    model = create_model(args.model, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 Model: {num_params:,} parameters ({num_params/1e6:.2f}M)\n")
    
    # Loss & Optimizer
    criterion = DiceBCELoss(
        bce_weight=Config.BCE_WEIGHT,
        dice_weight=Config.DICE_WEIGHT
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    # Wandb
    if args.wandb:
        wandb.init(
            project=Config.WANDB_PROJECT,
            name=f"{args.model}_{args.epochs}ep",
            config={
                'model': args.model,
                'img_size': Config.IMG_SIZE,
                'feature_size': Config.FEATURE_SIZE,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': Config.LEARNING_RATE,
                'train_ratio': args.train_ratio,
                'val_ratio': Config.VAL_RATIO,
                'device': device.type,
                'use_amp': use_amp,
            }
        )
    
    # Create a config object for trainer functions
    class TrainConfig:
        TARGET_KEY = Config.TARGET_KEY
        USE_AMP = use_amp
        GRAD_CLIP = Config.GRAD_CLIP
    
    train_config = TrainConfig()
    
    # Training loop
    best_val_loss = float("inf")
    early_stop_counter = 0
    
    print("🚀 Starting training...\n")
    
    for epoch in range(args.epochs):
        # Train
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            device, train_config, epoch, args.epochs, args.wandb
        )
        
        # Validate
        avg_val_loss, avg_val_dice = validate_one_epoch(
            model, val_loader, criterion, device, train_config, 
            epoch, args.epochs, args.wandb
        )
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch+1, avg_val_loss, 
                          save_path, is_best=True)
        else:
            early_stop_counter += 1
        
        # Logging
        cur_lr = optimizer.param_groups[0]["lr"]
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/dice": avg_val_dice,
                "learning_rate": cur_lr
            })
        
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"LR: {cur_lr:.2e}")
        
        # Early stopping
        if early_stop_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if args.wandb:
        wandb.finish()
    
    print("\n✅ Training complete!")
    print(f"💾 Best model saved to: {os.path.join(args.save_dir, f'{args.model}_best.pth')}\n")


if __name__ == "__main__":
    main()