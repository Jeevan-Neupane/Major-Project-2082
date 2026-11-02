"""
Command-line argument parser for training
"""
import argparse
from src.configs.config import CONFIG_REGISTRY, list_configs


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train X-Ray to 3D Bone Reconstruction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SwinX2S on Femur dataset
  python -m src.train --config swinx2s_femur
  
  # Train TransVert on Hip dataset with custom epochs
  python -m src.train --config transvert_hip --epochs 50
  
  # Train with custom batch size and learning rate
  python -m src.train --config swinx2s_verse19 --batch_size 2 --lr 5e-4
  
  # List all available configurations
  python -m src.train --list_configs
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Configuration name (e.g., swinx2s_femur, transvert_hip)"
    )
    
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List all available configurations and exit"
    )
    
    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save models (overrides config)"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        choices=['tiny', 'small', 'base', 'large'],
        default=None,
        help="SwinX2S model size: tiny (32M), small (130M), base (314M), large (558M)"
    )
    
    args = parser.parse_args()
    
    # Handle list_configs
    if args.list_configs:
        list_configs()
        exit(0)
    
    # Validate config is provided
    if not args.config:
        parser.error("--config is required (or use --list_configs to see available options)")
    
    # Validate config exists
    if args.config not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        parser.error(f"Config '{args.config}' not found. Available: {available}")
    
    return args


def apply_args_to_config(config_class, args):
    """Apply command-line arguments to config class"""
    # Apply model size if specified (only for SwinX2S)
    if hasattr(args, 'model_size') and args.model_size and config_class.MODEL_TYPE == 'swinx2s':
        # SwinX2S required input dimensions:
        #   Tiny:                3D = 96×96×128,  2D = 96×128
        #   Small / Base / Large: 3D = 128×128×160, 2D = 128×160
        size_configs = {
            'tiny': {
                'IMG_SIZE': [96, 96, 128],
                'FEATURE_SIZE': 32,
                'DEPTHS': (2, 2, 6, 2),
                'NUM_HEADS': (1, 2, 4, 8),
                'WINDOW_SIZE': (8, 8),
            },
            'small': {
                'IMG_SIZE': [128, 128, 160],
                'FEATURE_SIZE': 64,
                'DEPTHS': (2, 2, 6, 2),
                'NUM_HEADS': (2, 4, 8, 16),
                'WINDOW_SIZE': (8, 8),
            },
            'base': {
                'IMG_SIZE': [128, 128, 160],
                'FEATURE_SIZE': 96,
                'DEPTHS': (2, 2, 18, 2),
                'NUM_HEADS': (3, 6, 12, 24),
                'WINDOW_SIZE': (8, 8),
            },
            'large': {
                'IMG_SIZE': [128, 128, 160],
                'FEATURE_SIZE': 128,
                'DEPTHS': (2, 2, 18, 2),
                'NUM_HEADS': (4, 8, 16, 32),
                'WINDOW_SIZE': (8, 8),
            },
        }
        
        size_config = size_configs[args.model_size]
        config_class.IMG_SIZE = size_config['IMG_SIZE']
        config_class.FEATURE_SIZE = size_config['FEATURE_SIZE']
        config_class.DEPTHS = size_config['DEPTHS']
        config_class.NUM_HEADS = size_config['NUM_HEADS']
        config_class.WINDOW_SIZE = size_config['WINDOW_SIZE']
        config_class.MODEL_SIZE = args.model_size
        
        if args.model_size in ['base', 'large'] and args.batch_size is None:
            config_class.BATCH_SIZE = 1
    elif hasattr(args, 'model_size') and args.model_size:
        config_class.MODEL_SIZE = args.model_size
    
    # Override config values if provided
    if args.epochs is not None:
        config_class.NUM_EPOCHS = args.epochs
    
    if args.batch_size is not None:
        config_class.BATCH_SIZE = args.batch_size
    
    if args.lr is not None:
        config_class.LEARNING_RATE = args.lr
    
    if args.save_dir is not None:
        config_class.SAVE_DIR = args.save_dir
    
    if args.no_wandb:
        config_class.USE_WANDB = False
    
    if args.seed is not None:
        config_class.RANDOM_SEED = args.seed
    
    return config_class
