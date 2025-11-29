import os
import torch

class Config:
    # Data paths
    DATA_ROOT = r"C:\Users\jeevan\OneDrive\Desktop\Major Project\swinx2s-project\data\femur"
    SAVE_DIR = r"C:\Users\jeevan\OneDrive\Desktop\Major Project\swinx2s-project\saved_models"
    
    # Model architecture
    IMG_SIZE = [64, 128, 128]  # D, H, W
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 48
    DEPTHS = (2, 2, 2, 2)
    NUM_HEADS = (2, 4, 6, 8)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    
    # Training
    BATCH_SIZE = 1  # Increase to 2-4 if you have GPU with enough VRAM
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    GRAD_CLIP = 1.0
    PATIENCE = 10
    
    # Loss weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    RANDOM_SEED = 42
    
    # Device - ✅ AUTO-DETECT
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = torch.cuda.is_available()  # Only use mixed precision on GPU
    
    # Logging
    USE_WANDB = True
    WANDB_PROJECT = "SwinX2S-Training"
    
    # Target key in dataset
    TARGET_KEY = "vol"
    BINARY_THRESH = 0.5
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
    
    @classmethod
    def print_device_info(cls):
        """Print device information"""
        print(f"{'='*60}")
        print(f"🖥️  Device Configuration")
        print(f"{'='*60}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device: {cls.DEVICE.upper()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Mixed Precision (AMP): ✅ Enabled")
        else:
            print(f"GPU: ❌ Not available")
            print(f"Mixed Precision (AMP): ❌ Disabled (CPU mode)")
        
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"{'='*60}\n")