import os
import torch

# Base configuration class
class BaseConfig:
    """Base configuration with common settings"""
    
    # Model type
    MODEL_TYPE = "swinx2s"  # Options: "swinx2s", "transvert"
    
    # Common paths – resolved dynamically from this file's location
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # Training hyperparameters
    BATCH_SIZE = 1
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    GRAD_CLIP = 1.0
    PATIENCE = 10
    
    # Dataset split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Loss weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    
    # Seed
    RANDOM_SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = torch.cuda.is_available()
    
    # Logging
    USE_WANDB = True
    WANDB_PROJECT = "X-Ray-To-3d-Final"
    
    # Target settings
    TARGET_KEY = "vol"
    BINARY_THRESH = 0.5
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print(f"\n{'='*70}")
        print(f"⚙️  Configuration: {cls.__name__}")
        print(f"{'='*70}")
        print(f"Model Type:      {cls.MODEL_TYPE}")
        model_size = getattr(cls, 'MODEL_SIZE', 'default')
        print(f"Model Size:      {model_size}")
        print(f"Dataset:         {cls.DATASET_NAME}")
        print(f"Data Root:       {cls.DATA_ROOT}")
        print(f"Save Directory:  {cls.SAVE_DIR}")
        print(f"Image Size:      {cls.IMG_SIZE}")
        print(f"Batch Size:      {cls.BATCH_SIZE}")
        print(f"Epochs:          {cls.NUM_EPOCHS}")
        print(f"Learning Rate:   {cls.LEARNING_RATE}")
        print(f"Device:          {cls.DEVICE.upper()}")
        if torch.cuda.is_available():
            print(f"GPU:             {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*70}\n")


# ============================================================================
# SwinX2S Configurations
# ============================================================================

class SwinX2S_Femur_Tiny(BaseConfig):
    """SwinX2S Tiny for Femur - lightweight baseline
    Original CT: 85×85×85 (uniform, 1.5mm iso), X-ray: 128×128
    Tiny: 2D 96×128, 3D 96×96×128 | ~32M params
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "tiny"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = [96, 96, 128]
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Femur(BaseConfig):
    """SwinX2S OPTIMIZED for Femur - recommended default
    Original CT: 85×85×85 (uniform, 1.5mm iso), X-ray: 128×128

    Design rationale:
      - 96×96×128 is the closest SwinX2S-compatible size to 85³
        (only ~13% upsampling vs 85% with 128×128×160)
      - feature_size=48 gives ~70M params — more capacity than Tiny (32M)
        without the waste of Small (130M) on a simple anatomy
      - num_heads=(2,4,8,16) maintains 24-dim attention heads throughout
      - Femur has zero dimension variability (all 85³), so the model
        can learn a very consistent mapping

    2D input: 96×128 | 3D volume: 96×96×128 | ~70M params
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "femur_opt"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = [96, 96, 128]   # closest to 85³ that's divisible by 32
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 48          # ~70M params — sweet spot for femur
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)  # 24-dim heads at every stage
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3
    
    # Femur-specific training overrides
    LEARNING_RATE = 5e-5       # lower LR for stability
    PATIENCE = 15              # more patience before early stop
    NUM_EPOCHS = 200           # more epochs for convergence
    SAVE_INTERVAL = 10         # periodic checkpoints every 10 epochs


class SwinX2S_Femur_Small(BaseConfig):
    """SwinX2S Small for Femur - high-capacity variant
    Original CT: 85×85×85, X-ray: 128×128
    Standard: 2D 128×160, 3D 128×128×160 | ~130M params
    Note: significant upsampling from 85³ — use only if tiny/opt underfit
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = [128, 128, 160]
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Hip(BaseConfig):
    """SwinX2S configuration for Hip dataset
    Original CT: 192×192×192, X-ray: 288×288
    Standard variant: 2D X-ray 128×160, 3D volume 128×128×160
    Model: Swin-X2S-Small (~130M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "hip"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Hip")
    
    # Swin-X2S-Small configuration
    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64  # Hidden dimension
    DEPTHS = (2, 2, 6, 2)  # Block numbers
    NUM_HEADS = (2, 4, 8, 16)  # Head numbers
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Rib(BaseConfig):
    """SwinX2S configuration for Rib dataset
    Original CT: 213×213×213, X-ray: 320×320
    Standard variant: 2D X-ray 128×160, 3D volume 128×128×160
    Model: Swin-X2S-Base (~314M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "base"
    DATASET_NAME = "rib"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Rib")
    
    # Swin-X2S-Base configuration
    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 1  # Large model needs smaller batch
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 96  # Hidden dimension
    DEPTHS = (2, 2, 18, 2)  # Block numbers
    NUM_HEADS = (3, 6, 12, 24)  # Head numbers
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Verse19(BaseConfig):
    """SwinX2S configuration for Verse19 dataset
    Original CT: 96×96×96 (variable), X-ray: 96×96
    Tiny variant: 2D X-ray 96×128, 3D volume 96×96×128
    Model: Swin-X2S-Tiny (~32M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "tiny"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")
    
    # Swin-X2S-Tiny configuration
    IMG_SIZE = [96, 96, 128]  # [D, H, W] → 2D input: 96×128, 3D volume: 96×96×128
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Verse19_Small(BaseConfig):
    """SwinX2S Small for Verse19 - matches checkpoint saved_models_swinx2s_small_verse19
    FEATURE_SIZE=64, IMG_SIZE=[128,128,160] (Standard/Small variant)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")

    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_Verse20(BaseConfig):
    """SwinX2S configuration for Verse20 dataset
    Original CT: 133×137×133 (variable), X-ray: 96×96
    Tiny variant: 2D X-ray 96×128, 3D volume 96×96×128
    Model: Swin-X2S-Tiny (~32M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "tiny"
    DATASET_NAME = "verse20"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse20")
    
    # Swin-X2S-Tiny configuration (matches Verse19)
    IMG_SIZE = [96, 96, 128]  # [D, H, W] → 2D input: 96×128, 3D volume: 96×96×128
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


class SwinX2S_CTPelvic(BaseConfig):
    """SwinX2S configuration for CTPelvic dataset
    Original CT: 368×360×368 (highly variable), X-ray: 288×288
    Standard variant: 2D X-ray 128×160, 3D volume 128×128×160
    Model: Swin-X2S-Small (~130M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "ctpelvic"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "ctpelvik")
    
    # Swin-X2S-Small configuration
    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 1  # Large volumes need batch size 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64  # Hidden dimension
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (8, 8)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (2, 2, 2, 2, 2, 2)
    SPATIAL_DIMS = 3


# ============================================================================
# TransVert Configurations
# ============================================================================

class TransVert_Femur(BaseConfig):
    """TransVert configuration for Femur dataset
    Original CT: 85×85×85, X-ray: 128×128
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 8
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class TransVert_Hip(BaseConfig):
    """TransVert configuration for Hip dataset
    Original CT: 192×192×192, X-ray: 288×288
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "hip"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Hip")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


class TransVert_Rib(BaseConfig):
    """TransVert configuration for Rib dataset
    Original CT: 213×213×213, X-ray: 320×320
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "rib"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Rib")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 48
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


class TransVert_Verse19(BaseConfig):
    """TransVert configuration for Verse19 dataset
    Original CT: 96×96×96 (variable), X-ray: 96×96
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 8
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class TransVert_Verse20(BaseConfig):
    """TransVert configuration for Verse20 dataset
    Original CT: 133×137×133 (variable), X-ray: 96×96
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "verse20"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse20")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 8
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class TransVert_CTPelvic(BaseConfig):
    """TransVert configuration for CTPelvic dataset
    Original CT: 368×360×368 (highly variable), X-ray: 288×288
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "transvert"
    DATASET_NAME = "ctpelvic"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "ctpelvik")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


# ============================================================================
# Config Registry
# ============================================================================

CONFIG_REGISTRY = {
    # SwinX2S configs
    "swinx2s_femur": SwinX2S_Femur,            # recommended (optimized ~70M)
    "swinx2s_femur_tiny": SwinX2S_Femur_Tiny,   # lightweight ~32M
    "swinx2s_femur_small": SwinX2S_Femur_Small,  # high-capacity ~130M
    "swinx2s_hip": SwinX2S_Hip,
    "swinx2s_rib": SwinX2S_Rib,
    "swinx2s_verse19": SwinX2S_Verse19,          # tiny ~32M
    "swinx2s_verse19_small": SwinX2S_Verse19_Small,  # small ~130M (matches saved checkpoint)
    "swinx2s_verse20": SwinX2S_Verse20,
    "swinx2s_ctpelvic": SwinX2S_CTPelvic,
    
    # TransVert configs
    "transvert_femur": TransVert_Femur,
    "transvert_hip": TransVert_Hip,
    "transvert_rib": TransVert_Rib,
    "transvert_verse19": TransVert_Verse19,
    "transvert_verse20": TransVert_Verse20,
    "transvert_ctpelvic": TransVert_CTPelvic,
}


def get_config(config_name: str):
    """Get configuration by name"""
    if config_name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    return CONFIG_REGISTRY[config_name]


def list_configs():
    """List all available configurations"""
    print("\n" + "="*70)
    print("📋 Available Configurations")
    print("="*70)
    
    print("\n🔹 SwinX2S Configs:")
    for name in CONFIG_REGISTRY:
        if name.startswith("swinx2s"):
            print(f"  - {name}")
    
    print("\n🔹 TransVert Configs:")
    for name in CONFIG_REGISTRY:
        if name.startswith("transvert"):
            print(f"  - {name}")
    
    print("\n" + "="*70)
    print("Usage: python train.py --config <config_name>")
    print("="*70 + "\n")