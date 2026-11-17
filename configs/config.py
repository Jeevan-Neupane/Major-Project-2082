import os
import torch

# Base configuration class
class BaseConfig:
    """Base configuration with common settings"""
    
    # Model type
    MODEL_TYPE = "swinx2s"  # Options: "swinx2s", "dualviewcnn"
    
    # Common paths – resolved dynamically from this file's location
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # Training hyperparameters
    BATCH_SIZE = 1
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    OPTIMIZER = "adam"
    WEIGHT_DECAY = 0.0
    LR_SCHEDULER = "plateau"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    WARMUP_EPOCHS = 0
    MIN_LR = 0.0
    GRAD_CLIP = 1.0
    PATIENCE = 10
    NUM_WORKERS = 2
    PIN_MEMORY = torch.cuda.is_available()
    # Reproducibility is preferred for research runs.  Set this to False only
    # when the small performance cost of deterministic CUDA kernels matters.
    DETERMINISTIC = True
    # Training-only paired AP/LAT and target-volume augmentation.
    AUGMENT_TRAIN = True
    
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
        print(f"Optimiser:       {cls.OPTIMIZER}")
        print(f"LR Scheduler:    {cls.LR_SCHEDULER}")
        print(f"Device:          {cls.DEVICE.upper()}")
        if torch.cuda.is_available():
            print(f"GPU:             {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*70}\n")


class SwinX2SResearchConfig(BaseConfig):
    """Paper-aligned defaults for Swin-X2S experiments.

    The official implementation uses AdamW, a 3e-5 initial learning rate,
    linear warm-up for 20 epochs, cosine decay over 250 epochs, a 7x7 window,
    and one cross-view fusion block.  Batch size remains one to make runs
    reproducible on a single 32 GB GPU.
    """
    NUM_EPOCHS = 250
    LEARNING_RATE = 3e-5
    OPTIMIZER = "adamw"
    # The official repository uses 0.5 with layer-wise parameter groups.  This
    # implementation has no such grouping, so use the stable AdamW baseline.
    WEIGHT_DECAY = 0.05
    LR_SCHEDULER = "warmup_cosine"
    WARMUP_EPOCHS = 20
    MIN_LR = 0.0
    PATIENCE = 60
    SAVE_INTERVAL = 25
    BATCH_SIZE = 1
    WINDOW_SIZE = (7, 7)
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)


class DualViewCNNResearchConfig(BaseConfig):
    """Stable direct-mask baseline for paired AP/lateral X-rays."""
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    OPTIMIZER = "adam"
    WEIGHT_DECAY = 0.0
    LR_SCHEDULER = "plateau"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10
    PATIENCE = 30
    SAVE_INTERVAL = 25


class MultiScaleDualViewCNNResearchConfig(BaseConfig):
    """Defaults for multi-scale 2D-to-3D CNN fusion experiments."""
    NUM_EPOCHS = 200
    LEARNING_RATE = 2e-4
    OPTIMIZER = "adam"
    WEIGHT_DECAY = 0.0
    LR_SCHEDULER = "plateau"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10
    PATIENCE = 30
    SAVE_INTERVAL = 25
    BASE_CHANNELS = 16
    NUM_SCALES = 4


# ============================================================================
# SwinX2S Configurations
# ============================================================================

class SwinX2S_Femur_Tiny(SwinX2SResearchConfig):
    """SwinX2S Tiny for Femur - lightweight baseline
    Original CT: 85×85×85 (uniform, 1.5mm iso), X-ray: 128×128
    Tiny: 2D 96×128, 3D 96×96×128 | ~32M params
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "tiny"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = [96, 96, 128]
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Femur(SwinX2SResearchConfig):
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
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 48          # ~70M params — sweet spot for femur
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)  # 24-dim heads at every stage
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3
    
    # Femur-specific training overrides
    # Kept as an experimental medium-capacity ablation, not the default.


class SwinX2S_Femur_Small(SwinX2SResearchConfig):
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
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Hip(SwinX2SResearchConfig):
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
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64  # Hidden dimension
    DEPTHS = (2, 2, 6, 2)  # Block numbers
    NUM_HEADS = (2, 4, 8, 16)  # Head numbers
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Rib(SwinX2SResearchConfig):
    """SwinX2S configuration for Rib dataset
    Original CT: 213×213×213, X-ray: 320×320
    Standard variant: 2D X-ray 128×160, 3D volume 128×128×160
    Model: Swin-X2S-Base (~314M params)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "rib"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Rib")
    
    # Swin-X2S-Base configuration
    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 1  # Large model needs smaller batch
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Verse19(SwinX2SResearchConfig):
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
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Verse19_Small(SwinX2SResearchConfig):
    """SwinX2S Small for Verse19 - matches checkpoint saved_models_swinx2s_small_verse19
    FEATURE_SIZE=64, IMG_SIZE=[128,128,160] (Standard/Small variant)
    """
    MODEL_TYPE = "swinx2s"
    MODEL_SIZE = "small"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")

    IMG_SIZE = [128, 128, 160]  # [D, H, W] → 2D input: 128×160, 3D volume: 128×128×160
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 64
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (2, 4, 8, 16)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_Verse20(SwinX2SResearchConfig):
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
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURE_SIZE = 32
    DEPTHS = (2, 2, 6, 2)
    NUM_HEADS = (1, 2, 4, 8)
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


class SwinX2S_CTPelvic(SwinX2SResearchConfig):
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
    WINDOW_SIZE = (7, 7)
    NORM_NAME = "batch"
    FUSION_DEPTHS = (1, 1, 1, 1, 1, 1)
    SPATIAL_DIMS = 3


# ============================================================================
# DualViewCNN configurations
# ============================================================================

class DualViewCNN_Femur(DualViewCNNResearchConfig):
    """DualViewCNN configuration for Femur dataset
    Original CT: 85×85×85, X-ray: 128×128
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class DualViewCNN_Hip(DualViewCNNResearchConfig):
    """DualViewCNN configuration for Hip dataset
    Original CT: 192×192×192, X-ray: 288×288
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "hip"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Hip")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


class DualViewCNN_Rib(DualViewCNNResearchConfig):
    """DualViewCNN configuration for Rib dataset
    Original CT: 213×213×213, X-ray: 320×320
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "rib"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Rib")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 2
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 48
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


class DualViewCNN_Verse19(DualViewCNNResearchConfig):
    """DualViewCNN configuration for Verse19 dataset
    Original CT: 96×96×96 (variable), X-ray: 96×96
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class DualViewCNN_Verse20(DualViewCNNResearchConfig):
    """DualViewCNN configuration for Verse20 dataset
    Original CT: 133×137×133 (variable), X-ray: 96×96
    Aligned with Tiny variant: 2D 96×128 → 3D 96×96×128
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "verse20"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse20")
    
    IMG_SIZE = (96, 128)  # 2D X-ray size (H, W)
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 24  # 3D depth = 24 * 4 = 96


class DualViewCNN_CTPelvic(DualViewCNNResearchConfig):
    """DualViewCNN configuration for CTPelvic dataset
    Original CT: 368×360×368 (highly variable), X-ray: 288×288
    Aligned with Standard variant: 2D 128×160 → 3D 128×128×160
    """
    MODEL_TYPE = "dualviewcnn"
    DATASET_NAME = "ctpelvic"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "ctpelvik")
    
    IMG_SIZE = (128, 160)  # 2D X-ray size (H, W)
    BATCH_SIZE = 1
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    MID_CHANNELS = 32
    INTERMEDIATE_DEPTH = 32  # 3D depth = 32 * 4 = 128


# ============================================================================
# MultiScaleDualViewCNN Configurations
# ============================================================================

class MultiScaleDualViewCNN_Femur(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "femur"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Femur")
    IMG_SIZE = (96, 128)
    VOLUME_SIZE = (96, 96, 128)
    BATCH_SIZE = 2
    IN_CHANNELS = OUT_CHANNELS = 1


class MultiScaleDualViewCNN_Hip(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "hip"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Hip")
    IMG_SIZE = (128, 160)
    VOLUME_SIZE = (128, 128, 160)
    BATCH_SIZE = 1
    IN_CHANNELS = OUT_CHANNELS = 1


class MultiScaleDualViewCNN_Rib(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "rib"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Total Segmentator", "Rib")
    IMG_SIZE = (128, 160)
    VOLUME_SIZE = (128, 128, 160)
    BATCH_SIZE = 1
    IN_CHANNELS = OUT_CHANNELS = 1


class MultiScaleDualViewCNN_Verse19(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "verse19"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse19")
    IMG_SIZE = (96, 128)
    VOLUME_SIZE = (96, 96, 128)
    BATCH_SIZE = 2
    IN_CHANNELS = OUT_CHANNELS = 1


class MultiScaleDualViewCNN_Verse20(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "verse20"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "Verse", "verse20")
    IMG_SIZE = (96, 128)
    VOLUME_SIZE = (96, 96, 128)
    BATCH_SIZE = 2
    IN_CHANNELS = OUT_CHANNELS = 1


class MultiScaleDualViewCNN_CTPelvic(MultiScaleDualViewCNNResearchConfig):
    MODEL_TYPE = "multiscale_dualviewcnn"
    DATASET_NAME = "ctpelvic"
    DATA_ROOT = os.path.join(BaseConfig.DATA_DIR, "ctpelvik")
    IMG_SIZE = (128, 160)
    VOLUME_SIZE = (128, 128, 160)
    BATCH_SIZE = 1
    IN_CHANNELS = OUT_CHANNELS = 1


# ============================================================================
# Config Registry
# ============================================================================

CONFIG_REGISTRY = {
    # SwinX2S configs
    "swinx2s_femur": SwinX2S_Femur_Tiny,       # recommended paper-aligned baseline
    "swinx2s_femur_tiny": SwinX2S_Femur_Tiny,   # lightweight ~32M
    "swinx2s_femur_small": SwinX2S_Femur_Small,  # high-capacity ~130M
    "swinx2s_hip": SwinX2S_Hip,
    "swinx2s_rib": SwinX2S_Rib,
    "swinx2s_verse19": SwinX2S_Verse19,          # tiny ~32M
    "swinx2s_verse19_small": SwinX2S_Verse19_Small,  # small ~130M (matches saved checkpoint)
    "swinx2s_verse20": SwinX2S_Verse20,
    "swinx2s_ctpelvic": SwinX2S_CTPelvic,
    
    # Dual-view CNN configs
    "dualviewcnn_femur": DualViewCNN_Femur,
    "dualviewcnn_hip": DualViewCNN_Hip,
    "dualviewcnn_rib": DualViewCNN_Rib,
    "dualviewcnn_verse19": DualViewCNN_Verse19,
    "dualviewcnn_verse20": DualViewCNN_Verse20,
    "dualviewcnn_ctpelvic": DualViewCNN_CTPelvic,
    # Multi-scale dual-view CNN configs
    "multiscale_dualviewcnn_femur": MultiScaleDualViewCNN_Femur,
    "multiscale_dualviewcnn_hip": MultiScaleDualViewCNN_Hip,
    "multiscale_dualviewcnn_rib": MultiScaleDualViewCNN_Rib,
    "multiscale_dualviewcnn_verse19": MultiScaleDualViewCNN_Verse19,
    "multiscale_dualviewcnn_verse20": MultiScaleDualViewCNN_Verse20,
    "multiscale_dualviewcnn_ctpelvic": MultiScaleDualViewCNN_CTPelvic,
}


# ============================================================================
# Fair-benchmark configurations (added 16 Nov 2026)
# ----------------------------------------------------------------------------
# Same as the configs above except for the fixes found in the post-hoc audit:
#   * FLIP_AP_LR: the AP PNGs are mirrored relative to the volume's W axis.
#   * VIEW_GEOMETRY="correct": lateral image columns go to D (anterior-posterior)
#     and are copied along W, for DualViewCNN and SwinX2S (MultiScale already
#     did this; it has no geometry switch).
#   * the same training budget for all models (epochs, batch size, patience),
#   * normalization that works at small batch sizes (DualViewCNN: GroupNorm,
#     SwinX2S: InstanceNorm as in the official code),
#   * SwinX2S: Tiny on every dataset, official averaging of the two view outputs, standard Swin
#     window attention with drop path (SWIN_IMPL="official"; the old vendored code scrambled windows),
#   * CTPelvic: the deduplicated split, best weights only (no optimizer state).
# Learning rates are the old defaults; tune them on validation only (--lr).
# Names: fair_<model>_<dataset>, e.g. fair_swinx2s_hip.
# ============================================================================
FAIR_COMMON = dict(
    FLIP_AP_LR=True,
    VIEW_GEOMETRY="correct",
    NUM_EPOCHS=200,
    PATIENCE=40,
    BATCH_SIZE=2,
    SAVE_INTERVAL=0,
    SAVE_OPTIMIZER_STATE=False,
)
FAIR_MODEL_SETTINGS = {
    "dualviewcnn": dict(DVC_NORM="group"),
    "multiscale_dualviewcnn": dict(),
    "swinx2s": dict(MODEL_SIZE="tiny", FEATURE_SIZE=32, DEPTHS=(2, 2, 6, 2), NUM_HEADS=(1, 2, 4, 8),
                    WINDOW_SIZE=(7, 7), NORM_NAME="instance", SWIN_OUTPUT_MODE="average",
                    SWIN_IMPL="official", WARMUP_EPOCHS=20),
}
for _ds in ("femur", "hip", "rib", "verse19", "verse20", "ctpelvic"):
    _grid = CONFIG_REGISTRY[f"multiscale_dualviewcnn_{_ds}"].VOLUME_SIZE
    for _model, _settings in FAIR_MODEL_SETTINGS.items():
        _base = CONFIG_REGISTRY[f"{_model}_{_ds}"]
        _attrs = dict(FAIR_COMMON, **_settings)
        if _model == "swinx2s":
            _attrs["IMG_SIZE"] = list(_grid)          # same output grid as the CNNs
        if _ds == "ctpelvic":
            _attrs["SPLIT_MANIFEST"] = os.path.join(BaseConfig.PROJECT_ROOT, "splits",
                                                    "ctpelvic_seed42_dedup.json")
        _short = "multiscale" if _model == "multiscale_dualviewcnn" else _model
        CONFIG_REGISTRY[f"fair_{_short}_{_ds}"] = type(f"Fair_{_base.__name__}", (_base,), _attrs)


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
    
    print("\n🔹 DualViewCNN Configs:")
    for name in CONFIG_REGISTRY:
        if name.startswith("dualviewcnn"):
            print(f"  - {name}")

    print("\n🔹 MultiScaleDualViewCNN Configs:")
    for name in CONFIG_REGISTRY:
        if name.startswith("multiscale_dualviewcnn"):
            print(f"  - {name}")

    print("\n🔹 Fair-benchmark Configs (geometry + normalization fixes):")
    for name in CONFIG_REGISTRY:
        if name.startswith("fair_"):
            print(f"  - {name}")

    print("\n" + "="*70)
    print("Usage: python train.py --config <config_name>")
    print("="*70 + "\n")
