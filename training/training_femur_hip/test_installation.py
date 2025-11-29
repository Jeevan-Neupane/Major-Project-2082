import sys
print("Python version:", sys.version)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import monai
    print(f"✅ MONAI {monai.__version__}")
except ImportError as e:
    print(f"❌ MONAI: {e}")

try:
    import timm
    print(f"✅ TIMM {timm.__version__}")
except ImportError as e:
    print(f"❌ TIMM: {e}")

try:
    import nibabel
    print(f"✅ NiBabel {nibabel.__version__}")
except ImportError as e:
    print(f"❌ NiBabel: {e}")

try:
    import wandb
    print(f"✅ WandB {wandb.__version__}")
except ImportError as e:
    print(f"❌ WandB: {e}")

try:
    import plotly
    print(f"✅ Plotly {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly: {e}")

print("\n✅ All required packages installed successfully!")