"""
Simple test script to verify input/output shapes for SwinX2S and TransVert models.
"""
import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.swinx2s import SwinX2S
from models.transvert_model import TransVert


def test_swinx2s():
    """Test SwinX2S model - Simple version."""
    print("="*60)
    print("🧪 TESTING SWINX2S MODEL")
    print("="*60)
    
    # Small configuration
    img_size = [64, 128, 128]  # [D, H, W]
    batch_size = 1
    
    print(f"\n📋 Configuration:")
    print(f"   Image size: {img_size} (D, H, W)")
    print(f"   Batch size: {batch_size}")
    
    try:
        # Create model
        model = SwinX2S(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(2, 4, 6, 8),
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ Model created successfully")
        print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Create dummy inputs
        D, H, W = img_size
        xa = torch.randn(batch_size, 1, H, W)
        xb = torch.randn(batch_size, 1, H, W)
        
        print(f"\n📥 Input:")
        print(f"   View A: {list(xa.shape)}")
        print(f"   View B: {list(xb.shape)}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(xa, xb)
        
        print(f"\n📤 Output:")
        print(f"   Volume: {list(output.shape)}")
        print(f"\n✅ Test passed!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()


def test_transvert():
    """Test TransVert model - Simple version."""
    print("="*60)
    print("🧪 TESTING TRANSVERT MODEL")
    print("="*60)
    
    # Small configuration
    H, W = 128, 128
    batch_size = 1
    
    print(f"\n📋 Configuration:")
    print(f"   Input size: ({H}, {W})")
    print(f"   Batch size: {batch_size}")
    
    try:
        # Create model
        model = TransVert(
            in_channels=1,
            mid_channels=32,
            out_channels=1,
            intermediate_depth=16,  # Remove intermediate_width
        )
        
        num_params = model.get_num_params()
        print(f"\n✅ Model created successfully")
        print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Create dummy inputs
        sagittal = torch.randn(batch_size, 1, H, W)
        coronal = torch.randn(batch_size, 1, H, W)
        
        print(f"\n📥 Input:")
        print(f"   Sagittal: {list(sagittal.shape)}")
        print(f"   Coronal:  {list(coronal.shape)}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(sagittal, coronal)
        
        print(f"\n📤 Output:")
        print(f"   Volume: {list(output.shape)}")
        print(f"\n✅ Test passed!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    """Run simple tests."""
    print("\n🔬 MODEL TESTING\n")
    
    # Test SwinX2S
    test_swinx2s()
    
    # Test TransVert
    test_transvert()
    
    print("="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()