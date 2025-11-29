"""Test model import specifically"""
import sys
import traceback

try:
    print("Testing model import...")
    from models.swinx2s import SwinX2S
    print("✅ Model imported successfully!")
except Exception as e:
    print(f"❌ Error importing model: {e}")
    traceback.print_exc()