import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config

def main():
    """Display all configuration values."""
    print(f"\n{'='*70}")
    print(f"📋 CURRENT CONFIGURATION (Default Values)")
    print(f"{'='*70}\n")
    
    print("📁 Data Paths:")
    print(f"   DATA_ROOT:     {Config.DATA_ROOT}")
    print(f"   SAVE_DIR:      {Config.SAVE_DIR}")
    
    print("\n🏗️  Model Architecture:")
    print(f"   IMG_SIZE:      {Config.IMG_SIZE}")
    print(f"   IN_CHANNELS:   {Config.IN_CHANNELS}")
    print(f"   OUT_CHANNELS:  {Config.OUT_CHANNELS}")
    print(f"   FEATURE_SIZE:  {Config.FEATURE_SIZE}")
    print(f"   DEPTHS:        {Config.DEPTHS}")
    print(f"   NUM_HEADS:     {Config.NUM_HEADS}")
    print(f"   WINDOW_SIZE:   {Config.WINDOW_SIZE}")
    print(f"   NORM_NAME:     {Config.NORM_NAME}")
    
    print("\n🎓 Training Parameters:")
    print(f"   BATCH_SIZE:    {Config.BATCH_SIZE}")
    print(f"   NUM_EPOCHS:    {Config.NUM_EPOCHS}")
    print(f"   LEARNING_RATE: {Config.LEARNING_RATE}")
    print(f"   GRAD_CLIP:     {Config.GRAD_CLIP}")
    print(f"   PATIENCE:      {Config.PATIENCE}")
    
    print("\n⚖️  Loss Configuration:")
    print(f"   BCE_WEIGHT:    {Config.BCE_WEIGHT}")
    print(f"   DICE_WEIGHT:   {Config.DICE_WEIGHT}")
    
    print("\n📊 Data Split:")
    print(f"   TRAIN_RATIO:   {Config.TRAIN_RATIO} ({Config.TRAIN_RATIO*100:.0f}%)")
    print(f"   VAL_RATIO:     {Config.VAL_RATIO} ({Config.VAL_RATIO*100:.0f}%)")
    print(f"   TEST_RATIO:    {1-Config.TRAIN_RATIO-Config.VAL_RATIO:.2f} ({(1-Config.TRAIN_RATIO-Config.VAL_RATIO)*100:.0f}%)")
    print(f"   RANDOM_SEED:   {Config.RANDOM_SEED}")
    
    print("\n💻 Device & Performance:")
    print(f"   DEVICE:        {Config.DEVICE}")
    print(f"   USE_AMP:       {'✅ Enabled' if Config.USE_AMP else '❌ Disabled'}")
    
    print("\n📝 Logging:")
    print(f"   USE_WANDB:     {'✅ Enabled' if Config.USE_WANDB else '❌ Disabled'}")
    print(f"   WANDB_PROJECT: {Config.WANDB_PROJECT}")
    
    print("\n🎯 Other:")
    print(f"   TARGET_KEY:    {Config.TARGET_KEY}")
    print(f"   BINARY_THRESH: {Config.BINARY_THRESH}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()