"""
Inference Script for SwinX2S and TransVert Models
Accepts two X-ray images (sagittal and coronal views) and generates 3D reconstruction

DEFAULT USAGE (edit these paths):
    MODEL = 'swinx2s'  or 'transvert'
    CHECKPOINT = './checkpoints/swinx2s_best.pth'
    IMAGE_A = './data/femur/sample1_sagittal.png'
    IMAGE_B = './data/femur/sample1_coronal.png'
    SAVE_OUTPUT = './output/reconstruction.npy'  (optional, None to skip saving)
    THRESHOLD = 0.5
    NO_VIS = False

Then run: python inference.py
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from configs.config import Config
from models.swinx2s import SwinX2S
from models.transvert_model import TransVert


# ============================================
# DEFAULT VALUES - EDIT THESE BEFORE RUNNING
# ============================================
DEFAULT_MODEL = 'swinx2s'  # 'swinx2s' or 'transvert'
DEFAULT_CHECKPOINT = './checkpoints/swinx2s_best.pth'
DEFAULT_IMAGE_A = './data/femur/sample1_sagittal.png'
DEFAULT_IMAGE_B = './data/femur/sample1_coronal.png'
DEFAULT_SAVE_OUTPUT = None  # './output/reconstruction.npy' or None
DEFAULT_THRESHOLD = 0.5
DEFAULT_NO_VIS = False
# ============================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference: Generate 3D reconstruction from two X-ray views',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=['swinx2s', 'transvert'],
                        help='Model architecture to use')
    
    # Input images
    parser.add_argument('--image-a', type=str, default=DEFAULT_IMAGE_A,
                        help='Path to first X-ray image (sagittal view)')
    parser.add_argument('--image-b', type=str, default=DEFAULT_IMAGE_B,
                        help='Path to second X-ray image (coronal view)')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to model checkpoint (.pth file)')
    
    # Output options
    parser.add_argument('--save-output', type=str, default=DEFAULT_SAVE_OUTPUT,
                        help='Path to save output volume (.npy file)')
    parser.add_argument('--no-vis', action='store_true', default=DEFAULT_NO_VIS,
                        help='Disable 3D visualization')
    
    # Visualization arguments
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Threshold for 3D visualization (0-1)')
    
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


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess a single X-ray image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (H, W)
    
    Returns:
        Preprocessed image tensor [1, 1, H, W]
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to target size
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_array)[None, None, ...]
    
    return img_tensor


def plot_3d_volume(volume, threshold=0.5, title="3D Reconstruction"):
    """
    Visualize 3D volume using Plotly isosurface.
    
    Args:
        volume: 3D numpy array [D, H, W]
        threshold: Threshold for isosurface
        title: Plot title
    """
    # Normalize to [0,1]
    vol = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    D, H, W = vol.shape
    
    # Generate coordinates
    x = np.arange(D).repeat(H * W)
    y = np.tile(np.arange(H).repeat(W), D)
    z = np.tile(np.arange(W), D * H)
    values = vol.flatten()
    
    # Create figure
    fig = go.Figure(data=go.Isosurface(
        x=x, y=y, z=z, value=values,
        isomin=threshold, isomax=1.0, surface_count=1,
        colorscale="Viridis", opacity=0.8,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="Depth",
            yaxis_title="Height",
            zaxis_title="Width"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700
    )
    
    fig.show()


def plot_input_and_output(img_a, img_b, output_volume, threshold=0.5):
    """
    Show input images and output 3D reconstruction side by side.
    
    Args:
        img_a: First input image [H, W]
        img_b: Second input image [H, W]
        output_volume: Output 3D volume [D, H, W]
        threshold: Threshold for 3D visualization
    """
    # Normalize volume
    vol = (output_volume - output_volume.min()) / (output_volume.max() - output_volume.min() + 1e-8)
    D, H, W = vol.shape
    
    # Generate 3D coordinates
    x = np.arange(D).repeat(H * W)
    y = np.tile(np.arange(H).repeat(W), D)
    z = np.tile(np.arange(W), D * H)
    values = vol.flatten()
    
    # Create subplots: 2 images + 1 3D volume
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Input: View A (Sagittal)", "Input: View B (Coronal)", "Output: 3D Reconstruction"),
        column_widths=[0.3, 0.3, 0.4],
        horizontal_spacing=0.05
    )
    
    # Add first input image
    fig.add_trace(
        go.Heatmap(z=img_a, colorscale='gray', showscale=False),
        row=1, col=1
    )
    
    # Add second input image
    fig.add_trace(
        go.Heatmap(z=img_b, colorscale='gray', showscale=False),
        row=1, col=2
    )
    
    # Add 3D reconstruction
    fig.add_trace(
        go.Isosurface(
            x=x, y=y, z=z, value=values,
            isomin=threshold, isomax=1.0, surface_count=1,
            colorscale="Viridis", opacity=0.8,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    
    fig.update_layout(
        title="3D Reconstruction from Two X-ray Views",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=500
    )
    
    fig.show()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(Config.DEVICE)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"🔮 INFERENCE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:          {args.model.upper()}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Input Image A:  {args.image_a}")
    print(f"Input Image B:  {args.image_b}")
    print(f"Device:         {device.type.upper()}")
    print(f"Visualization:  {'✅ Enabled' if not args.no_vis else '❌ Disabled'}")
    if args.save_output:
        print(f"Save Output:    {args.save_output}")
    print(f"Threshold:      {args.threshold}")
    print(f"{'='*60}\n")
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint '{args.checkpoint}' does not exist!")
        print(f"   Please edit DEFAULT_CHECKPOINT at the top of this file")
        return
    
    if not os.path.exists(args.image_a):
        print(f"❌ Error: Image A '{args.image_a}' does not exist!")
        print(f"   Please edit DEFAULT_IMAGE_A at the top of this file")
        return
    
    if not os.path.exists(args.image_b):
        print(f"❌ Error: Image B '{args.image_b}' does not exist!")
        print(f"   Please edit DEFAULT_IMAGE_B at the top of this file")
        return
    
    # Load and preprocess images
    print("📸 Loading input images...")
    try:
        target_size = (Config.IMG_SIZE[1], Config.IMG_SIZE[2])  # (H, W)
        img_a_tensor = load_and_preprocess_image(args.image_a, target_size)
        img_b_tensor = load_and_preprocess_image(args.image_b, target_size)
        print(f"   Image A shape: {img_a_tensor.shape}")
        print(f"   Image B shape: {img_b_tensor.shape}\n")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return
    
    # Move to device
    img_a_tensor = img_a_tensor.to(device)
    img_b_tensor = img_b_tensor.to(device)
    
    # Create model
    model = create_model(args.model, device)
    
    # Load checkpoint
    print(f"🔍 Loading checkpoint...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"   Loaded from epoch: {epoch}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Model loaded successfully\n")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Run inference
    print("🚀 Running inference...")
    with torch.no_grad():
        # Use autocast only if on CUDA
        if device.type == 'cuda':
            with torch.amp.autocast(device.type):
                output = model(img_a_tensor, img_b_tensor)
        else:
            output = model(img_a_tensor, img_b_tensor)
    
    # Get output volume
    output_probs = torch.sigmoid(output[0, 0]).detach().cpu().numpy()
    print(f"✅ Inference complete!")
    print(f"   Output shape: {output_probs.shape}\n")
    
    # Save output if requested
    if args.save_output:
        output_dir = os.path.dirname(args.save_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        np.save(args.save_output, output_probs)
        print(f"💾 Output saved to: {args.save_output}\n")
    
    # Visualize
    if not args.no_vis:
        print("📊 Generating visualization...")
        
        # Get input images as numpy arrays
        img_a_np = img_a_tensor[0, 0].cpu().numpy()
        img_b_np = img_b_tensor[0, 0].cpu().numpy()
        
        # Show inputs and output together
        plot_input_and_output(
            img_a_np, 
            img_b_np, 
            output_probs, 
            threshold=args.threshold
        )
        
        print("✅ Visualization complete!\n")
    
    # Print statistics
    print(f"{'='*60}")
    print(f"📊 OUTPUT STATISTICS")
    print(f"{'='*60}")
    print(f"Min value:      {output_probs.min():.4f}")
    print(f"Max value:      {output_probs.max():.4f}")
    print(f"Mean value:     {output_probs.mean():.4f}")
    print(f"Std value:      {output_probs.std():.4f}")
    print(f"Shape:          {output_probs.shape}")
    print(f"{'='*60}\n")
    
    print("🎉 Done!\n")


if __name__ == "__main__":
    main()