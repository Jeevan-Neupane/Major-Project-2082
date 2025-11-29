"""
TransVert: Transformer-based Vertebrae Reconstruction from Two Orthogonal X-ray Views
Complete model in single file with all components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Encoders (2D → thin 3D)
# ---------------------------

class EncoderSagittal(nn.Module):
    """
    Encoder for Sagittal view (Width x Height, expanded to include minimal depth dimension).
    Input:  [B, C, H, W] (2D sagittal view)
    Output: [B, out_channels, 1, H//4, W//4]
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> add depth dim -> [B, C, 1, H, W]
        x = x.unsqueeze(2)
        x = self.encoder(x)
        return x  # [B, out_channels, 1, H//4, W//4]


class EncoderCoronal(nn.Module):
    """
    Encoder for Coronal view (Depth x Height, expanded to include minimal width dimension).
    Input:  [B, C, H, W] (2D coronal view)
    Output: [B, out_channels, 1, H//4, W//4]
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> add depth dim -> [B, C, 1, H, W]
        x = x.unsqueeze(2)
        x = self.encoder(x)
        return x  # [B, out_channels, 1, H//4, W//4]


# ---------------------------
# Volumetric Expander
# ---------------------------

class VolumetricExpander(nn.Module):
    """
    Expands thin 3D features to full 3D volume.
    Input:  [B, C, 1, H, W]
    Output: [B, C, target_depth, H, W]
    """
    def __init__(self, target_depth=16, channels=32):
        super().__init__()
        self.target_depth = target_depth
        
        # Calculate number of upsampling steps needed
        num_steps = 0
        current_depth = 1
        while current_depth < target_depth:
            current_depth *= 2
            num_steps += 1
        
        # Build progressive upsampling layers
        layers = []
        for i in range(num_steps):
            layers.append(nn.ConvTranspose3d(channels, channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)))
            layers.append(nn.ReLU(inplace=True))
        
        self.expand = nn.Sequential(*layers)
        
        # Refinement after expansion
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Expand depth dimension
        x = self.expand(x)
        
        # Crop to exact target depth if needed
        if x.shape[2] > self.target_depth:
            extra = x.shape[2] - self.target_depth
            start = extra // 2
            x = x[:, :, start:start + self.target_depth, :, :]
        
        # Refine features
        x = self.refine(x)
        
        return x  # [B, channels, target_depth, H, W]


# ---------------------------
# Feature Fusion
# ---------------------------

class FeatureFusion(nn.Module):
    """
    Simple concatenation-based fusion of two feature volumes.
    Input:  depth_expanded [B, C, D, H, W], width_expanded [B, C, D, H, W]
    Output: fused [B, 2*C, D, H, W]
    """
    def __init__(self):
        super().__init__()

    def forward(self, depth_expanded, width_expanded):
        """
        Args:
            depth_expanded: [B, C, D, H, W]
            width_expanded: [B, C, D, H, W]
        Returns:
            Concatenated features [B, 2*C, D, H, W]
        """
        return torch.cat([depth_expanded, width_expanded], dim=1)


# ---------------------------
# 3D Decoder
# ---------------------------

class Volume3DDecoder(nn.Module):
    """
    3D decoder that upsamples fused features to final output volume.
    Uses anisotropic convolutions for controlled expansion.
    Input:  [B, in_channels, D, H, W]
    Output: [B, out_channels, D_out, H_out, W_out]
    """
    def __init__(self, in_channels=64, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            # First upsampling stage: 2x in each spatial dimension
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),

            # Second upsampling stage: 2x in each spatial dimension
            nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels//2),

            # Final refinement while maintaining dimensions
            nn.Conv3d(in_channels//2, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels//2),

            # Final output layer
            nn.Conv3d(in_channels//2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: Fused features [B, in_channels, D, H, W]
        Returns:
            Decoded 3D volume [B, out_channels, D_out, H_out, W_out]
        """
        x = self.decoder(x)
        return x


# ---------------------------
# TransVert Main Model
# ---------------------------

class TransVert(nn.Module):
    """
    TransVert model for 3D vertebrae reconstruction from sagittal and coronal X-ray views.
    
    Architecture:
    1. Encode sagittal and coronal views independently
    2. Expand both to 3D volumes
    3. Fuse expanded features
    4. Decode to final 3D volume
    
    Args:
        in_channels: Number of input channels (1 for grayscale X-rays)
        mid_channels: Feature dimension for encoders
        out_channels: Number of output channels (1 for binary segmentation)
        intermediate_depth: Target depth for 3D volume
    """
    
    def __init__(
        self,
        in_channels=1,
        mid_channels=32,
        out_channels=1,
        intermediate_depth=16,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.intermediate_depth = intermediate_depth
        
        # Encoders for two orthogonal views
        self.sagittal_encoder = EncoderSagittal(
            in_channels=in_channels,
            out_channels=mid_channels
        )
        self.coronal_encoder = EncoderCoronal(
            in_channels=in_channels,
            out_channels=mid_channels
        )
        
        # Expanders for creating 3D volumes (both use same expander now)
        self.sagittal_expander = VolumetricExpander(
            target_depth=intermediate_depth,
            channels=mid_channels
        )
        self.coronal_expander = VolumetricExpander(
            target_depth=intermediate_depth,
            channels=mid_channels
        )
        
        # Fusion module
        self.fusion = FeatureFusion()
        
        # Decoder to final 3D volume
        self.decoder = Volume3DDecoder(
            in_channels=mid_channels * 2,  # After concatenation
            out_channels=out_channels
        )
    
    def forward(self, sagittal_input, coronal_input):
        """
        Forward pass with two orthogonal X-ray views.
        
        Args:
            sagittal_input: Sagittal X-ray view [B, C, H, W]
            coronal_input: Coronal X-ray view [B, C, H, W]
            
        Returns:
            3D reconstruction [B, out_channels, D, H, W]
        """
        # Encode both views
        s_enc = self.sagittal_encoder(sagittal_input)   # [B, mid_C, 1, H', W']
        c_enc = self.coronal_encoder(coronal_input)     # [B, mid_C, 1, H', W']
        
        # Expand both to 3D volumes
        s_vol = self.sagittal_expander(s_enc)           # [B, mid_C, D, H', W']
        c_vol = self.coronal_expander(c_enc)            # [B, mid_C, D, H', W']
        
        # Fuse features
        fused = self.fusion(s_vol, c_vol)               # [B, 2*mid_C, D, H', W']
        
        # Decode to final volume
        output = self.decoder(fused)                    # [B, out_C, D_out, H_out, W_out]
        
        return output
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self):
        return (f"TransVert(in_channels={self.in_channels}, "
                f"mid_channels={self.mid_channels}, "
                f"out_channels={self.out_channels}, "
                f"intermediate_depth={self.intermediate_depth})")


# Convenience exports
__all__ = ["TransVert"]