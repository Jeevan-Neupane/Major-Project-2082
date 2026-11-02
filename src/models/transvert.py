"""
TransVert: CNN-based 3D Bone Reconstruction from Two Orthogonal X-ray Views.

Architecture (matching the paper diagram):
  1. Two shared-architecture encoders each produce a thin 3D feature slab
     [B, C, 1, H/4, W/4] from a single X-ray view.
  2. Map-and-Fuse block — the key contribution:
       AP   path  →  expand along **depth**  (dim 2):  [B,C,  1,H',W'] → [B,C,D,H',W']
       Lat  path  →  expand along **width**  (dim 4):  [B,C,H',W',  1] → [B,C,H',W',D]
                      then permuted to                  [B,C,D,H',W']
     Each view fills in information along the axis that was collapsed
     during X-ray projection.
  3. Channel-wise concatenation → [B, 2C, D, H', W']
  4. 3D decoder with two 2× upsampling stages → [B, 1, 4D, 4H', 4W']
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Encoder  (2D → thin 3D slab)
# ---------------------------

class XRayEncoder(nn.Module):
    """
    Encode a single-channel X-ray into a thin 3D feature slab.

    Input:  [B, C_in, H, W]  (2D X-ray)
    Output: [B, C_out, 1, H//4, W//4]

    Two stride-2 3D convolutions downsample H and W by 4× while
    keeping the depth dimension at 1.
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16,
                      kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels,
                      kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # [B, C, H, W] → [B, C, 1, H, W]
        x = x.unsqueeze(2)
        return self.encoder(x)  # [B, out_ch, 1, H//4, W//4]


# Legacy aliases (the two encoders have identical architecture)
EncoderSagittal = XRayEncoder
EncoderCoronal = XRayEncoder


# ---------------------------
# Volumetric Expander
# ---------------------------

class VolumetricExpander(nn.Module):
    """
    Progressively double a thin spatial axis until it reaches *target_depth*.

    Args:
        target_depth: desired size of the expanded axis.
        channels: number of feature channels (unchanged).
        expand_dim: which axis to expand —
            ``'depth'`` uses ConvTranspose3d with kernel/stride ``(2,1,1)``
            ``'width'`` uses ConvTranspose3d with kernel/stride ``(1,1,2)``
    """
    def __init__(self, target_depth=16, channels=32, expand_dim='depth'):
        super().__init__()
        self.target_depth = target_depth
        self.expand_dim = expand_dim

        # Number of doubling steps: 1→2→4→…→target_depth
        num_steps = 0
        current = 1
        while current < target_depth:
            current *= 2
            num_steps += 1

        if expand_dim == 'depth':
            k, s = (2, 1, 1), (2, 1, 1)
        elif expand_dim == 'width':
            k, s = (1, 1, 2), (1, 1, 2)
        else:
            raise ValueError(f"expand_dim must be 'depth' or 'width', got {expand_dim}")

        layers = []
        for _ in range(num_steps):
            layers += [
                nn.ConvTranspose3d(channels, channels, kernel_size=k, stride=s),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
            ]
        self.expand = nn.Sequential(*layers)

        # Post-expansion refinement (isotropic 3×3×3 convolutions)
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.expand(x)

        # Crop to exact target if doubling overshot (e.g. target=24 rounds up to 32)
        dim = 2 if self.expand_dim == 'depth' else 4
        if x.shape[dim] > self.target_depth:
            extra = x.shape[dim] - self.target_depth
            start = extra // 2
            slices = [slice(None)] * 5
            slices[dim] = slice(start, start + self.target_depth)
            x = x[tuple(slices)]

        return self.refine(x)


# ---------------------------
# Feature Fusion
# ---------------------------

class FeatureFusion(nn.Module):
    """Channel-wise concatenation of two 3D feature volumes."""
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.cat([a, b], dim=1)


# ---------------------------
# 3D Decoder
# ---------------------------

class Volume3DDecoder(nn.Module):
    """
    3D decoder that progressively upsamples fused features to the target
    volume resolution.

    Two ConvTranspose3d stages (each 2× in all dims) followed by a
    refinement convolution and a 1×1×1 output head.

    Input:  [B, in_channels, D, H, W]
    Output: [B, out_channels, 4D, 4H, 4W]
    """
    def __init__(self, in_channels=64, out_channels=1):
        super().__init__()
        mid = in_channels // 2
        self.decoder = nn.Sequential(
            # Stage 1:  2× upsample in all spatial dims
            nn.ConvTranspose3d(in_channels, in_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),

            # Stage 2:  2× upsample in all spatial dims
            nn.ConvTranspose3d(in_channels, mid,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(mid),

            # Refinement
            nn.Conv3d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(mid),

            # Output head
            nn.Conv3d(mid, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


# ---------------------------
# TransVert Main Model
# ---------------------------

class TransVert(nn.Module):
    """
    TransVert: 3D bone reconstruction from two orthogonal X-ray views.

    Key insight from the paper diagram:
        The two views are expanded along **orthogonal** spatial axes
        before fusion so that each X-ray contributes information along
        the axis that was collapsed during imaging.

    Data flow::

        AP  view  ─── Encoder ──► [B,C, 1,H',W']
                                       │  expand depth
                                       ▼
                                  [B,C, D,H',W'] ──┐
                                                    ├── concat ── Decoder ── output
                                  [B,C, D,H',W'] ──┘
                                       ▲  permute back
                                       │  expand width
        Lat view  ─── Encoder ──► [B,C, 1,H',W'] → permute → [B,C,H',W', 1]

    Args:
        in_channels: input channels (1 for grayscale X-ray).
        mid_channels: encoder feature dimension.
        out_channels: output channels (1 for binary segmentation).
        intermediate_depth: target size for the expanded axis.
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

        # Shared-architecture encoders for two orthogonal views
        self.sagittal_encoder = XRayEncoder(in_channels, mid_channels)
        self.coronal_encoder  = XRayEncoder(in_channels, mid_channels)

        # --- Map-and-Fuse: orthogonal expanders ---
        self.sagittal_expander = VolumetricExpander(
            target_depth=intermediate_depth,
            channels=mid_channels,
            expand_dim='depth',     # AP  → expand along depth  (dim 2)
        )
        self.coronal_expander = VolumetricExpander(
            target_depth=intermediate_depth,
            channels=mid_channels,
            expand_dim='width',     # Lat → expand along width  (dim 4)
        )

        # Concatenation fusion
        self.fusion = FeatureFusion()

        # 3D decoder (input channels = 2 × mid_channels after concat)
        self.decoder = Volume3DDecoder(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
        )

    def forward(self, sagittal_input, coronal_input):
        """
        Args:
            sagittal_input: AP X-ray      [B, C, H, W]  (any H, W)
            coronal_input:  Lateral X-ray [B, C, H, W]

        Returns:
            3D volume prediction [B, out_channels, D_target, H, W]
            where D_target = intermediate_depth × 4, and H/W always
            match the input exactly regardless of divisibility.
        """
        # Record target output size before any downsampling
        H, W = sagittal_input.shape[-2], sagittal_input.shape[-1]
        D_target = self.intermediate_depth * 4

        # ── Encode both views ──
        s_enc = self.sagittal_encoder(sagittal_input)       # [B, C, 1, H', W']
        c_enc = self.coronal_encoder(coronal_input)         # [B, C, 1, H', W']

        # ── Map-and-Fuse: expand along orthogonal axes ──

        # AP path: expand depth (dim 2)  — thin dim is already at dim 2
        #   [B, C, 1, H', W']  →  [B, C, D, H', W']
        s_vol = self.sagittal_expander(s_enc)

        # Lat path: move the thin dim from depth (dim 2) to width (dim 4),
        # expand along width, then permute back.
        #   [B, C, 1, H', W']  →  permute  →  [B, C, H', W', 1]
        #                       →  expand   →  [B, C, H', W', D]
        #                       →  permute  →  [B, C, D, H', W']
        c_enc = c_enc.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H', W', 1]
        c_vol = self.coronal_expander(c_enc)                # [B, C, H', W', D]
        c_vol = c_vol.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, D, H', W']

        # ── Fusion + Decode ──
        fused  = self.fusion(s_vol, c_vol)                  # [B, 2C, D, H', W']
        output = self.decoder(fused)                        # [B, 1, ~4D, ~4H', ~4W']

        # ── Snap to exact target size ──
        # If H/W are not divisible by 4, the encoder's floor-division and the
        # decoder's fixed 4× upsampling leave a 1-pixel discrepancy.
        # A single trilinear resize brings output to exact (D_target, H, W).
        if output.shape[-3:] != (D_target, H, W):
            output = F.interpolate(
                output, size=(D_target, H, W),
                mode="trilinear", align_corners=False,
            )

        return output

    def get_num_params(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (
            f"TransVert(in_channels={self.in_channels}, "
            f"mid_channels={self.mid_channels}, "
            f"out_channels={self.out_channels}, "
            f"intermediate_depth={self.intermediate_depth})"
        )


# Convenience exports
__all__ = ["TransVert"]