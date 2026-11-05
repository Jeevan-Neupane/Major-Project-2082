"""Multi-scale dual-view CNN for biplanar X-ray to 3D mask reconstruction.

This is a clean-room implementation of the *multi-scale 2D-to-3D fusion*
family used in the NAAMII biplanar reconstruction benchmark.  AP and lateral
views are encoded independently at four resolutions.  At every decoder scale,
their 2D feature maps are lifted into complementary 3D slabs and fused with
the upsampled 3D representation from the preceding scale.

It predicts a binary segmentation logit volume, rather than CT intensities;
therefore it is deliberately named MultiScaleDualViewCNN, not X2CT-GAN.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides *channels*."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct2D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class ConvNormAct3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class Residual2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct2D(in_channels, out_channels, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class MultiScaleEncoder2D(nn.Module):
    """Independent 2D pyramid encoder used for each radiographic view."""
    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        blocks = []
        previous = in_channels
        for index, current in enumerate(channels):
            blocks.append(Residual2DBlock(previous, current, stride=1 if index == 0 else 2))
            previous = current
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class MultiScaleDualViewCNN(nn.Module):
    """Fuse AP and lateral X-rays at every decoder resolution.

    Args:
        output_size: target ``(depth, height, width)`` segmentation volume.
        base_channels: width of the finest 2D/3D decoder stage.
        num_scales: encoder-decoder pyramid depth.  Four scales support the
            repository's 96x96x128 and 128x128x160 output presets.
    """
    def __init__(
        self,
        output_size: Sequence[int],
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        num_scales: int = 4,
    ):
        super().__init__()
        if len(output_size) != 3:
            raise ValueError("output_size must be (depth, height, width)")
        if num_scales < 2:
            raise ValueError("num_scales must be at least two")

        self.output_size = tuple(int(v) for v in output_size)
        self.num_scales = num_scales
        encoder_channels = [base_channels * (2 ** i) for i in range(num_scales)]
        decoder_channels = list(reversed(encoder_channels))

        self.ap_encoder = MultiScaleEncoder2D(in_channels, encoder_channels)
        self.lat_encoder = MultiScaleEncoder2D(in_channels, encoder_channels)

        # Coarse-to-fine 2D decoders.  Both views retain their own parameters.
        self.ap_decoders = nn.ModuleList()
        self.lat_decoders = nn.ModuleList()
        for index, channels in enumerate(decoder_channels):
            skip_channels = decoder_channels[index]
            previous_channels = 0 if index == 0 else decoder_channels[index - 1]
            self.ap_decoders.append(Residual2DBlock(skip_channels + previous_channels, channels))
            self.lat_decoders.append(Residual2DBlock(skip_channels + previous_channels, channels))

        # Two lifted view features plus the preceding 3D decoder feature.
        self.fusion_blocks = nn.ModuleList()
        for index, channels in enumerate(decoder_channels):
            previous_channels = 0 if index == 0 else decoder_channels[index - 1]
            self.fusion_blocks.append(Residual3DBlock(2 * channels + previous_channels, channels))
        self.head = nn.Conv3d(decoder_channels[-1], out_channels, kernel_size=1)

    @staticmethod
    def _lift_ap(feature: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        """Lift AP features across the unobserved depth axis."""
        depth, height, width = size
        feature = F.interpolate(feature, size=(height, width), mode="bilinear", align_corners=False)
        return feature.unsqueeze(2).expand(-1, -1, depth, -1, -1)

    @staticmethod
    def _lift_lateral(feature: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        """Lift lateral features across width, mapping its horizontal axis to depth."""
        depth, height, width = size
        # [B,C,H,W] -> [B,C,H,W,1]; interpolate W into volume depth then
        # permute to the common [B,C,D,H,W] convention.
        feature = feature.unsqueeze(-1)
        feature = F.interpolate(feature, size=(height, depth, width), mode="trilinear", align_corners=False)
        return feature.permute(0, 1, 3, 2, 4).contiguous()

    def _scale_size(self, scale_from_coarsest: int) -> tuple[int, int, int]:
        divisor = 2 ** (self.num_scales - 1 - scale_from_coarsest)
        return tuple(max(1, value // divisor) for value in self.output_size)

    def forward(self, ap_image: torch.Tensor, lateral_image: torch.Tensor) -> torch.Tensor:
        if ap_image.shape != lateral_image.shape:
            raise ValueError("AP and lateral inputs must have the same tensor shape")
        ap_features = list(reversed(self.ap_encoder(ap_image)))
        lat_features = list(reversed(self.lat_encoder(lateral_image)))

        ap_previous = lat_previous = fused_previous = None
        for index, (ap_skip, lat_skip) in enumerate(zip(ap_features, lat_features)):
            if index == 0:
                ap_decoded = self.ap_decoders[index](ap_skip)
                lat_decoded = self.lat_decoders[index](lat_skip)
            else:
                ap_up = F.interpolate(ap_previous, size=ap_skip.shape[-2:], mode="bilinear", align_corners=False)
                lat_up = F.interpolate(lat_previous, size=lat_skip.shape[-2:], mode="bilinear", align_corners=False)
                ap_decoded = self.ap_decoders[index](torch.cat((ap_skip, ap_up), dim=1))
                lat_decoded = self.lat_decoders[index](torch.cat((lat_skip, lat_up), dim=1))

            scale_size = self._scale_size(index)
            fused_inputs = [self._lift_ap(ap_decoded, scale_size), self._lift_lateral(lat_decoded, scale_size)]
            if fused_previous is not None:
                fused_inputs.insert(0, F.interpolate(fused_previous, size=scale_size, mode="trilinear", align_corners=False))
            fused_previous = self.fusion_blocks[index](torch.cat(fused_inputs, dim=1))
            ap_previous, lat_previous = ap_decoded, lat_decoded

        logits = self.head(fused_previous)
        if logits.shape[-3:] != self.output_size:
            logits = F.interpolate(logits, size=self.output_size, mode="trilinear", align_corners=False)
        return logits

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = ["MultiScaleDualViewCNN"]
