import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UnetrUpBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 kernel_size: int, upsample_kernel_size: int, norm_name: str, res_block: bool = False):
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims, in_channels, out_channels,
            kernel_size=upsample_kernel_size, stride=upsample_kernel_size,
            conv_only=True, is_transposed=True
        )
        if res_block:
            self.conv_block = UnetResBlock(spatial_dims, out_channels + out_channels, out_channels,
                                           kernel_size=kernel_size, stride=1, norm_name=norm_name)
        else:
            self.conv_block = UnetBasicBlock(spatial_dims, out_channels + out_channels, out_channels,
                                             kernel_size=kernel_size, stride=1, norm_name=norm_name)
    
    def forward(self, x, skip):
        x = self.transp_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class UnetrBasicBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, norm_name: str, res_block: bool = False):
        super().__init__()
        if res_block:
            self.layer = UnetResBlock(spatial_dims, in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride, norm_name=norm_name)
        else:
            self.layer = UnetBasicBlock(spatial_dims, in_channels, out_channels,
                                        kernel_size=kernel_size, stride=stride, norm_name=norm_name)
    
    def forward(self, x):
        return self.layer(x)

class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float = None):
        super().__init__()
        self.conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=1,
                                   dropout=dropout, bias=True, act=None, norm=None, conv_only=False)
    
    def forward(self, x):
        return self.conv(x)