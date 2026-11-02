import torch
import torch.nn as nn
from typing import MutableMapping, Sequence, Tuple, Union

from .unetr_blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from .swin_transformer import SwinTransformer
from .transfusion import TransFusion

FeaturesDictType = MutableMapping[str, torch.Tensor]

class SwinX2S(nn.Module):
    def __init__(self,
                 img_size: Union[Sequence[int], int],
                 in_channels: int,
                 out_channels: int,
                 feature_size: int = 96,
                 depths: Sequence[int] = (2, 2, 18, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 norm_name: Union[Tuple, str] = "instance",
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 spatial_dims: int = 3,
                 fusion_depths: Sequence[int] = (2, 2, 2, 2, 2, 2),
                 window_size: Sequence[int] = (8, 8)) -> None:
        super().__init__()
        assert isinstance(img_size, (list, tuple)) and len(img_size) == 3, "img_size must be [D,H,W]"
        D, H, W = img_size
        assert D % 32 == 0, "Depth D should be divisible by 32."
        self.img_size = img_size

        # 3D encoders
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_channels, feature_size, 3, 1, norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims, feature_size, feature_size, 3, 1, norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims, 2*feature_size, 2*feature_size, 3, 1, norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims, 4*feature_size, 4*feature_size, 3, 1, norm_name, res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims, 8*feature_size, 8*feature_size, 3, 1, norm_name, res_block=True)
        self.encoder10 = UnetrBasicBlock(spatial_dims, 16*feature_size, 16*feature_size, 3, 1, norm_name, res_block=True)

        # 3D decoders
        self.decoder5 = UnetrUpBlock(spatial_dims, 16*feature_size, 8*feature_size, 3, 2, norm_name, res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims, 8*feature_size, 4*feature_size, 3, 2, norm_name, res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims, 4*feature_size, 2*feature_size, 3, 2, norm_name, res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims, 2*feature_size, feature_size, 3, 2, norm_name, res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims, feature_size, feature_size, 3, 2, norm_name, res_block=True)
        self.out = UnetOutBlock(spatial_dims, feature_size, out_channels)

        # cross-attention
        self.cross_atte6 = TransFusion(hidden_size=feature_size*16, num_layers=fusion_depths[5],
                                       mlp_dim=feature_size*32, num_heads=num_heads[3],
                                       dropout_rate=drop_rate, atte_dropout_rate=attn_drop_rate,
                                       roi_size=img_size, scale=32)

        # 2D Swin
        self.swinViT = SwinTransformer(in_chans=in_channels, embed_dim=feature_size, depths=depths,
                                       img_size=img_size[1:], window_size=window_size,
                                       patch_size=2, num_heads=num_heads)

        # 2D -> 3D inflation
        self.expand_ = nn.ModuleList([nn.Conv3d(1, int(img_size[0] / (2 ** i)), 1)
                                      for i in range(len(depths) + 2)])
        for e in self.expand_:
            nn.init.constant_(e.weight, 1.0)
            nn.init.constant_(e.bias, 0.0)

        # fusion head
        self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=True)

    def _inflate_list_to_3d(self, feats_2d_list, *, pos: bool):
        out = []
        for i, x in enumerate(feats_2d_list):
            x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(1)
            x = self.expand_[i](x)
            x = x.permute(0, 2, 1, 3, 4)
            out.append(x)
        return out

    def forward_view_encoder(self, x, pos=True):
        feats = self.swinViT(x)
        feats.insert(0, x.permute(0, 2, 3, 1))
        vols = self._inflate_list_to_3d(feats, pos=pos)

        x_enc0 = self.encoder1(vols[0])
        x_enc1 = self.encoder2(vols[1])
        x_enc2 = self.encoder3(vols[2])
        x_enc3 = self.encoder4(vols[3])
        x_enc4 = self.encoder5(vols[4])
        x_dec4 = self.encoder10(vols[5])
        return {"enc0": x_enc0, "enc1": x_enc1, "enc2": x_enc2, "enc3": x_enc3, "enc4": x_enc4, "dec4": x_dec4}

    def forward_view_decoder(self, feats: FeaturesDictType):
        x = self.decoder5(feats["dec4"], feats["enc4"])
        x = self.decoder4(x, feats["enc3"])
        x = self.decoder3(x, feats["enc2"])
        x = self.decoder2(x, feats["enc1"])
        x = self.decoder1(x, feats["enc0"])
        return self.out(x)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        xa_e = self.forward_view_encoder(xa, pos=True)
        xb_e = self.forward_view_encoder(xb, pos=False)
        xa_e["dec4"], xb_e["dec4"] = self.cross_atte6(xa_e["dec4"], xb_e["dec4"])
        ya = self.forward_view_decoder(xa_e)
        yb = self.forward_view_decoder(xb_e)
        y = torch.cat([ya, yb], dim=1)
        return self.fuse(y)