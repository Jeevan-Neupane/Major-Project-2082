import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import MutableMapping, Sequence, Tuple, Union

from .swinX2S.unetr_blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from .swinX2S.swin_transformer import SwinTransformer
from .swinX2S.transfusion import TransFusion

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
                 window_size: Sequence[int] = (8, 8),
                 geometry: str = "legacy",
                 output_mode: str = "fuse",
                 swin_impl: str = "legacy") -> None:
        """Adapted from the official Swin-X2S code (MIT licence, (c) 2025 liukuan5625).

        Extra options (defaults reproduce the November 2026 models exactly):
            geometry: "legacy" inflates the lateral view exactly like the AP view
                (image columns on W, copies along D), which is geometrically wrong for
                our data. "correct" puts the lateral image columns (anterior-posterior)
                on D and copies along W. Volume axes are (D, H, W) = (anterior-posterior,
                superior-inferior, left-right). The official code permutes the second view
                too, but for a different image layout, so it is not reused as is.
            swin_impl: "legacy" keeps the November 2026 window code of models/swinX2S/swin_transformer.py
                (windows built with unfold + view, which mixes pixels and channels, and no drop path).
                "official" uses standard Swin window partition/reverse and applies drop path (rate up
                to 0.2 over depth), as in the official Swin-X2S code.
            output_mode: "fuse" (our 1x1x1 conv over both view outputs) or "average"
                (official behaviour: the prediction is the mean of the two view logits;
                the per-view logits are kept in ``self.view_logits`` for the official
                two-view loss, see utils/training/losses.py::two_view_loss).
        """
        super().__init__()
        if geometry not in ("legacy", "correct"):
            raise ValueError(f"geometry must be 'legacy' or 'correct', got {geometry}")
        if swin_impl not in ("legacy", "official"):
            raise ValueError(f"swin_impl must be 'legacy' or 'official', got {swin_impl}")
        self.swin_impl = swin_impl
        if output_mode not in ("fuse", "average"):
            raise ValueError(f"output_mode must be 'fuse' or 'average', got {output_mode}")
        self.geometry = geometry
        self.output_mode = output_mode
        assert isinstance(img_size, (list, tuple)) and len(img_size) == 3, "img_size must be [D,H,W]"
        D, H, W = img_size
        assert D % 32 == 0, "Depth D should be divisible by 32."
        if geometry == "correct":
            assert W % 32 == 0, "Width W should be divisible by 32 for geometry='correct'."
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

        if swin_impl == "official":
            from .swinX2S.swin_transformer import SwinTransformerBlock
            for block in self.swinViT.modules():
                if isinstance(block, SwinTransformerBlock):
                    block.official_windows = True

        # 2D -> 3D inflation
        self.expand_ = nn.ModuleList([nn.Conv3d(1, int(img_size[0] / (2 ** i)), 1)
                                      for i in range(len(depths) + 2)])
        for e in self.expand_:
            nn.init.constant_(e.weight, 1.0)
            nn.init.constant_(e.bias, 0.0)

        if geometry == "correct":
            # lateral view: copies along W (one 1x1x1 conv per scale, initialised as a plain copy)
            self.expand_lat_ = nn.ModuleList([nn.Conv3d(1, int(W / (2 ** i)), 1)
                                              for i in range(len(depths) + 2)])
            for e in self.expand_lat_:
                nn.init.constant_(e.weight, 1.0)
                nn.init.constant_(e.bias, 0.0)

        # fusion head (our addition; the official model averages the two view outputs)
        if output_mode == "fuse":
            self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=True)

    def _inflate_list_to_3d(self, feats_2d_list, *, pos: bool):
        """Turn 2D features [B, H_i, W_i, C] into volumes [B, C, D_i, H_i, W_i].

        pos=True is the AP view, pos=False the lateral view. In "legacy" mode both views are
        treated the same way (the flag is ignored, as in the November 2026 models).
        """
        out = []
        for i, x in enumerate(feats_2d_list):
            x = x.permute(0, 3, 1, 2)                       # [B, C, rows, cols]
            if self.geometry == "correct" and not pos:
                depth_i = self.expand_[i].out_channels
                # lateral columns run anterior-posterior: resize them to D_i
                x = F.interpolate(x, size=(x.shape[2], depth_i), mode="bilinear", align_corners=False)
                x = x.permute(0, 1, 3, 2).unsqueeze(1)      # [B, 1, C, D_i, H_i]
                x = self.expand_lat_[i](x)                  # [B, W_i, C, D_i, H_i]  (copies along W)
                x = x.permute(0, 2, 3, 4, 1)                # [B, C, D_i, H_i, W_i]
            else:
                x = x.unsqueeze(1)                          # [B, 1, C, H_i, W_i]
                x = self.expand_[i](x)                      # [B, D_i, C, H_i, W_i]  (copies along D)
                x = x.permute(0, 2, 1, 3, 4)                # [B, C, D_i, H_i, W_i]
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
        if self.output_mode == "average":
            self.view_logits = (ya, yb)
            return (ya + yb) / 2
        y = torch.cat([ya, yb], dim=1)
        return self.fuse(y)