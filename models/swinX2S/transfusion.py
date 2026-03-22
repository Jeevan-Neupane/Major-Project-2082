import math
import torch
import torch.nn as nn
from typing import Union, Sequence

class Attention(nn.Module):
    def __init__(self, num_heads=8, hidden_size=768, atte_dropout_rate=0.0):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(atte_dropout_rate)
        self.proj_dropout = nn.Dropout(atte_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def _reshape(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        q1, k1, v1 = self._reshape(self.query(x1)), self._reshape(self.key(x1)), self._reshape(self.value(x1))
        q2, k2, v2 = self._reshape(self.query(x2)), self._reshape(self.key(x2)), self._reshape(self.value(x2))

        attn1 = self.softmax((q1 @ k2.transpose(-1, -2)) / math.sqrt(self.attention_head_size))
        attn1 = self.attn_dropout(attn1)
        ctx1 = (attn1 @ v2).permute(0, 2, 1, 3).contiguous().view(x1.size(0), x1.size(1), -1)
        out1 = self.proj_dropout(self.out(ctx1))

        attn2 = self.softmax((q2 @ k1.transpose(-1, -2)) / math.sqrt(self.attention_head_size))
        attn2 = self.attn_dropout(attn2)
        ctx2 = (attn2 @ v1).permute(0, 2, 1, 3).contiguous().view(x2.size(0), x2.size(1), -1)
        out2 = self.proj_dropout(self.out(ctx2))

        return out1, out2

class Block(nn.Module):
    def __init__(self, hidden_size=768, mlp_dim=1536, dropout_rate=0.5, num_heads=8, atte_dropout_rate=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(num_heads=num_heads, hidden_size=hidden_size, atte_dropout_rate=atte_dropout_rate)
    
    def forward(self, x1, x2):
        x1 = self.attention_norm(x1)
        x2 = self.attention_norm(x2)
        x1, x2 = self.attn(x1, x2)
        return x1, x2

class TransFusion(nn.Module):
    def __init__(self, hidden_size=768, num_layers=2, mlp_dim=1536, dropout_rate=0.5,
                 num_heads=8, atte_dropout_rate=0.0, roi_size: Union[Sequence[int], int] = (64, 64, 64), scale=16):
        super().__init__()
        if isinstance(roi_size, int):
            roi_size = [roi_size, roi_size, roi_size]
        patch_size = (1, 1, 1)
        self.size = (roi_size[0] // patch_size[0] // scale,
                     roi_size[1] // patch_size[1] // scale,
                     roi_size[2] // patch_size[2] // scale)
        n_patches = self.size[0] * self.size[1] * self.size[2]
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.patch_embeddings = nn.Conv3d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer = nn.ModuleList([
            Block(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate,
                  num_heads=num_heads, atte_dropout_rate=atte_dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x1, x2):
        x1 = self.patch_embeddings(x1)
        x2 = self.patch_embeddings(x2)
        x1 = x1.flatten(2).transpose(-1, -2)
        x2 = x2.flatten(2).transpose(-1, -2)
        x1 = self.dropout(x1 + self.position_embeddings)
        x2 = self.dropout(x2 + self.position_embeddings)
        for blk in self.layer:
            x1, x2 = blk(x1, x2)
        x1 = self.encoder_norm(x1)
        x2 = self.encoder_norm(x2)
        B, N, hidden = x1.size()
        x1 = x1.permute(0, 2, 1).contiguous().view(B, hidden, self.size[0], self.size[1], self.size[2])
        x2 = x2.permute(0, 2, 1).contiguous().view(B, hidden, self.size[0], self.size[1], self.size[2])
        return x1, x2