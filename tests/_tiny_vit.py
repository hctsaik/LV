"""測試用迷你 ViT：有 attn.qkv / norm1 / forward_features，不載真實 DINO。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T


class TinyAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]
        a = (q @ k.transpose(-2, -1)) * ((c // self.num_heads) ** -0.5)
        a = a.softmax(-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(o)


class TinyBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, *, dim=16, heads=2, patch=8, depth=2, n_register=0):
        super().__init__()
        self.embed_dim = dim
        self.num_heads = heads
        self.patch_size = patch
        self.num_register_tokens = n_register
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, n_register, dim)) if n_register else None
        )
        self.blocks = nn.ModuleList([TinyBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x):
        b, _c, h, w = x.shape
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)
        parts = [self.cls_token.expand(b, -1, -1)]
        if self.register_tokens is not None:
            parts.append(self.register_tokens.expand(b, -1, -1))
        parts.append(tok)
        x = torch.cat(parts, 1)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        r = int(self.num_register_tokens)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:1 + r],
            "x_norm_patchtokens": x_norm[:, 1 + r:],
            "x_prenorm": x,
        }

    def forward(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


class TinyExplainer:
    def __init__(self, model: TinyViT):
        self.model = model.eval()
        self.device = torch.device("cpu")
        self.transform = T.Compose([T.ToTensor()])
