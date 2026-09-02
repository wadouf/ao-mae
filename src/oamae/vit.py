from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


def sincos_position(tokens_per_side: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError('Position dimension must be divisible by four')
    grid = torch.arange(tokens_per_side, dtype=torch.float32)
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    quarter = dim // 4
    omega = torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter)
    parts = []
    for axis in (y, x):
        angle = axis.reshape(-1, 1) * omega.reshape(1, -1)
        parts.extend([angle.sin(), angle.cos()])
    return torch.cat(parts, dim=1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.projection(out.transpose(1, 2).reshape(b, n, d))


class CrossAttention(nn.Module):
    """Optical queries with projected SAR keys and values."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_query = nn.Linear(dim, dim, bias=True)
        self.to_key = nn.Linear(dim, dim, bias=True)
        self.to_value = nn.Linear(dim, dim, bias=True)
        self.projection = nn.Linear(dim, dim)

    def forward(self, optical: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        b, n, d = optical.shape
        m = radar.shape[1]
        q = self.to_query(optical).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k = self.to_key(radar).reshape(b, m, self.heads, d // self.heads).transpose(1, 2)
        v = self.to_value(radar).reshape(b, m, self.heads, d // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.projection(out.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GatedFusionBlock(Block):
    """Equation 6: a standard optical block plus a gated SAR-to-optical update.

    The optical stream stays the primary representation. The cross-attention update
    is scaled token by token by the effective gate, so radar contributes only where
    optical evidence is degraded and the radar evidence is structurally reliable.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__(dim, heads, mlp_ratio)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_radar = nn.LayerNorm(dim)
        self.cross_attention = CrossAttention(dim, heads)

    def forward(self, x: torch.Tensor, radar: torch.Tensor | None = None, gate: torch.Tensor | None = None) -> torch.Tensor:
        x = super().forward(x)
        if radar is None:
            return x
        update = self.cross_attention(self.norm_cross(x), self.norm_radar(radar))
        if gate is not None:
            update = update * gate.unsqueeze(-1)
        return x + update
