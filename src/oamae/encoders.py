from __future__ import annotations

import torch
import torch.nn as nn

from .config import OAMAEConfig
from .vit import Block, GatedFusionBlock, PatchEmbed, sincos_position


class RadarEncoder(nn.Module):
    """ViT-Tiny over Sentinel-1 VV and VH, projected to the optical width."""

    def __init__(self, cfg: OAMAEConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.radar_bands, cfg.radar_dim, cfg.patch_size)
        self.register_buffer('position', sincos_position(cfg.tokens_per_side(), cfg.radar_dim), persistent=False)
        self.blocks = nn.ModuleList([Block(cfg.radar_dim, cfg.radar_heads, cfg.mlp_ratio) for _ in range(cfg.radar_depth)])
        self.norm = nn.LayerNorm(cfg.radar_dim)
        self.to_optical = nn.Linear(cfg.radar_dim, cfg.optical_dim)

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(radar) + self.position
        for block in self.blocks:
            x = block(x)
        return self.to_optical(self.norm(x))


class OpticalEncoder(nn.Module):
    """ViT-S/16 over the ten Sentinel-2 bands, with gated SAR fusion in the final blocks."""

    def __init__(self, cfg: OAMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.optical_bands, cfg.optical_dim, cfg.patch_size)
        self.register_buffer('position', sincos_position(cfg.tokens_per_side(), cfg.optical_dim), persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.optical_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        plain = cfg.optical_depth - cfg.fusion_blocks
        if plain < 0:
            raise ValueError('fusion_blocks cannot exceed optical_depth')
        blocks: list[nn.Module] = [Block(cfg.optical_dim, cfg.optical_heads, cfg.mlp_ratio) for _ in range(plain)]
        blocks += [GatedFusionBlock(cfg.optical_dim, cfg.optical_heads, cfg.mlp_ratio) for _ in range(cfg.fusion_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.optical_dim)

    def forward(
        self,
        optical: torch.Tensor,
        radar_tokens: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_layers: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embed(optical)
        if mask is not None:
            keep = (~mask).to(x.dtype).unsqueeze(-1)
            x = x * keep + self.mask_token.to(x.dtype) * (1.0 - keep)
        x = x + self.position

        wanted = set(return_layers or ())
        intermediates: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if isinstance(block, GatedFusionBlock):
                x = block(x, radar_tokens, gate)
            else:
                x = block(x)
            if index in wanted:
                intermediates.append(x)
        return self.norm(x), intermediates
