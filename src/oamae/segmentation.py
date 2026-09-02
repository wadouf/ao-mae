from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OAMAEConfig
from .encoders import OpticalEncoder, RadarEncoder
from .support import SARReliabilityGate, cloud_gate, effective_gate, observable_support


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        return x + self.norm2(self.conv2(h))


class FeaturePyramidAdapter(nn.Module):
    """Maps intermediate transformer blocks to a dense feature map in R^{HxWx64}."""

    def __init__(self, cfg: OAMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.side = cfg.tokens_per_side()
        self.projections = nn.ModuleList(
            [nn.Conv2d(cfg.optical_dim, cfg.adapter_channels, 1) for _ in cfg.pyramid_layers]
        )
        steps = 0
        size = self.side
        while size < cfg.image_size:
            size *= 2
            steps += 1
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(cfg.adapter_channels, cfg.adapter_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(8, cfg.adapter_channels),
                    nn.GELU(),
                )
                for _ in range(steps)
            ]
        )

    def forward(self, intermediates: list[torch.Tensor]) -> torch.Tensor:
        if len(intermediates) != len(self.projections):
            raise ValueError('Unexpected number of pyramid inputs')
        batch = intermediates[0].shape[0]
        fused = None
        for tokens, projection in zip(intermediates, self.projections):
            grid = tokens.transpose(1, 2).reshape(batch, -1, self.side, self.side)
            mapped = projection(grid)
            fused = mapped if fused is None else fused + mapped
        for stage in self.upsample:
            fused = stage(F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False))
        return fused


class ChangeDecoder(nn.Module):
    """Lightweight residual decoder with one downsampling and one upsampling path."""

    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.down(x))
        h = self.up(F.interpolate(h, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return self.head(h + x).squeeze(1)


class OAMAEChangeDetector(nn.Module):
    """Stage II: few-shot dense change detection with explicit deferral."""

    def __init__(self, cfg: OAMAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OAMAEConfig()
        self.optical_encoder = OpticalEncoder(self.cfg)
        self.radar_encoder = RadarEncoder(self.cfg)
        self.reliability = SARReliabilityGate(self.cfg.token_size)
        self.adapter = FeaturePyramidAdapter(self.cfg)
        self.combine = nn.Conv2d(2 * self.cfg.adapter_channels, self.cfg.adapter_channels, 1)
        self.decoder = ChangeDecoder(self.cfg.adapter_channels, self.cfg.decoder_blocks)

    def freeze_encoders(self) -> None:
        """Primary evaluation setting: the Stage-I encoders are frozen."""
        for module in (self.optical_encoder, self.radar_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def encode_date(self, optical: torch.Tensor, radar: torch.Tensor, token_cloud: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        gate_cloud = cloud_gate(token_cloud, cfg.cloud_gate_alpha, cfg.cloud_gate_threshold)
        gate_radar = self.reliability(radar)
        gate = effective_gate(gate_cloud, gate_radar)

        radar_tokens = self.radar_encoder(radar)
        _, intermediates = self.optical_encoder(
            optical,
            radar_tokens=radar_tokens,
            gate=gate.flatten(1),
            return_layers=cfg.pyramid_layers,
        )
        features = self.adapter(intermediates)
        diagnostics = {'cloud_gate': gate_cloud, 'radar_reliability': gate_radar, 'effective_gate': gate}
        return features, diagnostics

    def forward(
        self,
        optical_t1: torch.Tensor,
        optical_t2: torch.Tensor,
        radar_t1: torch.Tensor,
        radar_t2: torch.Tensor,
        cloud_t1: torch.Tensor,
        cloud_t2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        support = observable_support(cloud_t1, cloud_t2, cfg.token_size, cfg.seed_threshold, cfg.refinement_delta, cfg.hard_threshold)

        features_t1, diagnostics_t1 = self.encode_date(optical_t1, radar_t1, support['cloud_token_t1'])
        features_t2, diagnostics_t2 = self.encode_date(optical_t2, radar_t2, support['cloud_token_t2'])

        difference = (features_t2 - features_t1).abs()
        product = features_t2 * features_t1
        delta = self.combine(torch.cat([difference, product], dim=1))

        logits = self.decoder(delta)
        probability = torch.sigmoid(logits)
        binary = (probability >= cfg.probability_threshold)

        return {
            'probability': probability,
            'binary': binary,
            'v12': support['v12'],
            'coverage': support['coverage'],
            'cloud_gate': diagnostics_t2['cloud_gate'],
            'radar_reliability': diagnostics_t2['radar_reliability'],
            'effective_gate': diagnostics_t2['effective_gate'],
            'cloud_gate_t1': diagnostics_t1['cloud_gate'],
            'radar_reliability_t1': diagnostics_t1['radar_reliability'],
            'effective_gate_t1': diagnostics_t1['effective_gate'],
        }

    @staticmethod
    def operational_output(binary: torch.Tensor, v12: torch.Tensor) -> torch.Tensor:
        """Three-state output: 0 no change, 1 change, 2 unresolved outside the support."""
        state = binary.to(torch.uint8)
        return torch.where(v12, state, torch.full_like(state, 2))
