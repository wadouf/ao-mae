from __future__ import annotations

import torch
import torch.nn as nn

from .config import OAMAEConfig
from .encoders import OpticalEncoder, RadarEncoder
from .losses import reconstruction_loss, redundancy_reduction_loss, structural_fallback_loss
from .masking import cloud_mix
from .support import SARReliabilityGate, broadcast_tokens, cloud_gate, effective_gate, refined_cloud
from .targets import opacity_clamp, past_only_target, safety_weight
from .vit import Block, sincos_position


class ReconstructionDecoder(nn.Module):
    """Offline MAE decoder used during Stage I only."""

    def __init__(self, cfg: OAMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.project = nn.Linear(cfg.optical_dim, cfg.decoder_dim)
        self.register_buffer('position', sincos_position(cfg.tokens_per_side(), cfg.decoder_dim), persistent=False)
        self.blocks = nn.ModuleList([Block(cfg.decoder_dim, cfg.decoder_heads, cfg.mlp_ratio) for _ in range(cfg.decoder_depth)])
        self.norm = nn.LayerNorm(cfg.decoder_dim)
        self.head = nn.Linear(cfg.decoder_dim, cfg.patch_size ** 2 * cfg.optical_bands)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.project(tokens) + self.position
        for block in self.blocks:
            x = block(x)
        patches = self.head(self.norm(x))
        cfg = self.cfg
        side = cfg.tokens_per_side()
        batch = patches.shape[0]
        patches = patches.reshape(batch, side, side, cfg.patch_size, cfg.patch_size, cfg.optical_bands)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, cfg.optical_bands, cfg.image_size, cfg.image_size)


class OAMAEPretrainer(nn.Module):
    """Stage I: observability-aligned masked pretraining.

    Cloud-Mix masking, bounded past-only retrieval and the reconstruction path run
    offline. Only the pretrained optical and SAR encoders transfer to Stage II.
    """

    def __init__(self, cfg: OAMAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OAMAEConfig()
        self.optical_encoder = OpticalEncoder(self.cfg)
        self.radar_encoder = RadarEncoder(self.cfg)
        self.reliability = SARReliabilityGate(self.cfg.token_size)
        self.decoder = ReconstructionDecoder(self.cfg)

    def forward(
        self,
        optical: torch.Tensor,
        radar: torch.Tensor,
        cloud: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        token_cloud, _ = refined_cloud(cloud, cfg.token_size, cfg.seed_threshold, cfg.refinement_delta)
        gate_cloud = cloud_gate(token_cloud, cfg.cloud_gate_alpha, cfg.cloud_gate_threshold)
        gate_radar = self.reliability(radar)
        gate = effective_gate(gate_cloud, gate_radar)

        mask = cloud_mix(
            token_cloud, optical, cfg.token_size,
            cfg.mask_ratio, cfg.mask_cloud_fraction, cfg.mask_struct_fraction,
            generator=generator,
        )
        radar_tokens = self.radar_encoder(radar)
        optical_tokens, _ = self.optical_encoder(optical, radar_tokens=radar_tokens, gate=gate.flatten(1), mask=mask)
        reconstruction = self.decoder(optical_tokens)
        return {
            'reconstruction': reconstruction,
            'optical_tokens': optical_tokens,
            'radar_tokens': radar_tokens,
            'mask': mask,
            'token_cloud': token_cloud,
            'cloud_gate': gate_cloud,
            'radar_reliability': gate_radar,
            'effective_gate': gate,
        }

    def loss(self, batch: dict[str, torch.Tensor], generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Equation 14: L_pre = lambda_rec L_rec + lambda_str L_str + lambda_rr L_rr."""
        cfg = self.cfg
        optical, radar, cloud = batch['optical'], batch['radar'], batch['cloud']
        outputs = self.forward(optical, radar, cloud, generator=generator)

        retrieval = past_only_target(
            batch['history_optical'], batch['history_cloud'], batch['ages_days'],
            cfg.token_size, cfg.maximum_age_days, cfg.maximum_candidates, cfg.clear_threshold,
        )
        safety, distance = safety_weight(
            radar, batch['history_radar'], retrieval['selected_indices'],
            cfg.token_size, cfg.safety_floor, cfg.safety_coefficient,
        )
        clamp = opacity_clamp(outputs['token_cloud'], cfg.hard_threshold)

        side = cfg.tokens_per_side()
        masked = outputs['mask'].reshape(-1, side, side)
        fallback = retrieval['fallback']
        scored = masked & ~fallback
        weight = broadcast_tokens(clamp * safety * scored.to(safety.dtype), cfg.token_size)
        fallback_weight = broadcast_tokens((masked & fallback).to(safety.dtype), cfg.token_size)

        l_rec = reconstruction_loss(outputs['reconstruction'], retrieval['target'], weight, cfg.gradient_loss_weight)
        l_str = structural_fallback_loss(outputs['reconstruction'], radar, fallback_weight)
        l_rr = redundancy_reduction_loss(outputs['optical_tokens'], outputs['radar_tokens'], cfg.vicreg_variance_target)
        total = cfg.lambda_rec * l_rec + cfg.lambda_str * l_str + cfg.lambda_rr * l_rr

        return {
            'loss': total,
            'l_rec': l_rec,
            'l_str': l_str,
            'l_rr': l_rr,
            'fallback_rate': fallback.to(total.dtype).mean(),
            'target_availability': 1.0 - fallback.to(total.dtype).mean(),
            'safety_distance': distance.mean(),
        }

    def export_encoders(self) -> dict[str, torch.Tensor]:
        """State dict of the components transferred to Stage II."""
        state = {f'optical_encoder.{k}': v for k, v in self.optical_encoder.state_dict().items()}
        state.update({f'radar_encoder.{k}': v for k, v in self.radar_encoder.state_dict().items()})
        state.update({f'reliability.{k}': v for k, v in self.reliability.state_dict().items()})
        return state
