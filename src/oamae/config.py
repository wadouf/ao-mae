from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OAMAEConfig:
    """Hyperparameters of OA-MAE.

    Values marked "paper" are stated in the manuscript or in SCIENTIFIC_CONTRACT.md.
    Values marked "implementation" are not fixed by the manuscript and are exposed
    here so that a run can declare the setting it used.
    """

    # Inputs (paper)
    image_size: int = 256
    patch_size: int = 16
    optical_bands: int = 10
    radar_bands: int = 2

    # Optical stream, ViT-S/16 (paper)
    optical_dim: int = 384
    optical_depth: int = 12
    optical_heads: int = 6

    # SAR stream, ViT-Tiny (paper)
    radar_dim: int = 192
    radar_depth: int = 12
    radar_heads: int = 3

    # Gated SAR-to-optical fusion in the final optical blocks (paper)
    fusion_blocks: int = 4

    # External cloud support, equations 1 to 3 (paper)
    token_size: int = 16
    seed_threshold: float = 0.20
    refinement_delta: float = 0.30
    hard_threshold: float = 0.85

    # Cloud gate, equation 4 (paper: sigma(10 (c' - 0.50)))
    cloud_gate_alpha: float = 10.0
    cloud_gate_threshold: float = 0.50

    # Cloud-Mix, equation 7 (paper)
    mask_ratio: float = 0.75
    mask_cloud_fraction: float = 0.50
    mask_struct_fraction: float = 0.25

    # Past-only targets, equations 8 to 11 (paper)
    maximum_age_days: int = 90
    maximum_candidates: int = 3
    clear_threshold: float = 0.15
    safety_floor: float = 0.10
    safety_coefficient: float = 10.0

    # Stage I objective, equation 14 (paper)
    lambda_rec: float = 1.00
    lambda_str: float = 0.50
    lambda_rr: float = 0.10

    # Stage II supervision, equation 16 (paper)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    probability_threshold: float = 0.50

    # Stage II heads (paper: feature maps in R^{HxWx64}, 3 residual blocks)
    adapter_channels: int = 64
    decoder_blocks: int = 3
    pyramid_layers: tuple[int, ...] = (2, 5, 8, 11)

    # Stage I reconstruction decoder, offline only (implementation)
    decoder_dim: int = 192
    decoder_depth: int = 4
    decoder_heads: int = 3

    # Not fixed by the manuscript (implementation)
    mlp_ratio: float = 4.0
    gradient_loss_weight: float = 0.50
    vicreg_variance_target: float = 1.0
    drop_path: float = 0.0

    def tokens_per_side(self) -> int:
        return self.image_size // self.patch_size

    def token_count(self) -> int:
        return self.tokens_per_side() ** 2
