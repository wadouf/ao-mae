from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_tokens(cloud: torch.Tensor, token_size: int) -> torch.Tensor:
    """Equation 1: token-level cloud probability by average pooling."""
    if cloud.dim() == 3:
        cloud = cloud.unsqueeze(1)
    height, width = cloud.shape[-2:]
    if height % token_size or width % token_size:
        raise ValueError('Cloud dimensions must be divisible by the token size')
    return F.avg_pool2d(cloud, token_size).squeeze(1)


def refine_tokens(token_cloud: torch.Tensor, seed_threshold: float, delta: float) -> torch.Tensor:
    """Equation 2: deterministic refinement, external to the learned predictor."""
    seeded = (token_cloud >= seed_threshold).to(token_cloud.dtype)
    return torch.clamp(token_cloud + delta * token_cloud * seeded, max=1.0)


def broadcast_tokens(tokens: torch.Tensor, token_size: int) -> torch.Tensor:
    return tokens.repeat_interleave(token_size, dim=-2).repeat_interleave(token_size, dim=-1)


def refined_cloud(cloud: torch.Tensor, token_size: int, seed_threshold: float, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the refined cloud probability at token level and at pixel level."""
    tokens = refine_tokens(pool_tokens(cloud, token_size), seed_threshold, delta)
    return tokens, broadcast_tokens(tokens, token_size)


def observable_support(
    cloud_t1: torch.Tensor,
    cloud_t2: torch.Tensor,
    token_size: int = 16,
    seed_threshold: float = 0.20,
    delta: float = 0.30,
    hard_threshold: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Equation 3: per-date visibility masks and the shared support V12.

    The support is external and method invariant: it depends only on the cloud
    product and the fixed thresholds, never on model confidence.
    """
    tokens_t1, pixels_t1 = refined_cloud(cloud_t1, token_size, seed_threshold, delta)
    tokens_t2, pixels_t2 = refined_cloud(cloud_t2, token_size, seed_threshold, delta)
    mask_t1 = (pixels_t1 <= hard_threshold)
    mask_t2 = (pixels_t2 <= hard_threshold)
    v12 = mask_t1 & mask_t2
    return {
        'cloud_token_t1': tokens_t1,
        'cloud_token_t2': tokens_t2,
        'cloud_pixel_t1': pixels_t1,
        'cloud_pixel_t2': pixels_t2,
        'mask_t1': mask_t1,
        'mask_t2': mask_t2,
        'v12': v12,
        'coverage': v12.flatten(1).float().mean(dim=1),
    }


def cloud_gate(token_cloud: torch.Tensor, alpha: float = 10.0, threshold: float = 0.50) -> torch.Tensor:
    """Equation 4: radar influence grows as optical evidence degrades."""
    return torch.sigmoid(alpha * (token_cloud - threshold))


def _standardize(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((flat - mean) / std).view_as(value)


def radar_descriptors(radar: torch.Tensor, token_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Structural salience and a speckle proxy, pooled to token level and standardized.

    Salience is the mean gradient magnitude inside the token. The speckle proxy is
    the within-token coefficient of variation, which grows where the backscatter is
    noisy rather than structured.
    """
    gradient_y = torch.zeros_like(radar)
    gradient_x = torch.zeros_like(radar)
    gradient_y[..., 1:, :] = radar[..., 1:, :] - radar[..., :-1, :]
    gradient_x[..., :, 1:] = radar[..., :, 1:] - radar[..., :, :-1]
    magnitude = torch.sqrt(gradient_y.pow(2) + gradient_x.pow(2) + 1e-12).mean(dim=1, keepdim=True)
    salience = F.avg_pool2d(magnitude, token_size).squeeze(1)

    intensity = radar.mean(dim=1, keepdim=True)
    local_mean = F.avg_pool2d(intensity, token_size)
    local_square = F.avg_pool2d(intensity.pow(2), token_size)
    variance = (local_square - local_mean.pow(2)).clamp_min(0.0)
    speckle = (variance.sqrt() / local_mean.abs().clamp_min(1e-6)).squeeze(1)

    return _standardize(salience), _standardize(speckle)


class SARReliabilityGate(nn.Module):
    """Equation 5: learned suppression of structurally weak radar evidence."""

    def __init__(self, token_size: int = 16) -> None:
        super().__init__()
        self.token_size = token_size
        self.theta_structure = nn.Parameter(torch.tensor(1.0))
        self.theta_noise = nn.Parameter(torch.tensor(1.0))
        self.theta_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        salience, speckle = radar_descriptors(radar, self.token_size)
        logit = self.theta_structure * salience - self.theta_noise * speckle + self.theta_bias
        return torch.sigmoid(logit)


def effective_gate(cloud_gate_value: torch.Tensor, radar_reliability: torch.Tensor) -> torch.Tensor:
    """The effective fusion gate is the product of the two gates."""
    return cloud_gate_value * radar_reliability
