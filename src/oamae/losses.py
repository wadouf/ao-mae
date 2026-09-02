from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gradient_y = torch.zeros_like(x)
    gradient_x = torch.zeros_like(x)
    gradient_y[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    gradient_x[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    return gradient_y, gradient_x


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    total = weight.sum()
    if total <= 0:
        return value.sum() * 0.0
    return (value * weight).sum() / total


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pixel_weight: torch.Tensor,
    gradient_weight: float = 0.50,
) -> torch.Tensor:
    """Equation 13: weighted pixel and gradient reconstruction over the scored tokens.

    pixel_weight carries the product of the opacity clamp and the safety weight,
    broadcast to pixels, and is zero outside the masked non-fallback tokens.
    """
    weight = pixel_weight.unsqueeze(1)
    pixel_term = (prediction - target).abs().mean(dim=1, keepdim=True)

    pred_y, pred_x = spatial_gradients(prediction)
    target_y, target_x = spatial_gradients(target)
    gradient_term = ((pred_y - target_y).abs() + (pred_x - target_x).abs()).mean(dim=1, keepdim=True)

    return _weighted_mean(pixel_term + gradient_weight * gradient_term, weight)


def structural_fallback_loss(
    prediction: torch.Tensor,
    radar: torch.Tensor,
    fallback_weight: torch.Tensor,
) -> torch.Tensor:
    """Structural supervision where no valid past optical target exists.

    Radiometry cannot be supervised without an optical target, so only the gradient
    structure is matched, against the current radar observation. This is the
    implementation reading of the structural fallback described in the paper.
    """
    pred_y, pred_x = spatial_gradients(prediction)
    radar_y, radar_x = spatial_gradients(radar)
    pred_structure = torch.sqrt(pred_y.pow(2) + pred_x.pow(2) + 1e-12).mean(dim=1, keepdim=True)
    radar_structure = torch.sqrt(radar_y.pow(2) + radar_x.pow(2) + 1e-12).mean(dim=1, keepdim=True)

    def normalize(value: torch.Tensor) -> torch.Tensor:
        flat = value.flatten(1)
        mean = flat.mean(dim=1).view(-1, 1, 1, 1)
        std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return (value - mean) / std

    difference = (normalize(pred_structure) - normalize(radar_structure)).abs()
    return _weighted_mean(difference, fallback_weight.unsqueeze(1))


def redundancy_reduction_loss(
    optical_tokens: torch.Tensor,
    radar_tokens: torch.Tensor,
    variance_target: float = 1.0,
) -> torch.Tensor:
    """Reduced VICReg-style term: cross-modal invariance plus a feature-variance hinge."""
    invariance = F.mse_loss(optical_tokens, radar_tokens)
    variance = torch.tensor(0.0, device=optical_tokens.device, dtype=optical_tokens.dtype)
    for stream in (optical_tokens, radar_tokens):
        flat = stream.reshape(-1, stream.shape[-1])
        deviation = (flat.var(dim=0, unbiased=False) + 1e-6).sqrt()
        variance = variance + F.relu(variance_target - deviation).mean()
    return invariance + variance


def focal_loss(probability: torch.Tensor, reference: torch.Tensor, support: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    p = probability.clamp(1e-6, 1 - 1e-6)
    y = reference.to(p.dtype)
    pt = p * y + (1 - p) * (1 - y)
    weight = alpha * y + (1 - alpha) * (1 - y)
    loss = -weight * (1 - pt).pow(gamma) * pt.log()
    return _weighted_mean(loss, support.to(p.dtype))


def dice_loss(probability: torch.Tensor, reference: torch.Tensor, support: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    mask = support.to(probability.dtype)
    p = probability * mask
    y = reference.to(probability.dtype) * mask
    intersection = (p * y).flatten(1).sum(dim=1)
    total = p.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    return (1 - (2 * intersection + epsilon) / (total + epsilon)).mean()


def supervised_loss(
    probability: torch.Tensor,
    reference: torch.Tensor,
    support: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Equation 16: focal plus Dice, evaluated only on the common support."""
    return focal_loss(probability, reference, support, gamma, alpha) + dice_loss(probability, reference, support)
