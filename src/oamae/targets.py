from __future__ import annotations

import torch

from .support import broadcast_tokens, pool_tokens


def eligible_history(ages_days: torch.Tensor, maximum_age_days: int) -> torch.Tensor:
    """Equation 8: strictly earlier observations inside a bounded recency window."""
    return (ages_days > 0) & (ages_days <= maximum_age_days)


def past_only_target(
    history_optical: torch.Tensor,
    history_cloud: torch.Tensor,
    ages_days: torch.Tensor,
    token_size: int = 16,
    maximum_age_days: int = 90,
    maximum_candidates: int = 3,
    clear_threshold: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Equation 9: token-wise median of the clearest eligible past observations.

    history_optical is (batch, time, band, height, width), history_cloud is
    (batch, time, height, width) and ages_days is (batch, time). No candidate may
    occur at or after the current date, which is enforced through ages_days.
    """
    batch, time, bands, height, width = history_optical.shape
    tokens_y, tokens_x = height // token_size, width // token_size

    token_cloud = torch.stack([pool_tokens(history_cloud[:, t], token_size) for t in range(time)], dim=1)
    eligible = eligible_history(ages_days, maximum_age_days)
    valid = eligible.view(batch, time, 1, 1) & (token_cloud <= clear_threshold)

    ranking = token_cloud.masked_fill(~valid, float('inf'))
    order = ranking.argsort(dim=1)[:, :maximum_candidates]
    selected_cloud = ranking.gather(1, order)
    selected_valid = torch.isfinite(selected_cloud)

    ages = ages_days.view(batch, time, 1, 1).expand(-1, -1, tokens_y, tokens_x)
    selected_ages = ages.gather(1, order).masked_fill(~selected_valid, -1)

    pixel_order = broadcast_tokens(order, token_size)
    gather_index = pixel_order.unsqueeze(2).expand(-1, -1, bands, -1, -1)
    candidates = history_optical.gather(1, gather_index)

    pixel_valid = broadcast_tokens(selected_valid, token_size).unsqueeze(2)
    large = torch.where(pixel_valid, candidates, torch.full_like(candidates, float('nan')))
    target = large.nanmedian(dim=1).values
    target = torch.nan_to_num(target, nan=0.0)

    fallback = ~selected_valid.any(dim=1)
    return {
        'target': target,
        'fallback': fallback,
        'selected_indices': order.masked_fill(~selected_valid, -1),
        'selected_ages': selected_ages,
        'candidate_count': selected_valid.sum(dim=1),
    }


def safety_weight(
    current_radar: torch.Tensor,
    history_radar: torch.Tensor,
    selected_indices: torch.Tensor,
    token_size: int = 16,
    safety_floor: float = 0.10,
    safety_coefficient: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equations 10 and 11: downweight targets whose radar context has drifted.

    The descriptor is the token-mean backscatter of each polarization. The distance
    is the L2 norm between the current descriptor and the median over the selected
    past observations.
    """
    batch, time, channels, height, width = history_radar.shape
    tokens_y, tokens_x = height // token_size, width // token_size

    def descriptor(volume: torch.Tensor) -> torch.Tensor:
        flat = volume.reshape(-1, channels, height, width)
        pooled = torch.stack([pool_tokens(flat[:, c], token_size) for c in range(channels)], dim=1)
        return pooled.reshape(*volume.shape[:-3], channels, tokens_y, tokens_x)

    current = descriptor(current_radar)
    history = descriptor(history_radar)

    valid = selected_indices >= 0
    index = selected_indices.clamp_min(0).unsqueeze(2).expand(-1, -1, channels, -1, -1)
    gathered = history.gather(1, index)
    gathered = torch.where(valid.unsqueeze(2), gathered, torch.full_like(gathered, float('nan')))
    median = torch.nan_to_num(gathered.nanmedian(dim=1).values, nan=0.0)

    distance = torch.linalg.vector_norm(current - median, dim=1)
    weight = torch.clamp(1.0 - safety_coefficient * distance, min=safety_floor)
    return weight, distance


def opacity_clamp(refined_token_cloud: torch.Tensor, hard_threshold: float = 0.85) -> torch.Tensor:
    """The hard clamp that forbids direct optical reconstruction under opaque cloud."""
    return (refined_token_cloud <= hard_threshold).to(refined_token_cloud.dtype)
