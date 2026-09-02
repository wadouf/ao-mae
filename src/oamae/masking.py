from __future__ import annotations

import torch
import torch.nn.functional as F


def structural_salience(optical: torch.Tensor, token_size: int) -> torch.Tensor:
    """Mean gradient magnitude per token, used to rank structurally salient tokens."""
    gradient_y = torch.zeros_like(optical)
    gradient_x = torch.zeros_like(optical)
    gradient_y[..., 1:, :] = optical[..., 1:, :] - optical[..., :-1, :]
    gradient_x[..., :, 1:] = optical[..., :, 1:] - optical[..., :, :-1]
    magnitude = torch.sqrt(gradient_y.pow(2) + gradient_x.pow(2) + 1e-12).mean(dim=1, keepdim=True)
    return F.avg_pool2d(magnitude, token_size).flatten(1)


def _top_k_indices(score: torch.Tensor, k: int, taken: torch.Tensor) -> torch.Tensor:
    masked = score.masked_fill(taken, float('-inf'))
    return masked.topk(k, dim=1).indices


def cloud_mix(
    token_cloud: torch.Tensor,
    optical: torch.Tensor,
    token_size: int,
    mask_ratio: float = 0.75,
    cloud_fraction: float = 0.50,
    struct_fraction: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Equation 7: Omega = Omega_cloud union Omega_struct union Omega_rand.

    The mixed policy avoids the degenerate regime in which only cloudy or only
    high-gradient locations are reconstructed. Returns a boolean mask over tokens,
    True where the token is masked before optical encoding.
    """
    batch, tokens = token_cloud.flatten(1).shape
    total = int(round(mask_ratio * tokens))
    n_cloud = int(round(cloud_fraction * total))
    n_struct = int(round(struct_fraction * total))
    n_random = max(0, total - n_cloud - n_struct)

    cloud_score = token_cloud.flatten(1)
    struct_score = structural_salience(optical, token_size)
    random_score = torch.rand(batch, tokens, device=optical.device, generator=generator)

    taken = torch.zeros(batch, tokens, dtype=torch.bool, device=optical.device)
    for score, count in ((cloud_score, n_cloud), (struct_score, n_struct), (random_score, n_random)):
        if count <= 0:
            continue
        indices = _top_k_indices(score, count, taken)
        taken.scatter_(1, indices, True)
    return taken
