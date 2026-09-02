from __future__ import annotations

import numpy as np


def pool_tokens(cloud: np.ndarray, token_size: int) -> np.ndarray:
    height, width = cloud.shape
    if height % token_size or width % token_size:
        raise ValueError("Cloud array dimensions must be divisible by token size")
    return cloud.reshape(height // token_size, token_size, width // token_size, token_size).mean(axis=(1, 3))


def refine_tokens(token_cloud: np.ndarray, seed_threshold: float, delta: float) -> np.ndarray:
    return np.minimum(1.0, token_cloud + delta * token_cloud * (token_cloud >= seed_threshold)).astype(np.float32)


def broadcast_tokens(tokens: np.ndarray, token_size: int) -> np.ndarray:
    return np.repeat(np.repeat(tokens, token_size, axis=0), token_size, axis=1)


def compute_support(cloud_t1: np.ndarray, cloud_t2: np.ndarray, token_size: int = 16, seed_threshold: float = 0.20, delta: float = 0.30, hard_threshold: float = 0.85):
    t1 = refine_tokens(pool_tokens(cloud_t1, token_size), seed_threshold, delta)
    t2 = refine_tokens(pool_tokens(cloud_t2, token_size), seed_threshold, delta)
    p1 = broadcast_tokens(t1, token_size)
    p2 = broadcast_tokens(t2, token_size)
    m1 = p1 <= hard_threshold
    m2 = p2 <= hard_threshold
    return p1.astype(np.float32), p2.astype(np.float32), m1.astype(np.uint8), m2.astype(np.uint8), (m1 & m2).astype(np.uint8)
