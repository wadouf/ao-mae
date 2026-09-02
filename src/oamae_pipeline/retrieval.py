from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .observability import pool_tokens


@dataclass
class RetrievalResult:
    selected_indices: np.ndarray
    selected_ages: np.ndarray
    target: np.ndarray
    fallback: np.ndarray


def build_past_targets(current_time: pd.Timestamp, history_times: list[pd.Timestamp], history_optical: np.ndarray, history_cloud: np.ndarray, token_size: int = 16, maximum_age_days: int = 90, maximum_candidates: int = 3, clear_threshold: float = 0.15) -> RetrievalResult:
    if history_optical.ndim != 4:
        raise ValueError("history_optical must be time, band, height, width")
    eligible = []
    for index, timestamp in enumerate(history_times):
        age = (current_time - timestamp).total_seconds() / 86400.0
        if 0 < age <= maximum_age_days:
            eligible.append((index, int(round(age))))
    height, width = history_cloud.shape[-2:]
    th, tw = height // token_size, width // token_size
    selected = np.full((maximum_candidates, th, tw), -1, dtype=np.int16)
    ages = np.full((maximum_candidates, th, tw), -1, dtype=np.int16)
    target = np.zeros((history_optical.shape[1], height, width), dtype=np.float32)
    fallback = np.ones((th, tw), dtype=np.uint8)
    if not eligible:
        return RetrievalResult(selected, ages, target, fallback)
    token_cloud = np.stack([pool_tokens(history_cloud[i], token_size) for i, _ in eligible])
    for y in range(th):
        for x in range(tw):
            ranked = []
            for local_index, (source_index, age) in enumerate(eligible):
                cloud_value = float(token_cloud[local_index, y, x])
                if cloud_value <= clear_threshold:
                    ranked.append((cloud_value, age, source_index))
            ranked.sort(key=lambda item: (item[0], item[1], item[2]))
            ranked = ranked[:maximum_candidates]
            if not ranked:
                continue
            fallback[y, x] = 0
            y0, y1 = y * token_size, (y + 1) * token_size
            x0, x1 = x * token_size, (x + 1) * token_size
            patches = []
            for rank, (_, age, source_index) in enumerate(ranked):
                selected[rank, y, x] = source_index
                ages[rank, y, x] = age
                patches.append(history_optical[source_index, :, y0:y1, x0:x1])
            target[:, y0:y1, x0:x1] = np.median(np.stack(patches), axis=0)
    return RetrievalResult(selected, ages, target, fallback)
