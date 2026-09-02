from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import OAMAEConfig

STAGE2_KEYS = ('optical_t1', 'optical_t2', 'radar_t1', 'radar_t2', 'cloud_t1', 'cloud_t2')
STAGE1_KEYS = ('optical', 'radar', 'cloud', 'history_optical', 'history_radar', 'history_cloud', 'history_ages_days')


class BundleContractError(ValueError):
    pass


@dataclass
class Normalization:
    """Per-band statistics frozen before training and hashed into the checkpoint record."""

    optical_mean: np.ndarray
    optical_std: np.ndarray
    radar_mean: np.ndarray
    radar_std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> 'Normalization':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            optical_mean=np.asarray(payload['optical_mean'], dtype=np.float32),
            optical_std=np.asarray(payload['optical_std'], dtype=np.float32),
            radar_mean=np.asarray(payload['radar_mean'], dtype=np.float32),
            radar_std=np.asarray(payload['radar_std'], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            'optical_mean': self.optical_mean.tolist(),
            'optical_std': self.optical_std.tolist(),
            'radar_mean': self.radar_mean.tolist(),
            'radar_std': self.radar_std.tolist(),
        }, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    def apply_optical(self, array: np.ndarray) -> np.ndarray:
        mean = self.optical_mean.reshape(-1, 1, 1)
        std = np.maximum(self.optical_std.reshape(-1, 1, 1), 1e-6)
        return ((array - mean) / std).astype(np.float32)

    def apply_radar(self, array: np.ndarray) -> np.ndarray:
        mean = self.radar_mean.reshape(-1, 1, 1)
        std = np.maximum(self.radar_std.reshape(-1, 1, 1), 1e-6)
        return ((array - mean) / std).astype(np.float32)


def identity_normalization(cfg: OAMAEConfig) -> Normalization:
    return Normalization(
        optical_mean=np.zeros(cfg.optical_bands, dtype=np.float32),
        optical_std=np.ones(cfg.optical_bands, dtype=np.float32),
        radar_mean=np.zeros(cfg.radar_bands, dtype=np.float32),
        radar_std=np.ones(cfg.radar_bands, dtype=np.float32),
    )


def _require(bundle: dict[str, np.ndarray], keys: Sequence[str], path: Path) -> None:
    missing = [key for key in keys if key not in bundle]
    if missing:
        raise BundleContractError(
            f'{path.name} does not satisfy the bundle contract: missing {missing}. '
            'Expected the observed-mode arrays described in MODEL_ADAPTER_CONTRACT.md.'
        )


def _check_shape(array: np.ndarray, expected: tuple[int, ...], name: str, path: Path) -> None:
    if array.shape[-2:] != expected[-2:] or (len(expected) == 3 and array.shape[0] != expected[0]):
        raise BundleContractError(f'{path.name}: {name} has shape {array.shape}, expected {expected}')


def compute_normalization(paths: Iterable[Path], cfg: OAMAEConfig) -> Normalization:
    """Streaming per-band mean and standard deviation over the training pool."""
    optical_sum = np.zeros(cfg.optical_bands, dtype=np.float64)
    optical_square = np.zeros(cfg.optical_bands, dtype=np.float64)
    radar_sum = np.zeros(cfg.radar_bands, dtype=np.float64)
    radar_square = np.zeros(cfg.radar_bands, dtype=np.float64)
    optical_count = radar_count = 0

    for path in paths:
        with np.load(path) as data:
            bundle = {key: data[key] for key in data.files}
        for key in ('optical_t1', 'optical_t2', 'optical'):
            if key in bundle:
                array = bundle[key].astype(np.float64)
                optical_sum += array.sum(axis=(1, 2))
                optical_square += (array ** 2).sum(axis=(1, 2))
                optical_count += array.shape[1] * array.shape[2]
        for key in ('radar_t1', 'radar_t2', 'radar'):
            if key in bundle:
                array = bundle[key].astype(np.float64)
                radar_sum += array.sum(axis=(1, 2))
                radar_square += (array ** 2).sum(axis=(1, 2))
                radar_count += array.shape[1] * array.shape[2]

    if not optical_count or not radar_count:
        raise BundleContractError('No optical or radar arrays were found while computing normalization')

    optical_mean = optical_sum / optical_count
    radar_mean = radar_sum / radar_count
    return Normalization(
        optical_mean=optical_mean.astype(np.float32),
        optical_std=np.sqrt(np.maximum(optical_square / optical_count - optical_mean ** 2, 0)).astype(np.float32),
        radar_mean=radar_mean.astype(np.float32),
        radar_std=np.sqrt(np.maximum(radar_square / radar_count - radar_mean ** 2, 0)).astype(np.float32),
    )


class ChangeDataset(Dataset):
    """Stage II: bi-temporal pairs with the reference mask and the cloud product."""

    def __init__(self, paths: Sequence[Path], cfg: OAMAEConfig, normalization: Normalization | None = None, with_reference: bool = True) -> None:
        self.paths = [Path(p) for p in paths]
        self.cfg = cfg
        self.normalization = normalization or identity_normalization(cfg)
        self.with_reference = with_reference

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path) as data:
            bundle = {key: data[key] for key in data.files}
        keys = STAGE2_KEYS + (('reference',) if self.with_reference else ())
        _require(bundle, keys, path)

        size = self.cfg.image_size
        for key in ('optical_t1', 'optical_t2'):
            _check_shape(bundle[key], (self.cfg.optical_bands, size, size), key, path)
        for key in ('radar_t1', 'radar_t2'):
            _check_shape(bundle[key], (self.cfg.radar_bands, size, size), key, path)

        item: dict[str, torch.Tensor | str] = {'sample_id': path.stem}
        for key in ('optical_t1', 'optical_t2'):
            item[key] = torch.from_numpy(self.normalization.apply_optical(bundle[key].astype(np.float32)))
        for key in ('radar_t1', 'radar_t2'):
            item[key] = torch.from_numpy(self.normalization.apply_radar(bundle[key].astype(np.float32)))
        for key in ('cloud_t1', 'cloud_t2'):
            item[key] = torch.from_numpy(bundle[key].astype(np.float32))
        if self.with_reference:
            item['reference'] = torch.from_numpy(bundle['reference'].astype(np.float32))
        return item


class PretrainDataset(Dataset):
    """Stage I: a current acquisition with its bounded past-only history."""

    def __init__(self, paths: Sequence[Path], cfg: OAMAEConfig, normalization: Normalization | None = None) -> None:
        self.paths = [Path(p) for p in paths]
        self.cfg = cfg
        self.normalization = normalization or identity_normalization(cfg)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path) as data:
            bundle = {key: data[key] for key in data.files}
        _require(bundle, STAGE1_KEYS, path)

        size = self.cfg.image_size
        _check_shape(bundle['optical'], (self.cfg.optical_bands, size, size), 'optical', path)
        _check_shape(bundle['radar'], (self.cfg.radar_bands, size, size), 'radar', path)

        history_optical = np.stack([self.normalization.apply_optical(frame.astype(np.float32)) for frame in bundle['history_optical']])
        history_radar = np.stack([self.normalization.apply_radar(frame.astype(np.float32)) for frame in bundle['history_radar']])

        return {
            'sample_id': path.stem,
            'optical': torch.from_numpy(self.normalization.apply_optical(bundle['optical'].astype(np.float32))),
            'radar': torch.from_numpy(self.normalization.apply_radar(bundle['radar'].astype(np.float32))),
            'cloud': torch.from_numpy(bundle['cloud'].astype(np.float32)),
            'history_optical': torch.from_numpy(history_optical),
            'history_radar': torch.from_numpy(history_radar),
            'history_cloud': torch.from_numpy(bundle['history_cloud'].astype(np.float32)),
            'ages_days': torch.from_numpy(bundle['history_ages_days'].astype(np.float32)),
        }


def collate(items: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    batch: dict[str, torch.Tensor | list[str]] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        batch[key] = values if isinstance(values[0], str) else torch.stack(values)
    return batch
