from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import OAMAEConfig
from .segmentation import OAMAEChangeDetector

REQUIRED_KEYS = ('optical_t1', 'optical_t2', 'radar_t1', 'radar_t2', 'cloud_t1', 'cloud_t2')
DIAGNOSTIC_KEYS = ('cloud_gate', 'radar_reliability', 'effective_gate')

_CACHE: dict[tuple[str, str], OAMAEChangeDetector] = {}


def build_config(config: dict[str, Any] | None) -> OAMAEConfig:
    """Map the pipeline configuration onto the model hyperparameters."""
    cfg = OAMAEConfig()
    if not config:
        return cfg
    support = config.get('cloud_support', {})
    cfg.token_size = int(support.get('token_size_pixels', cfg.token_size))
    cfg.seed_threshold = float(support.get('seed_threshold', cfg.seed_threshold))
    cfg.refinement_delta = float(support.get('refinement_delta', cfg.refinement_delta))
    cfg.hard_threshold = float(support.get('hard_threshold', cfg.hard_threshold))

    spatial = config.get('spatial', {})
    cfg.image_size = int(spatial.get('tile_size_pixels', cfg.image_size))

    sentinel2 = config.get('sentinel2', {})
    bands = sentinel2.get('bands')
    if bands:
        cfg.optical_bands = len(bands)

    retrieval = config.get('retrieval', {})
    cfg.maximum_age_days = int(retrieval.get('maximum_age_days', cfg.maximum_age_days))
    cfg.maximum_candidates = int(retrieval.get('maximum_candidates_per_token', cfg.maximum_candidates))
    cfg.safety_floor = float(retrieval.get('safety_floor', cfg.safety_floor))
    cfg.safety_coefficient = float(retrieval.get('safety_coefficient', cfg.safety_coefficient))

    inference = config.get('inference', {})
    cfg.probability_threshold = float(inference.get('probability_threshold', cfg.probability_threshold))
    return cfg


def load_model(checkpoint: str | Path, config: dict[str, Any] | None = None, device: str | torch.device = 'cpu') -> OAMAEChangeDetector:
    """Load a Stage-II checkpoint. A checkpoint stores its own configuration when available."""
    path = Path(checkpoint)
    key = (str(path.resolve()), str(device))
    if key in _CACHE:
        return _CACHE[key]

    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and 'state_dict' in payload:
        state = payload['state_dict']
        stored = payload.get('config')
        cfg = OAMAEConfig(**stored) if isinstance(stored, dict) else build_config(config)
    else:
        state = payload
        cfg = build_config(config)

    model = OAMAEChangeDetector(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f'Checkpoint {path} is missing parameters: {sorted(missing)[:8]}')
    if unexpected:
        raise RuntimeError(f'Checkpoint {path} has unexpected parameters: {sorted(unexpected)[:8]}')
    model.to(device).eval()
    _CACHE[key] = model
    return model


def _tensor(value: Any, device: str | torch.device, channels: int | None) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    tensor = torch.from_numpy(array)
    if channels is None:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
    else:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def predict_batch(*, batch: dict[str, Any], checkpoint: str, config: dict[str, Any] | None = None, device: str | None = None) -> dict[str, np.ndarray]:
    """Adapter entry point declared in config/model_adapters.yaml.

    Returns the change probability on the full grid together with the internal
    gates produced during inference. The external support V12 is computed from the
    cloud product, never from model confidence.
    """
    missing = [key for key in REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError(f'Batch is missing required arrays: {missing}')

    target_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint, config, target_device)

    single = np.asarray(batch['cloud_t1']).ndim == 2
    inputs = {
        'optical_t1': _tensor(batch['optical_t1'], target_device, model.cfg.optical_bands),
        'optical_t2': _tensor(batch['optical_t2'], target_device, model.cfg.optical_bands),
        'radar_t1': _tensor(batch['radar_t1'], target_device, model.cfg.radar_bands),
        'radar_t2': _tensor(batch['radar_t2'], target_device, model.cfg.radar_bands),
        'cloud_t1': _tensor(batch['cloud_t1'], target_device, None),
        'cloud_t2': _tensor(batch['cloud_t2'], target_device, None),
    }

    with torch.no_grad():
        outputs = model(**inputs)

    def to_numpy(value: torch.Tensor, dtype: type) -> np.ndarray:
        array = value.detach().cpu().numpy().astype(dtype)
        return array[0] if single else array

    result: dict[str, np.ndarray] = {
        'probability': to_numpy(outputs['probability'], np.float32),
        'v12': to_numpy(outputs['v12'], np.uint8),
    }
    for key in DIAGNOSTIC_KEYS:
        result[key] = to_numpy(outputs[key], np.float32)
    return result
