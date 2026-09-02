
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def find_root(config: dict) -> Path:
    return Path(config['project'].get('package_root', '.')).resolve() if 'package_root' in config.get('project', {}) else Path('.').resolve()


def reference_array_dir(root: Path, config: dict) -> Path:
    return root / config['project']['reference_root'] / 'qualitative_benchmark' / 'oa_mae_revision' / 'data' / 'arrays'


def reference_samples(root: Path, config: dict) -> pd.DataFrame:
    samples = root / config['project']['reference_root'] / 'qualitative_benchmark' / 'oa_mae_revision' / 'data' / 'samples.csv'
    if samples.exists():
        return pd.read_csv(samples)
    fallback = root / config['project']['reference_root'] / 'qualitative_benchmark' / 'results' / 'scene_manifest.csv'
    return pd.read_csv(fallback)


def iter_reference_bundles(root: Path, config: dict) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    for npz_path in sorted(reference_array_dir(root, config).glob('SCN_*.npz')):
        with np.load(npz_path) as data:
            yield npz_path.stem, {key: data[key] for key in data.files}


def sample_city(sample_id: str) -> str:
    parts = sample_id.split('_')
    return parts[1] if len(parts) > 2 else 'unknown'



def load_archived_probability(bundle: dict[str, np.ndarray], method: str) -> np.ndarray:
    name = method.lower()
    for key in (f'probability_{name}', f'probability_{name.replace("_", "")}'):
        if key in bundle:
            return bundle[key].astype(np.float32)
    raise KeyError(f'No archived probability array for method {method}')
