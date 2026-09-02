#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathlib import Path

import numpy as np
import pandas as pd

from oamae_pipeline.adapters import checkpoint_path, register_repository, resolve_all
from oamae_pipeline.common import sha256_file, write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.model_interface import ModelAdapter
from oamae_pipeline.reference_io import load_archived_probability, sample_city

DESCRIPTION = 'Run model inference through the declared adapters, or replay archived prediction arrays for internal reference methods.'

DIAGNOSTIC_KEYS = ['cloud_gate', 'radar_reliability', 'effective_gate']


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapters', default='config/model_adapters.yaml')
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    parser.add_argument('--few-shot-k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    few_shot_k = args.few_shot_k if args.few_shot_k is not None else cfg['splits']['few_shot_k'][0]
    seed = args.seed if args.seed is not None else cfg['splits']['seeds'][0]
    threshold = cfg['inference']['probability_threshold']

    in_dir = root / 'data' / 'processed' / 'scene_bundles'
    support_dir = root / 'data' / 'processed' / 'support'
    out_dir = root / 'data' / 'predictions' / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = resolve_all(args.adapters)
    for method in methods.values():
        if method.kind == 'python_module':
            if not method.available:
                raise SystemExit(f"{method.name}: {'; '.join(method.missing)}")
            register_repository(method)

    rows = []
    gate_rows = []
    for npz_path in sorted(in_dir.glob('SCN_*.npz')):
        sid = npz_path.stem
        city = sample_city(sid)
        bundle = dict(np.load(npz_path))
        support = dict(np.load(support_dir / f'{sid}_support.npz'))
        for name, method in methods.items():
            diagnostics: dict[str, np.ndarray] = {}
            if method.kind == 'internal_reference':
                prob = load_archived_probability(bundle, name)
                source = 'archived_array'
                checkpoint = ''
                checkpoint_sha = ''
            else:
                path = checkpoint_path(method, city, few_shot_k, seed)
                if not path.exists():
                    raise SystemExit(f'{name}: checkpoint not found for {city} K{few_shot_k} seed{seed}: {path}')
                adapter = ModelAdapter(method.callable_path, path, cfg)
                result = adapter.predict(bundle)
                prob = result.probability
                diagnostics = result.diagnostics
                source = method.callable_path
                checkpoint = str(path)
                checkpoint_sha = sha256_file(path)

            binary = (prob >= threshold).astype(np.uint8)
            out_npz = out_dir / f'{sid}_{name}.npz'
            np.savez_compressed(out_npz, probability=prob, binary=binary, v12=support['v12'], **diagnostics)
            rows.append({
                'sample_id': sid,
                'method': name,
                'source': source,
                'checkpoint_path': checkpoint,
                'checkpoint_sha256': checkpoint_sha,
                'few_shot_k': few_shot_k if method.kind == 'python_module' else '',
                'seed': seed if method.kind == 'python_module' else '',
                'prediction_path': str(out_npz.relative_to(root)),
                'mean_probability': float(prob.mean()),
            })
            if diagnostics:
                gate_rows.append({
                    'sample_id': sid,
                    'method': name,
                    **{key: float(np.asarray(diagnostics[key]).mean()) for key in DIAGNOSTIC_KEYS if key in diagnostics},
                })

    pd.DataFrame(rows).to_csv(out_dir / 'prediction_manifest.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(out_dir / 'gate_manifest.csv', index=False)
    write_json(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'inference_summary.json', {
        'prediction_count': len(rows),
        'mode': args.mode,
        'few_shot_k': few_shot_k,
        'seed': seed,
    })
    print(out_dir / 'prediction_manifest.csv')


if __name__ == '__main__':
    main()
