#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from pathlib import Path

import pandas as pd

from oamae_pipeline.adapters import checkpoint_path, resolve_all
from oamae_pipeline.common import git_commit, sha256_file, write_json
from oamae_pipeline.config import load_config

DESCRIPTION = 'Resolve declared model adapters and record the identity of every checkpoint they will use.'


def city_codes(root: Path) -> list[str]:
    features = json.loads((root / 'config' / 'cities.geojson').read_text(encoding='utf-8'))['features']
    return [feature['properties']['code'] for feature in features]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapters', default='config/model_adapters.yaml')
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    methods = resolve_all(args.adapters)

    cities = city_codes(root)
    ks = cfg['splits']['few_shot_k']
    seeds = cfg['splits']['seeds']

    rows = []
    blocked = []
    for name, method in methods.items():
        if method.kind == 'internal_reference':
            rows.append({
                'method': name,
                'adapter': method.kind,
                'source': 'archived_prediction_arrays',
                'repository': '',
                'repository_commit': '',
                'checkpoint_path': '',
                'checkpoint_sha256': '',
                'left_out_city': '',
                'few_shot_k': '',
                'seed': '',
            })
            continue
        if not method.available:
            blocked.append({'method': name, 'missing': method.missing})
            continue
        commit = git_commit(method.repository) if method.repository else 'unavailable'
        for city in cities:
            for k in ks:
                for seed in seeds:
                    path = checkpoint_path(method, city, k, seed)
                    if not path.exists():
                        blocked.append({'method': name, 'missing': [f'checkpoint not found: {path}']})
                        continue
                    rows.append({
                        'method': name,
                        'adapter': method.kind,
                        'source': method.callable_path,
                        'repository': str(method.repository),
                        'repository_commit': commit,
                        'checkpoint_path': str(path),
                        'checkpoint_sha256': sha256_file(path),
                        'left_out_city': city,
                        'few_shot_k': k,
                        'seed': seed,
                    })

    out_dir = root / 'outputs' / f'{args.mode}_run' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / 'checkpoint_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    write_json(out_dir / 'checkpoint_resolution.json', {
        'mode': args.mode,
        'resolved_records': len(rows),
        'blocked': blocked,
    })

    if blocked:
        write_json(root / 'outputs' / f'{args.mode}_run' / 'BLOCKED.json', {
            'stage': '09_train_or_load_models',
            'reason': 'declared methods could not be resolved to an adapter and checkpoints',
            'blocked': blocked,
            'recovery': 'set the repository and checkpoint root environment variables in .env, or remove the method from config/model_adapters.yaml',
        })
        for item in blocked[:10]:
            print(f"BLOCKED {item['method']}: {'; '.join(item['missing'])}", file=sys.stderr)
        raise SystemExit(1)

    print(manifest)


if __name__ == '__main__':
    main()
