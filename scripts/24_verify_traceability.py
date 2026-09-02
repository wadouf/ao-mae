#!/usr/bin/env python3
"""Check that every published value of a run recomputes from the archived arrays.

The chain is verified link by link: thresholding, support identity, metric identity,
aggregation identity, and the manuscript value derivation. Each link reports the
number of records checked so that a partial run is never mistaken for a full one.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

import numpy as np
import pandas as pd

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.metrics import compute
from oamae_pipeline.observability import compute_support

DESCRIPTION = 'Verify that predictions, support, metrics, tables and manuscript values are mutually recomputable.'
TOLERANCE = 1e-9


def check_thresholding(manifest: pd.DataFrame, root: Path, threshold: float) -> dict:
    checked = failures = 0
    examples = []
    for _, row in manifest.iterrows():
        with np.load(root / row['prediction_path']) as data:
            if 'binary' not in data.files:
                continue
            expected = (data['probability'] >= threshold).astype(np.uint8)
            checked += 1
            if not np.array_equal(expected, data['binary'].astype(np.uint8)):
                failures += 1
                if len(examples) < 5:
                    examples.append(row['prediction_path'])
    return {'checked': checked, 'failures': failures, 'examples': examples}


def check_support(manifest: pd.DataFrame, root: Path, bundles: Path, cfg: dict) -> dict:
    support_cfg = cfg['cloud_support']
    checked = failures = 0
    examples = []
    for sample_id in sorted(manifest['sample_id'].unique()):
        bundle_path = bundles / f'{sample_id}.npz'
        rows = manifest[manifest['sample_id'] == sample_id]
        if not bundle_path.exists():
            continue
        with np.load(bundle_path) as data:
            if 'cloud_t1' not in data.files or 'cloud_t2' not in data.files:
                continue
            *_, expected = compute_support(
                data['cloud_t1'], data['cloud_t2'],
                token_size=int(support_cfg['token_size_pixels']),
                seed_threshold=float(support_cfg['seed_threshold']),
                delta=float(support_cfg['refinement_delta']),
                hard_threshold=float(support_cfg['hard_threshold']),
            )
        for _, row in rows.iterrows():
            with np.load(root / row['prediction_path']) as data:
                if 'v12' not in data.files:
                    continue
                checked += 1
                if not np.array_equal(expected.astype(bool), data['v12'].astype(bool)):
                    failures += 1
                    if len(examples) < 5:
                        examples.append(row['prediction_path'])
    return {'checked': checked, 'failures': failures, 'examples': examples}


def check_metrics(metrics: pd.DataFrame, manifest: pd.DataFrame, root: Path, bundles: Path, threshold: float) -> dict:
    lookup = manifest.set_index(['sample_id', 'method'])['prediction_path'].to_dict()
    checked = failures = 0
    examples = []
    for _, row in metrics.iterrows():
        key = (row['sample_id'], row['method'])
        bundle_path = bundles / f"{row['sample_id']}.npz"
        if key not in lookup or not bundle_path.exists():
            continue
        with np.load(bundle_path) as data:
            if 'reference' not in data.files:
                continue
            reference = data['reference'].astype(np.uint8)
        with np.load(root / lookup[key]) as data:
            recomputed = compute(reference, data['probability'].astype(np.float32), data['v12'].astype(np.uint8), threshold)
        checked += 1
        for field in ('tp', 'fp', 'fn', 'tn'):
            if field in row and int(recomputed[field]) != int(row[field]):
                failures += 1
                if len(examples) < 5:
                    examples.append({'sample_id': row['sample_id'], 'method': row['method'], 'field': field,
                                     'recomputed': int(recomputed[field]), 'table': int(row[field])})
                break
        else:
            if 'iou' in row and abs(float(recomputed['iou']) - float(row['iou'])) > 1e-6:
                failures += 1
                if len(examples) < 5:
                    examples.append({'sample_id': row['sample_id'], 'method': row['method'], 'field': 'iou',
                                     'recomputed': float(recomputed['iou']), 'table': float(row['iou'])})
    return {'checked': checked, 'failures': failures, 'examples': examples}


def check_aggregation(results: Path) -> dict:
    runs_path, summary_path = results / 'benchmark_main_runs.csv', results / 'benchmark_main_summary.csv'
    if not runs_path.exists() or not summary_path.exists():
        return {'checked': 0, 'failures': 0, 'note': 'benchmark tables absent'}
    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)
    recomputed = runs.groupby(['method', 'K'], as_index=False)['iou'].mean().rename(columns={'iou': 'iou_recomputed'})
    merged = summary.merge(recomputed, on=['method', 'K'], how='left')
    deviation = (merged['iou_mean'] - merged['iou_recomputed']).abs()
    return {
        'checked': int(len(merged)),
        'failures': int((deviation > 1e-6).sum()),
        'maximum_deviation': float(deviation.max()) if len(merged) else 0.0,
    }


def check_manuscript_values(root: Path) -> dict:
    values_path = root / 'manuscript' / 'results' / 'values.json'
    if not values_path.exists():
        return {'checked': 0, 'failures': 0, 'note': 'manuscript values absent'}
    import subprocess
    before = values_path.read_text(encoding='utf-8')
    result = subprocess.run([sys.executable, str(root / 'manuscript' / 'scripts' / 'build_values.py')],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return {'checked': 1, 'failures': 1, 'note': result.stderr.strip()[:300]}
    after = values_path.read_text(encoding='utf-8')
    return {'checked': len(json.loads(after)), 'failures': 0 if before == after else 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='observed')
    parser.add_argument('--bundles', default='data/processed/scene_bundles')
    parser.add_argument('--results', default=None, help='defaults to outputs/<mode>_run/results')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    root = Path('.').resolve()
    cfg = load_config(args.config)
    threshold = float(cfg['inference']['probability_threshold'])
    bundles = Path(args.bundles)
    results = Path(args.results) if args.results else root / 'outputs' / f'{args.mode}_run' / 'results'

    manifest_path = root / 'data' / 'predictions' / args.mode / 'prediction_manifest.csv'
    metrics_path = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'pixel_metrics.csv'
    if not manifest_path.exists():
        raise SystemExit(f'No prediction manifest at {manifest_path}; run stage 10 first.')
    manifest = pd.read_csv(manifest_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    checks = {
        'thresholding': check_thresholding(manifest, root, threshold),
        'support_identity': check_support(manifest, root, bundles, cfg),
        'metric_identity': check_metrics(metrics, manifest, root, bundles, threshold) if len(metrics) else {'checked': 0, 'failures': 0, 'note': 'pixel metrics absent'},
        'aggregation_identity': check_aggregation(results),
        'manuscript_values': check_manuscript_values(root),
    }
    for check in checks.values():
        check['status'] = 'PASS' if check['failures'] == 0 and check['checked'] > 0 else ('FAIL' if check['failures'] else 'NOT_CHECKED')

    report = {
        'mode': args.mode,
        'status': 'PASS' if all(c['status'] == 'PASS' for c in checks.values()) else 'INCOMPLETE' if not any(c['status'] == 'FAIL' for c in checks.values()) else 'FAIL',
        'checks': checks,
    }
    output = Path(args.output) if args.output else root / 'outputs' / f'{args.mode}_run' / 'TRACEABILITY.json'
    write_json(output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
