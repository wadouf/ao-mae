#!/usr/bin/env python3
"""Aggregate per-sample metrics into the benchmark tables that carry the reported results."""
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
from oamae_pipeline.reference_io import sample_city

DESCRIPTION = 'Build benchmark_main_runs, benchmark_main_summary and the primary comparison from per-sample metrics.'
PRIMARY = ('OA_MAE', 'CROMA')


def city_names(root: Path) -> dict[str, str]:
    features = json.loads((root / 'config' / 'cities.geojson').read_text(encoding='utf-8'))['features']
    return {f['properties']['code']: f['properties']['city'].replace(' ', '_') for f in features}


def city_cluster_bootstrap(values_by_city: dict[str, np.ndarray], draws: int, level: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Nonparametric bootstrap resampling cities, the primary statistical unit."""
    cities = sorted(values_by_city)
    means = np.array([values_by_city[city].mean() for city in cities])
    if len(cities) == 1:
        value = float(means[0])
        return value, value, value
    index = rng.integers(0, len(cities), size=(draws, len(cities)))
    samples = means[index].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(means.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='observed')
    parser.add_argument('--output', default=None, help='defaults to outputs/<mode>_run/results')
    args = parser.parse_args()

    root = Path('.').resolve()
    cfg = load_config(args.config)
    statistics = cfg.get('statistics', {})
    draws = int(statistics.get('bootstrap_replicates', 50000))
    level = float(statistics.get('confidence_level', 0.95))
    rng = np.random.default_rng(int(cfg['project']['random_seed']))

    tables = root / 'outputs' / f'{args.mode}_run' / 'tables'
    metrics = pd.read_csv(tables / 'pixel_metrics.csv')
    manifest = pd.read_csv(root / 'data' / 'predictions' / args.mode / 'prediction_manifest.csv')

    columns = ['sample_id', 'method'] + [c for c in ('few_shot_k', 'seed') if c in manifest.columns]
    merged = metrics.merge(manifest[columns], on=['sample_id', 'method'], how='left')
    if 'few_shot_k' not in merged.columns or merged['few_shot_k'].isna().all():
        raise SystemExit('The prediction manifest carries no few_shot_k or seed; rerun stage 10 first.')

    names = city_names(root)
    merged['city'] = [names.get(sample_city(s), sample_city(s)) for s in merged['sample_id']]
    merged = merged.rename(columns={'few_shot_k': 'K'})
    merged['K'] = merged['K'].astype(int)
    merged['seed'] = merged['seed'].astype(int)

    runs = merged.groupby(['city', 'K', 'seed', 'method'], as_index=False)[['iou', 'f1', 'auprc']].mean()
    output = Path(args.output) if args.output else root / 'outputs' / f'{args.mode}_run' / 'results'
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / 'benchmark_main_runs.csv', index=False)

    summary_rows = []
    for (method, k), group in runs.groupby(['method', 'K']):
        by_city = {city: sub['iou'].to_numpy() for city, sub in group.groupby('city')}
        mean, low, high = city_cluster_bootstrap(by_city, draws, level, rng)
        summary_rows.append({
            'method': method, 'K': int(k),
            'iou_mean': mean, 'iou_ci_low': low, 'iou_ci_high': high,
            'f1_mean': float(group['f1'].mean()), 'auprc_mean': float(group['auprc'].mean()),
            'inference_unit': statistics.get('primary_cluster', 'city'),
            'n_cities': int(group['city'].nunique()),
            'n_seed_runs': int(len(group)),
            'bootstrap_draws': draws,
            'bootstrap_type': 'city_cluster_nonparametric',
        })
    summary = pd.DataFrame(summary_rows).sort_values(['method', 'K'])
    summary.to_csv(output / 'benchmark_main_summary.csv', index=False)

    treatment, control = PRIMARY
    primary_k = int(sorted(runs['K'].unique())[len(sorted(runs['K'].unique())) // 2])
    paired = runs[runs['K'] == primary_k].pivot_table(index=['city', 'seed'], columns='method', values='iou')
    if treatment in paired.columns and control in paired.columns:
        paired = paired.dropna(subset=[treatment, control])
        delta = paired[treatment] - paired[control]
        by_city = {city: sub.to_numpy() for city, sub in delta.groupby(level=0)}
        mean, low, high = city_cluster_bootstrap(by_city, draws, level, rng)
        city_means = delta.groupby(level=0).mean()
        write_json(output / 'benchmark_primary_comparison.json', {
            'treatment': treatment, 'control': control, 'K': primary_k,
            'delta_iou_mean': mean, 'delta_iou_ci_low': low, 'delta_iou_ci_high': high,
            'fraction_positive_city_means': float((city_means > 0).mean()),
            'fraction_positive_paired_seed_runs': float((delta > 0).mean()),
            'inference_unit': statistics.get('primary_cluster', 'city'),
            'n_cities': int(city_means.size),
            'n_paired_seed_runs': int(delta.size),
            'bootstrap_draws': draws,
            'bootstrap_type': 'city_cluster_nonparametric',
        })

    write_json(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'aggregation_summary.json', {
        'mode': args.mode,
        'sample_metric_rows': int(len(merged)),
        'run_rows': int(len(runs)),
        'summary_rows': int(len(summary)),
        'primary_k': primary_k,
    })
    print(output / 'benchmark_main_summary.csv')


if __name__ == '__main__':
    main()
