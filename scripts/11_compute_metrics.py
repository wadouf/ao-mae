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

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.metrics import compute

DESCRIPTION = 'Compute per-scene metrics and aggregated summaries.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    bundle_dir = root / 'data' / 'processed' / 'scene_bundles'
    pred_manifest = pd.read_csv(root / 'data' / 'predictions' / args.mode / 'prediction_manifest.csv')
    rows = []
    for _, row in pred_manifest.iterrows():
        sid = row['sample_id']
        bundle = dict(np.load(bundle_dir / f'{sid}.npz'))
        pred = dict(np.load(root / row['prediction_path']))
        metrics = compute(bundle['ground_truth'].astype(np.uint8), pred['probability'].astype(np.float32), pred['v12'].astype(np.uint8), cfg['inference']['probability_threshold'])
        metrics = {'sample_id': sid, 'method': row['method'], **metrics}
        rows.append(metrics)
    df = pd.DataFrame(rows)
    out_dir = root / 'outputs' / f'{args.mode}_run' / 'tables'
    df.to_csv(out_dir / 'pixel_metrics.csv', index=False)
    agg = df.groupby('method')[['iou', 'f1', 'auprc', 'precision', 'recall', 'coverage', 'positive_coverage', 'unresolved_positive_mass']].mean().reset_index()
    agg.to_csv(out_dir / 'metrics_by_method.csv', index=False)
    write_json(out_dir / 'metrics_summary.json', {'method_count': int(agg.shape[0]), 'sample_method_rows': int(df.shape[0])})
    print(out_dir / 'metrics_by_method.csv')


if __name__ == '__main__':
    main()
