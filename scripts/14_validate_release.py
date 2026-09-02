#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathlib import Path

import pandas as pd

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config

DESCRIPTION = 'Validate release completeness, required tables, and key figure outputs.'

REQUIRED = [
    'tables/aoi_manifest.csv',
    'tables/acquisition_manifest.csv',
    'tables/pair_candidates.csv',
    'tables/support_summary.csv',
    'tables/annotation_validation.csv',
    'tables/split_manifest.csv',
    'tables/metrics_by_method.csv',
    'tables/selected_cases.csv',
    'figures/QF1_severe_cloud_comparison.png',
    'figures/QF1_severe_cloud_comparison.pdf',
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    run_dir = root / 'outputs' / f'{args.mode}_run'
    missing = [rel for rel in REQUIRED if not (run_dir / rel).exists()]
    report = {'status': 'PASS' if not missing else 'FAIL', 'missing': missing, 'run_dir': str(run_dir)}
    write_json(run_dir / 'release' / 'validation_report.json', report)
    if missing:
        raise SystemExit(f'Missing release assets: {missing}')
    print(run_dir / 'release' / 'validation_report.json')


if __name__ == '__main__':
    main()
