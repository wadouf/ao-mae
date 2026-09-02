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
from oamae_pipeline.reference_io import reference_samples

DESCRIPTION = 'Discover acquisitions and write an acquisition manifest.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    samples = reference_samples(root, cfg)
    rows = []
    if 'sample_id' not in samples.columns:
        samples = samples.rename(columns={samples.columns[0]: 'sample_id'})
    for _, row in samples.iterrows():
        sid = row['sample_id']
        city = sid.split('_')[1]
        rows.extend([
            {'sample_id': sid, 'city': city, 'timepoint': 'T1', 'sensor': 'S2', 'source_mode': args.mode},
            {'sample_id': sid, 'city': city, 'timepoint': 'T2', 'sensor': 'S2', 'source_mode': args.mode},
            {'sample_id': sid, 'city': city, 'timepoint': 'T1', 'sensor': 'S1', 'source_mode': args.mode},
            {'sample_id': sid, 'city': city, 'timepoint': 'T2', 'sensor': 'S1', 'source_mode': args.mode},
            {'sample_id': sid, 'city': city, 'timepoint': 'T1', 'sensor': 'CLOUD', 'source_mode': args.mode},
            {'sample_id': sid, 'city': city, 'timepoint': 'T2', 'sensor': 'CLOUD', 'source_mode': args.mode},
        ])
    out = root / 'outputs' / f'{args.mode}_run' / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'acquisition_manifest.csv', index=False)
    write_json(out / 'acquisition_discovery_summary.json', {'rows': len(rows), 'sample_count': int(len(samples)), 'mode': args.mode})
    print(out / 'acquisition_manifest.csv')


if __name__ == '__main__':
    main()
