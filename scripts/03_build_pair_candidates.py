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

DESCRIPTION = 'Build bi-temporal pair candidates from discovered acquisitions.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    in_csv = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'acquisition_manifest.csv'
    df = pd.read_csv(in_csv)
    rows = []
    for sid, grp in df.groupby('sample_id'):
        city = grp['city'].iloc[0]
        rows.append({'sample_id': sid, 'city': city, 'has_s2_t1': True, 'has_s2_t2': True, 'has_s1_t1': True, 'has_s1_t2': True, 'has_cloud_t1': True, 'has_cloud_t2': True})
    out = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'pair_candidates.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    write_json(out.with_suffix('.json'), {'pair_count': len(rows), 'mode': args.mode})
    print(out)


if __name__ == '__main__':
    main()
