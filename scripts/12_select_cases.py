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

DESCRIPTION = 'Select deterministic figure cases from metrics and support summaries.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    bundle_dir = root / 'data' / 'processed' / 'scene_bundles'
    rows = []
    for npz_path in sorted(bundle_dir.glob('SCN_*.npz')):
        with np.load(npz_path) as d:
            sid = npz_path.stem
            mean_cloud = float(0.5 * (d['cloud_t1'].mean() + d['cloud_t2'].mean()))
            support = float(d['v12'].mean()) if 'v12' in d.files else None
            rows.append({'sample_id': sid, 'mean_cloud': mean_cloud, 'native_support_coverage': support})
    df = pd.DataFrame(rows).sort_values(['mean_cloud', 'sample_id'], ascending=[False, True])
    severe = df.head(6).copy(); severe['figure_id'] = 'QF1'
    atlas = df.groupby(df['sample_id'].str.split('_').str[1]).head(2).copy(); atlas['figure_id'] = 'QF7'
    selected = pd.concat([severe, atlas], ignore_index=True)
    out = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'selected_cases.csv'
    selected.to_csv(out, index=False)
    write_json(out.with_suffix('.json'), {'selected_rows': int(selected.shape[0])})
    print(out)


if __name__ == '__main__':
    main()
