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

DESCRIPTION = 'Validate annotation completeness and compute agreement summaries.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    manifest = pd.read_csv(root / 'data' / 'annotations' / 'annotation_manifest.csv')
    rows = []
    for _, row in manifest.iterrows():
        a = np.load(root / row['annotator_a_path'])['mask']
        b = np.load(root / row['annotator_b_path'])['mask']
        j = np.load(root / row['adjudication_path'])['mask']
        rows.append({
            'sample_id': row['sample_id'],
            'agreement': float((a == b).mean()),
            'a_vs_j': float((a == j).mean()),
            'b_vs_j': float((b == j).mean()),
            'all_same_shape': a.shape == b.shape == j.shape,
        })
    out = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'annotation_validation.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    write_json(out.with_suffix('.json'), {'sample_count': len(rows), 'mean_agreement': float(pd.DataFrame(rows)['agreement'].mean())})
    print(out)


if __name__ == '__main__':
    main()
