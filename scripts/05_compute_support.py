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
from oamae_pipeline.observability import compute_support

DESCRIPTION = 'Compute refined cloud support and V12 from cloud probability rasters.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    in_dir = root / 'data' / 'processed' / 'scene_bundles'
    out_dir = root / 'data' / 'processed' / 'support'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for npz_path in sorted(in_dir.glob('SCN_*.npz')):
        with np.load(npz_path) as d:
            p1, p2, m1, m2, v12 = compute_support(
                d['cloud_t1'].astype(np.float32), d['cloud_t2'].astype(np.float32),
                token_size=cfg['cloud_support']['token_size_pixels'],
                seed_threshold=cfg['cloud_support']['seed_threshold'],
                delta=cfg['cloud_support']['refinement_delta'],
                hard_threshold=cfg['cloud_support']['hard_threshold'],
            )
        np.savez_compressed(out_dir / f'{npz_path.stem}_support.npz', cloud_refined_t1=p1, cloud_refined_t2=p2, m_t1=m1, m_t2=m2, v12=v12)
        rows.append({'sample_id': npz_path.stem, 'coverage': float(v12.mean()), 'm_t1_coverage': float(m1.mean()), 'm_t2_coverage': float(m2.mean())})
    pd.DataFrame(rows).to_csv(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'support_summary.csv', index=False)
    write_json(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'support_summary.json', {'sample_count': len(rows), 'mode': args.mode})
    print(out_dir)


if __name__ == '__main__':
    main()
