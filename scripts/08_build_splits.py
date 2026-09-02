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
from oamae_pipeline.reference_io import reference_samples, sample_city

DESCRIPTION = 'Build leave-one-city-out splits and few-shot seed manifests.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    samples = reference_samples(root, cfg)
    sample_ids = list(samples['sample_id']) if 'sample_id' in samples.columns else list(samples.iloc[:,0])
    cities = sorted({sample_city(s) for s in sample_ids})
    rows = []
    for left_out in cities:
        train = [s for s in sample_ids if sample_city(s) != left_out]
        test = [s for s in sample_ids if sample_city(s) == left_out]
        for k in cfg['splits']['few_shot_k']:
            for seed in cfg['splits']['seeds']:
                rows.append({'left_out_city': left_out, 'few_shot_k': k, 'seed': seed, 'train_count': len(train), 'test_count': len(test)})
    out = root / 'outputs' / f'{args.mode}_run' / 'tables' / 'split_manifest.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    write_json(out.with_suffix('.json'), {'configuration_count': len(rows), 'cities': cities})
    print(out)


if __name__ == '__main__':
    main()
