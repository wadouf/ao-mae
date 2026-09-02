#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import shutil
from pathlib import Path

import pandas as pd

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config

DESCRIPTION = 'Install annotation manifests and derive the annotation task table.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    src_manifest = root / 'reference_complements' / 'annotations' / 'manifests' / 'annotation_manifest.csv'
    src_qa = root / 'reference_complements' / 'annotations' / 'manifests' / 'annotation_quality_summary.csv'
    dst_dir = root / 'data' / 'annotations'
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_manifest, dst_dir / 'annotation_manifest.csv')
    shutil.copy2(src_qa, dst_dir / 'annotation_quality_summary.csv')
    df = pd.read_csv(src_manifest)
    tasks = df[['sample_id', 'annotator_a_id', 'annotator_b_id', 'adjudicator_id']].copy()
    tasks.to_csv(dst_dir / 'annotation_tasks.csv', index=False)
    write_json(dst_dir / 'annotation_tasks_summary.json', {'task_count': int(len(tasks)), 'mode': args.mode})
    print(dst_dir / 'annotation_tasks.csv')


if __name__ == '__main__':
    main()
