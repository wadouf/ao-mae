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

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.reference_io import iter_reference_bundles

DESCRIPTION = 'Export or standardize rasters into raw and processed bundle locations.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    raw_dir = root / 'data' / 'raw' / args.mode
    proc_dir = root / 'data' / 'processed' / 'scene_bundles'
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sid, bundle in iter_reference_bundles(root, cfg):
        src = root / cfg['project']['reference_root'] / 'qualitative_benchmark' / 'oa_mae_revision' / 'data' / 'arrays' / f'{sid}.npz'
        shutil.copy2(src, raw_dir / src.name)
        shutil.copy2(src, proc_dir / src.name)
        count += 1
    write_json(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'raster_export_summary.json', {'bundle_count': count, 'mode': args.mode})
    print(proc_dir)


if __name__ == '__main__':
    main()
