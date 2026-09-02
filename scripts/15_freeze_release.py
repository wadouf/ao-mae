#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zipfile
from pathlib import Path

from oamae_pipeline.common import write_json
from oamae_pipeline.validation import build_checksum_manifest
from oamae_pipeline.config import load_config

DESCRIPTION = 'Freeze the release directory into a standalone archive with checksums.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    run_dir = root / 'outputs' / f'{args.mode}_run'
    release_dir = run_dir / 'release'
    release_dir.mkdir(parents=True, exist_ok=True)
    build_checksum_manifest(run_dir, release_dir / 'SHA256SUMS.csv')
    zip_path = release_dir / f'oamae_{args.mode}_run_release.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob('*')):
            if p.is_file() and p != zip_path:
                zf.write(p, p.relative_to(run_dir))
    write_json(release_dir / 'freeze_summary.json', {'zip_path': str(zip_path), 'mode': args.mode})
    print(zip_path)


if __name__ == '__main__':
    main()
