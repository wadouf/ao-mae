#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    '01_resolve_areas.py',
    '02_discover_acquisitions.py',
    '03_build_pair_candidates.py',
    '04_export_rasters.py',
    '05_compute_support.py',
    '06_prepare_annotations.py',
    '07_validate_annotations.py',
    '08_build_splits.py',
    '09_train_or_load_models.py',
    '10_run_inference.py',
    '11_compute_metrics.py',
    '12_select_cases.py',
    '13_render_figures.py',
    '14_validate_release.py',
    '15_freeze_release.py',
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full OA-MAE figure pipeline.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for stage in STAGES:
        cmd = [sys.executable, str(root / stage), '--config', args.config, '--mode', args.mode]
        print('Running', ' '.join(cmd))
        subprocess.run(cmd, check=True)
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
