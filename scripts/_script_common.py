#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

from oamae_pipeline.config import load_config


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    return parser.parse_args()
