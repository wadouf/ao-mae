#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

from oamae_pipeline.common import utc_now, write_json
from oamae_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checks = {
        "time": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "earth_engine_project_present": bool(os.environ.get("EARTHENGINE_PROJECT")),
        "output_root": cfg["project"]["output_root"],
    }
    required = ["numpy", "pandas", "geopandas", "rasterio", "ee", "torch", "matplotlib"]
    checks["imports"] = {}
    for name in required:
        try:
            importlib.import_module(name)
            checks["imports"][name] = "ok"
        except Exception as exc:
            checks["imports"][name] = f"error: {exc}"
    output = Path(cfg["project"]["output_root"]) / "logs" / "preflight.json"
    write_json(output, checks)
    failed = [name for name, status in checks["imports"].items() if status != "ok"]
    if failed:
        raise SystemExit(f"Missing imports: {failed}")
    print(output)

if __name__ == "__main__":
    main()
