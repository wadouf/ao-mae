#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from oamae_pipeline.common import write_json
from oamae_pipeline.config import load_config

DESCRIPTION = 'Resolve areas of interest into a canonical AOI manifest.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    cfg = load_config(args.config)
    gdf = gpd.read_file(root / 'config' / 'cities.geojson')
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        rows.append({
            'city_id': row.get('city_id', row.get('name', 'unknown')),
            'name': row.get('name', row.get('city_id', 'unknown')),
            'area_km2': round(float(gpd.GeoSeries([geom], crs=gdf.crs).to_crs(3857).area.iloc[0]) / 1e6, 3) if gdf.crs else None,
            'bounds': list(map(float, geom.bounds)),
            'crs': str(gdf.crs),
        })
    out = root / 'outputs' / f'{args.mode}_run' / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'aoi_manifest.csv', index=False)
    write_json(out / 'aoi_manifest.json', {'mode': args.mode, 'cities': rows})
    print(out / 'aoi_manifest.csv')


if __name__ == '__main__':
    main()
