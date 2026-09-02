from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import box


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def build_tile_grid(boundary: gpd.GeoDataFrame, tile_size_m: float, minimum_intersection: float) -> gpd.GeoDataFrame:
    if len(boundary) != 1:
        raise ValueError("Boundary must contain one feature")
    geom_wgs = boundary.geometry.iloc[0]
    centroid = geom_wgs.centroid
    local = utm_crs_for_lonlat(centroid.x, centroid.y)
    geom = gpd.GeoSeries([geom_wgs], crs=boundary.crs).to_crs(local).iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(math.floor(minx / tile_size_m) * tile_size_m, maxx, tile_size_m)
    ys = np.arange(math.floor(miny / tile_size_m) * tile_size_m, maxy, tile_size_m)
    records = []
    counter = 0
    for y in ys:
        for x in xs:
            tile = box(x, y, x + tile_size_m, y + tile_size_m)
            fraction = tile.intersection(geom).area / tile.area
            if fraction >= minimum_intersection:
                records.append({"tile_id": f"tile_{counter:05d}", "intersection_fraction": fraction, "geometry": tile})
                counter += 1
    return gpd.GeoDataFrame(records, crs=local)


def load_fallback_cities(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
