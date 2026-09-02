from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy


def write_cog(path: Path, array: np.ndarray, crs: str, transform, descriptions: Iterable[str], nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array if array.ndim == 3 else array[None, ...]
    count, height, width = data.shape
    temp = path.with_suffix(".tmp.tif")
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": count,
        "dtype": str(data.dtype), "crs": crs, "transform": transform,
        "compress": "DEFLATE", "tiled": True, "blockxsize": 256, "blockysize": 256,
        "nodata": nodata,
    }
    with rasterio.open(temp, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(descriptions, start=1):
            dst.set_band_description(index, name)
    rio_copy(temp, path, driver="COG", compress="DEFLATE", overview_resampling="average")
    temp.unlink(missing_ok=True)


def assert_aligned(paths: list[Path]) -> None:
    reference = None
    for path in paths:
        with rasterio.open(path) as src:
            signature = (src.width, src.height, src.crs.to_string(), tuple(src.transform), tuple(src.bounds))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Raster alignment mismatch: {path}")
