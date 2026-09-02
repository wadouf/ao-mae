from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ee
import google.auth


def initialize(project: str | None = None) -> None:
    project_id = project or os.environ.get("EARTHENGINE_PROJECT")
    if not project_id:
        raise RuntimeError("EARTHENGINE_PROJECT is required")
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path:
        key = Path(key_path)
        if not key.exists():
            raise FileNotFoundError(key)
        import json
        account = json.loads(key.read_text(encoding="utf-8"))["client_email"]
        credentials = ee.ServiceAccountCredentials(account, str(key))
        ee.Initialize(credentials, project=project_id)
    else:
        credentials, _ = google.auth.default()
        ee.Initialize(credentials, project=project_id)


def joined_s2_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start, end))
    cloud = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
             .filterBounds(aoi)
             .filterDate(start, end))
    joined = ee.Join.saveFirst("cloud_image").apply(
        primary=sr,
        secondary=cloud,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def add_cloud(image: ee.Image) -> ee.Image:
        cloud_image = ee.Image(image.get("cloud_image"))
        return image.addBands(cloud_image.select("probability").rename("cloud_probability"))

    return ee.ImageCollection(joined).map(add_cloud)


def s1_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def image_properties(collection: ee.ImageCollection, properties: list[str], limit: int = 10000) -> list[dict[str, Any]]:
    size = int(collection.size().getInfo())
    if size > limit:
        raise RuntimeError(f"Collection has {size} items, above limit {limit}")
    items = collection.toList(size)
    rows: list[dict[str, Any]] = []
    for index in range(size):
        image = ee.Image(items.get(index))
        row = image.toDictionary(properties).getInfo()
        row["system:index"] = image.get("system:index").getInfo()
        row["system:time_start"] = image.get("system:time_start").getInfo()
        rows.append(row)
    return rows
