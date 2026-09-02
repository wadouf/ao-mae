from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pypdf import PdfReader

from .common import sha256_file, write_json


def validate_probability_binary(probability: np.ndarray, binary: np.ndarray, threshold: float) -> None:
    expected = (probability >= threshold).astype(binary.dtype)
    if not np.array_equal(expected, binary):
        raise ValueError("Binary map does not match probability threshold")


def validate_pdf_text(path: Path) -> None:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if "\u2013" in text or "\u2014" in text:
        raise ValueError(f"Non-ASCII dash found in {path}")


def build_checksum_manifest(root: Path, output: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != output):
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(output, index=False)
