from __future__ import annotations

import string
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .metrics import error_map

ERROR_CMAP = ListedColormap(["#111111", "#2ca25f", "#f28e2b", "#4e79a7", "#d9d9d9"])
ERROR_NORM = BoundaryNorm(np.arange(-0.5, 5.5, 1), ERROR_CMAP.N)


def panel_label(ax: plt.Axes, index: int) -> None:
    letters = string.ascii_lowercase
    label = letters[index] if index < len(letters) else f"p{index + 1}"
    ax.text(-0.035, 1.025, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", clip_on=False)


def show_image(ax: plt.Axes, image: np.ndarray, title: str, cmap=None, vmin=None, vmax=None) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def add_thin_contours(ax: plt.Axes, reference: np.ndarray, support: np.ndarray) -> None:
    if np.any(reference):
        ax.contour(reference.astype(float), levels=[0.5], colors=["white"], linewidths=0.65)
    if np.any(support == 0) and np.any(support == 1):
        ax.contour(support.astype(float), levels=[0.5], colors=["black"], linewidths=0.45, linestyles="dashed")


def save(fig: plt.Figure, pdf: Path, png: Path, dpi: int = 300) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "OA-MAE Sentinel figure workflow"})
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def render_qf1(rows: list[dict], output_dir: Path, threshold: float = 0.5) -> tuple[Path, Path]:
    columns = ["Sentinel-2 T1", "Sentinel-2 T2", "Sentinel-1 change", "Mean cloud probability", "Reference and V12", "CROMA probability", "OA-MAE probability", "CROMA errors", "OA-MAE errors"]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(18, 2.55 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    panel = 0
    for row_index, item in enumerate(rows):
        ref = item["reference"].astype(np.uint8)
        support = item["v12"].astype(np.uint8)
        croma_binary = item["croma_probability"] >= threshold
        oa_binary = item["oamae_probability"] >= threshold
        images = [
            (item["rgb_t1"], None, None, None),
            (item["rgb_t2"], None, None, None),
            (item["sar_change"], "gray", item.get("sar_min"), item.get("sar_max")),
            (item["mean_cloud"], "viridis", 0, 1),
            (np.where(support, 1 + ref, 0), ListedColormap(["#d9d9d9", "#202020", "#f2c14e"]), 0, 2),
            (item["croma_probability"], "magma", 0, 1),
            (item["oamae_probability"], "magma", 0, 1),
            (error_map(ref, croma_binary, support), ERROR_CMAP, 0, 4),
            (error_map(ref, oa_binary, support), ERROR_CMAP, 0, 4),
        ]
        for column_index, (image, cmap, vmin, vmax) in enumerate(images):
            ax = axes[row_index, column_index]
            show_image(ax, image, columns[column_index], cmap=cmap, vmin=vmin, vmax=vmax)
            if column_index in {0, 1, 5, 6}:
                add_thin_contours(ax, ref, support)
            panel_label(ax, panel)
            panel += 1
        axes[row_index, 0].set_ylabel(item["row_label"], fontsize=8, labelpad=6)
    pdf = output_dir / "QF1_severe_cloud_comparison.pdf"
    png = output_dir / "QF1_severe_cloud_comparison.png"
    save(fig, pdf, png)
    return pdf, png
