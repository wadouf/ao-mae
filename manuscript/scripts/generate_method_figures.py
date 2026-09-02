from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C = {
    "optical": "#DDEAF7",
    "sar": "#F7E3D1",
    "cloud": "#E9E3F6",
    "model": "#E2F1E7",
    "loss": "#F7F0D6",
    "output": "#E9ECEF",
    "edge": "#34495E",
}


def node(ax, x, y, w, h, text, color, fs=8.6, bold=False):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color, edgecolor=C["edge"], linewidth=1.0,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")
    return p


def connect(ax, p1, p2, rad=0.0, label=None, label_shift=(0, 0)):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=11,
        connectionstyle=f"arc3,rad={rad}", linewidth=1.15, color=C["edge"],
    )
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_shift[0]
        my = (p1[1] + p2[1]) / 2 + label_shift[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={
        "Title": name.replace("_", " "),
        "Author": "OA-MAE",
        "Subject": "Method diagram",
    })
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1():
    fig, ax = plt.subplots(figsize=(16.2, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.025, 0.955, "Stage I: observability-aligned multimodal pretraining",
            fontsize=15, fontweight="bold", va="top")

    # Inputs
    node(ax, 0.025, 0.72, 0.105, 0.12, "Sentinel-2\noptical bands", C["optical"], bold=True)
    node(ax, 0.025, 0.56, 0.105, 0.10, "External cloud\nprobability", C["cloud"], bold=True)
    node(ax, 0.155, 0.72, 0.105, 0.12, "Sentinel-1\nVV and VH", C["sar"], bold=True)

    # Encoders and gates
    node(ax, 0.30, 0.72, 0.115, 0.12, "Optical encoder\nViT-S/16", C["model"], bold=True)
    node(ax, 0.30, 0.56, 0.115, 0.10, "Token cloud\nrefinement", C["cloud"])
    node(ax, 0.30, 0.87, 0.115, 0.065, "SAR encoder\nViT-Tiny", C["model"])

    node(ax, 0.455, 0.72, 0.13, 0.12, "Gated SAR-to-optical\ncross-attention", C["model"], bold=True)
    node(ax, 0.455, 0.56, 0.13, 0.10, "Cloud gate and\nSAR reliability", C["cloud"])
    node(ax, 0.455, 0.87, 0.13, 0.065, "Cloud-Mix\nmask", C["model"])

    node(ax, 0.625, 0.72, 0.135, 0.12, "Masked optical\nreconstruction", C["loss"], bold=True)
    node(ax, 0.625, 0.56, 0.135, 0.10, "Past-only Top-K targets\nwithin 90 days", C["loss"])
    node(ax, 0.625, 0.87, 0.135, 0.065, "Structural\nfallback", C["loss"])

    node(ax, 0.805, 0.68, 0.165, 0.18,
         "Stage I objective\nreconstruction + gradient\n+ structural fallback\n+ redundancy reduction",
         C["output"], bold=True)

    # Clean arrows
    connect(ax, (0.13, 0.78), (0.30, 0.78))
    connect(ax, (0.13, 0.61), (0.30, 0.61))
    connect(ax, (0.26, 0.78), (0.30, 0.90), rad=-0.08)
    connect(ax, (0.415, 0.78), (0.455, 0.78))
    connect(ax, (0.415, 0.61), (0.455, 0.61))
    connect(ax, (0.415, 0.90), (0.455, 0.90))
    connect(ax, (0.585, 0.78), (0.625, 0.78))
    connect(ax, (0.585, 0.61), (0.625, 0.61))
    connect(ax, (0.585, 0.90), (0.625, 0.90))
    connect(ax, (0.76, 0.78), (0.805, 0.78))
    connect(ax, (0.76, 0.61), (0.805, 0.72), rad=-0.08)
    connect(ax, (0.76, 0.90), (0.805, 0.83), rad=0.08)
    # Cloud and SAR to gates
    connect(ax, (0.13, 0.61), (0.455, 0.64), rad=-0.12)
    connect(ax, (0.26, 0.78), (0.455, 0.58), rad=0.12)

    ax.text(0.50, 0.505,
            "External cloud evidence controls fusion and defines a shared evaluation support; model confidence does not select the scored region.",
            ha="center", va="center", fontsize=9.3, style="italic")

    ax.text(0.025, 0.455, "Stage II: few-shot dense change detection with explicit deferral",
            fontsize=15, fontweight="bold", va="top")

    # Stage II in a clean straight pipeline
    node(ax, 0.035, 0.16, 0.115, 0.16, "T1 inputs\nS2 + S1 + cloud", C["optical"], bold=True)
    node(ax, 0.035, 0.00, 0.115, 0.12, "T2 inputs\nS2 + S1 + cloud", C["optical"], bold=True)
    node(ax, 0.20, 0.16, 0.15, 0.16, "Shared pretrained\nencoders", C["model"], bold=True)
    node(ax, 0.20, 0.00, 0.15, 0.12, "Shared pretrained\nencoders", C["model"], bold=True)
    node(ax, 0.405, 0.08, 0.15, 0.17, "Bi-temporal operator\n|F2 - F1| and F2 x F1", C["model"], bold=True)
    node(ax, 0.605, 0.08, 0.13, 0.17, "Lightweight dense\ndecoder", C["model"], bold=True)
    node(ax, 0.79, 0.17, 0.17, 0.12, "Change probability\nand binary prediction", C["output"], bold=True)
    node(ax, 0.79, 0.01, 0.17, 0.12, "V12 support, coverage\nand unresolved output", C["cloud"], bold=True)

    connect(ax, (0.15, 0.24), (0.20, 0.24))
    connect(ax, (0.15, 0.06), (0.20, 0.06))
    connect(ax, (0.35, 0.24), (0.405, 0.19))
    connect(ax, (0.35, 0.06), (0.405, 0.13))
    connect(ax, (0.555, 0.165), (0.605, 0.165))
    connect(ax, (0.735, 0.165), (0.79, 0.23))
    connect(ax, (0.15, 0.24), (0.79, 0.07), rad=-0.20,
            label="support from both dates", label_shift=(0.02, 0.015))
    connect(ax, (0.15, 0.06), (0.79, 0.07), rad=0.12)

    save(fig, "figure1_oamae_overview")


def figure2():
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.0))
    rng = np.random.default_rng(12)
    n = 72
    yy, xx = np.mgrid[0:n, 0:n]

    def blobs(shiftx, shifty):
        z = 0.10 + 0.06 * rng.random((n, n))
        for cx, cy, s, a in [(20 + shiftx, 20 + shifty, 12, 0.75),
                             (54 + shiftx, 18 + shifty, 9, 0.62),
                             (39 + shiftx, 52 + shifty, 14, 0.72)]:
            z += a * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*s*s))
        return np.clip(z, 0, 1)

    c1 = blobs(-3, 1); c2 = blobs(4, -2)

    def refine(c):
        k = 8; h = n // k
        pooled = c.reshape(h, k, h, k).mean((1, 3))
        r = np.minimum(1, pooled + 0.3 * pooled * (pooled >= 0.2))
        return np.repeat(np.repeat(r, k, 0), k, 1)

    r1 = refine(c1); r2 = refine(c2)
    m1 = r1 <= 0.85; m2 = r2 <= 0.85; v = m1 & m2
    gt = ((xx-45)**2 + (yy-42)**2 < 9**2) | ((xx > 12) & (xx < 25) & (yy > 47) & (yy < 58))
    unresolved = gt & (~v)
    arrays = [c1, c2, r1, r2, m1, m2, v, unresolved]
    titles = ["Cloud probability T1", "Cloud probability T2", "Refined token map T1", "Refined token map T2",
              "Visible support T1", "Visible support T2", "Joint support V12", "Unresolved positive support"]

    for ax, data, title in zip(axes.flat, arrays, titles):
        if data.dtype == bool:
            ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(data, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("External observability support and operational deferral", fontsize=15, y=1.01, fontweight="bold")
    fig.text(0.5, -0.015,
             "V12 is the intersection of deterministic per-date visibility masks. Predictive metrics are computed on V12, while coverage and unresolved positives are reported separately.",
             ha="center", fontsize=10)
    save(fig, "figure2_observable_support")


if __name__ == "__main__":
    figure1()
    figure2()
