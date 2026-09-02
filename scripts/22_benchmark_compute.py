#!/usr/bin/env python3
"""Matched compute benchmark for one 256 by 256 bi-temporal pair."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import platform
import statistics
import time

import pandas as pd
import torch
import torch.nn as nn

from oamae.config import OAMAEConfig
from oamae.inference import build_config
from oamae.segmentation import OAMAEChangeDetector
from oamae_pipeline.common import git_commit, write_json
from oamae_pipeline.config import load_config

DESCRIPTION = 'Measure parameters, operation counts, latency, memory and throughput under one convention.'
CONVENTION = '1_MAC_equals_2_FLOPs'


def count_macs(model: nn.Module, inputs: dict[str, torch.Tensor], cfg: OAMAEConfig) -> int:
    """Multiply-accumulate operations in the convolution and linear layers, plus attention."""
    total = 0
    handles = []

    def convolution(module: nn.Conv2d, _in, output):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1]
        total += output.shape[-1] * output.shape[-2] * output.shape[1] * (module.in_channels // module.groups) * kernel

    def linear(module: nn.Linear, inp, _out):
        nonlocal total
        total += inp[0].numel() // inp[0].shape[-1] * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear))

    with torch.no_grad():
        model(**inputs)
    for handle in handles:
        handle.remove()

    # scaled_dot_product_attention is not a module, so its matmuls are added analytically:
    # per block, queries times keys and the weighted sum of values, over both dates.
    tokens = cfg.token_count()
    self_attention = 2 * tokens * tokens * cfg.optical_dim * cfg.optical_depth
    self_attention += 2 * tokens * tokens * cfg.radar_dim * cfg.radar_depth
    cross_attention = 2 * tokens * tokens * cfg.optical_dim * cfg.fusion_blocks
    return total + 2 * (self_attention + cross_attention)


def measure_latency(run, repeats: int, warmup: int, device: str) -> tuple[float, float]:
    for _ in range(warmup):
        run()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    index = min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))
    return statistics.fmean(samples), samples[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None, help='optional weights; random initialisation otherwise')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='outputs/results/compute_benchmark.csv')
    args = parser.parse_args()

    root = Path('.').resolve()
    project = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg: OAMAEConfig = build_config(project)

    model = OAMAEChangeDetector(cfg)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['state_dict'] if 'state_dict' in payload else payload)
    model.to(device).eval()

    size = cfg.image_size
    batch = args.batch_size
    inputs = {
        'optical_t1': torch.randn(batch, cfg.optical_bands, size, size, device=device),
        'optical_t2': torch.randn(batch, cfg.optical_bands, size, size, device=device),
        'radar_t1': torch.randn(batch, cfg.radar_bands, size, size, device=device),
        'radar_t2': torch.randn(batch, cfg.radar_bands, size, size, device=device),
        'cloud_t1': torch.rand(batch, size, size, device=device),
        'cloud_t2': torch.rand(batch, size, size, device=device),
    }

    parameters = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, inputs, cfg) // batch

    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        mean_ms, p95_ms = measure_latency(lambda: model(**inputs), args.repeats, args.warmup, device)

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.startswith('cuda') else float('nan')
    state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    tile_area_km2 = (size * float(project['spatial']['pixel_size_m']) / 1000.0) ** 2
    throughput = batch / (mean_ms / 1000.0)

    row = {
        'method': 'OA_MAE',
        'params_m': round(parameters / 1e6, 3),
        'gflops_pair': round(2 * macs / 1e9, 3),
        'gmacs_pair': round(macs / 1e9, 3),
        'latency_mean_ms': round(mean_ms, 3),
        'latency_p95_ms': round(p95_ms, 3),
        'peak_vram_gb': round(peak_vram, 3),
        'throughput_pairs_s': round(throughput, 3),
        'model_size_mb': round(state_bytes / 1e6, 1),
        'area_km2_per_hour': round(throughput * 3600.0 * tile_area_km2, 3),
        'operation_count_convention': CONVENTION,
        'tile_pixels': size,
        'resolution_m': project['spatial']['pixel_size_m'],
        'tile_area_km2': round(tile_area_km2, 4),
        'batch_size': batch,
        'throughput_scope': 'compute_only_excludes_io_preprocessing_overlap_and_cloud_mask',
    }

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    write_json(root / 'outputs' / 'logs' / 'compute_environment.json', {
        'device': device,
        'torch': torch.__version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
        'code_commit': git_commit(root),
        'repeats': args.repeats,
        'warmup': args.warmup,
    })
    print(pd.DataFrame([row]).to_string(index=False))
    print(output)


if __name__ == '__main__':
    main()
