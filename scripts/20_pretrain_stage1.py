#!/usr/bin/env python3
"""Stage I: observability-aligned masked pretraining."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from oamae.config import OAMAEConfig
from oamae.data import Normalization, PretrainDataset, collate, compute_normalization
from oamae.inference import build_config
from oamae.pretrain import OAMAEPretrainer
from oamae_pipeline.common import append_jsonl, git_commit, sha256_text, utc_now, write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.reference_io import sample_city

DESCRIPTION = 'Pretrain the OA-MAE encoders with Cloud-Mix masking and past-only targets.'


def bundle_paths(directory: Path, held_out_city: str | None) -> list[Path]:
    paths = sorted(directory.glob('SCN_*.npz'))
    if held_out_city:
        paths = [p for p in paths if sample_city(p.stem) != held_out_city]
    if not paths:
        raise SystemExit(f'No Stage I bundles found in {directory}')
    return paths


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--bundles', default='data/processed/pretrain_bundles')
    parser.add_argument('--output', default='outputs/checkpoints/stage1')
    parser.add_argument('--held-out-city', default=None, help='city excluded from Stage I for this fold')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1.5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--normalization', default='outputs/checkpoints/normalization.json')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    root = Path('.').resolve()
    project = load_config(args.config)
    seed = args.seed if args.seed is not None else int(project['project']['random_seed'])
    torch.manual_seed(seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    cfg: OAMAEConfig = build_config(project)
    paths = bundle_paths(Path(args.bundles), args.held_out_city)

    normalization_path = Path(args.normalization)
    if normalization_path.exists():
        normalization = Normalization.load(normalization_path)
    else:
        normalization = compute_normalization(paths, cfg)
        normalization.save(normalization_path)
    normalization_hash = sha256_text(normalization_path.read_text(encoding='utf-8'))

    dataset = PretrainDataset(paths, cfg, normalization)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)

    model = OAMAEPretrainer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    generator = torch.Generator(device=device).manual_seed(seed)

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = int(state['epoch']) + 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = root / 'outputs' / 'logs' / 'stage1_pretraining.jsonl'
    split_hash = sha256_text(json.dumps([p.stem for p in paths], sort_keys=True))
    config_hash = sha256_text(json.dumps(cfg.__dict__, sort_keys=True, default=str))

    total_steps = max(1, args.epochs * len(loader))
    step = start_epoch * len(loader)
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals: dict[str, float] = {}
        for batch in loader:
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            scale = cosine_schedule(step, total_steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * scale

            result = model.loss(tensors, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            step += 1

        means = {key: value / len(loader) for key, value in totals.items()}
        record = {'timestamp': utc_now(), 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'], 'samples': len(dataset), **means}
        append_jsonl(log_path, record)
        print(json.dumps(record))

        torch.save({
            'epoch': epoch,
            'config': cfg.__dict__,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metadata': {
                'stage': 'I',
                'held_out_city': args.held_out_city,
                'seed': seed,
                'code_commit': git_commit(root),
                'config_hash': config_hash,
                'split_hash': split_hash,
                'normalization_hash': normalization_hash,
                'normalization_path': str(normalization_path),
                'training_log': str(log_path.relative_to(root)),
            },
        }, output / 'stage1_last.pt')

    encoders = output / f'stage1_encoders_{args.held_out_city or "all"}_seed{seed:02d}.pt'
    torch.save({
        'config': cfg.__dict__,
        'state_dict': model.export_encoders(),
        'metadata': {
            'stage': 'I',
            'held_out_city': args.held_out_city,
            'seed': seed,
            'epochs': args.epochs,
            'code_commit': git_commit(root),
            'config_hash': config_hash,
            'split_hash': split_hash,
            'normalization_hash': normalization_hash,
            'wall_clock_hours': (time.time() - started) / 3600.0,
        },
    }, encoders)
    write_json(root / 'outputs' / 'logs' / 'stage1_summary.json', {
        'encoders': str(encoders), 'epochs': args.epochs, 'samples': len(dataset),
        'held_out_city': args.held_out_city, 'seed': seed,
    })
    print(encoders)


if __name__ == '__main__':
    main()
