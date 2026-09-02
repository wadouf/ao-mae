#!/usr/bin/env python3
"""Stage II: few-shot dense change detection for one leave-one-city-out fold."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

import torch
from torch.utils.data import DataLoader

from oamae.config import OAMAEConfig
from oamae.data import ChangeDataset, Normalization, collate, identity_normalization
from oamae.inference import build_config
from oamae.losses import supervised_loss
from oamae.segmentation import OAMAEChangeDetector
from oamae_pipeline.common import append_jsonl, git_commit, sha256_file, sha256_text, utc_now, write_json
from oamae_pipeline.config import load_config
from oamae_pipeline.reference_io import sample_city

DESCRIPTION = 'Train the Stage II decoder on K labeled tile pairs per source city.'


def build_splits(directory: Path, held_out_city: str, few_shot_k: int, seed: int) -> dict[str, list[Path]]:
    """Deterministic few-shot selection. The held-out city never enters supervision."""
    paths = sorted(directory.glob('SCN_*.npz'))
    if not paths:
        raise SystemExit(f'No Stage II bundles found in {directory}')

    by_city: dict[str, list[Path]] = {}
    for path in paths:
        by_city.setdefault(sample_city(path.stem), []).append(path)
    if held_out_city not in by_city:
        raise SystemExit(f'Held-out city {held_out_city} is absent from {directory}')

    generator = torch.Generator().manual_seed(seed)
    train: list[Path] = []
    validation: list[Path] = []
    for city, city_paths in sorted(by_city.items()):
        if city == held_out_city:
            continue
        order = torch.randperm(len(city_paths), generator=generator).tolist()
        shuffled = [city_paths[i] for i in order]
        if len(shuffled) < few_shot_k:
            raise SystemExit(f'City {city} has {len(shuffled)} tiles, fewer than K={few_shot_k}')
        train.extend(shuffled[:few_shot_k])
        validation.extend(shuffled[few_shot_k:])
    return {'train': train, 'validation': validation, 'test': sorted(by_city[held_out_city])}


def evaluate(model: OAMAEChangeDetector, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0.0
    with torch.no_grad():
        for batch in loader:
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(
                tensors['optical_t1'], tensors['optical_t2'],
                tensors['radar_t1'], tensors['radar_t2'],
                tensors['cloud_t1'], tensors['cloud_t2'],
            )
            support = out['v12']
            predicted = out['binary'] & support
            reference = (tensors['reference'] > 0.5) & support
            tp += float((predicted & reference).sum())
            fp += float((predicted & ~reference).sum())
            fn += float((~predicted & reference).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / max(1e-12, precision + recall),
        'iou': tp / max(1.0, tp + fp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--bundles', default='data/processed/scene_bundles')
    parser.add_argument('--pretrained', required=True, help='Stage I encoder checkpoint')
    parser.add_argument('--held-out-city', required=True)
    parser.add_argument('--few-shot-k', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', default='outputs/checkpoints/oamae')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--unfreeze-encoders', action='store_true', help='depart from the primary frozen-encoder setting')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--normalization', default='outputs/checkpoints/normalization.json')
    args = parser.parse_args()

    root = Path('.').resolve()
    project = load_config(args.config)
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = torch.load(args.pretrained, map_location=device, weights_only=False)
    stored = pretrained.get('config')
    cfg: OAMAEConfig = OAMAEConfig(**stored) if isinstance(stored, dict) else build_config(project)

    normalization_path = Path(args.normalization)
    normalization = Normalization.load(normalization_path) if normalization_path.exists() else identity_normalization(cfg)
    normalization_hash = sha256_file(normalization_path) if normalization_path.exists() else 'identity'

    splits = build_splits(Path(args.bundles), args.held_out_city, args.few_shot_k, args.seed)
    loaders = {
        name: DataLoader(
            ChangeDataset(paths, cfg, normalization), batch_size=args.batch_size,
            shuffle=(name == 'train'), num_workers=args.workers, collate_fn=collate,
        )
        for name, paths in splits.items() if paths
    }

    model = OAMAEChangeDetector(cfg)
    missing, unexpected = model.load_state_dict(pretrained['state_dict'], strict=False)
    if unexpected:
        raise SystemExit(f'Stage I checkpoint carries unexpected parameters: {sorted(unexpected)[:6]}')
    if not args.unfreeze_encoders:
        model.freeze_encoders()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    log_path = root / 'outputs' / 'logs' / f'stage2_{args.held_out_city}_K{args.few_shot_k}_seed{args.seed:02d}.jsonl'

    best = {'f1': -1.0, 'epoch': -1, 'state': None}
    for epoch in range(args.epochs):
        model.train()
        if not args.unfreeze_encoders:
            model.optical_encoder.eval()
            model.radar_encoder.eval()
        running = 0.0
        for batch in loaders['train']:
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(
                tensors['optical_t1'], tensors['optical_t2'],
                tensors['radar_t1'], tensors['radar_t2'],
                tensors['cloud_t1'], tensors['cloud_t2'],
            )
            loss = supervised_loss(out['probability'], tensors['reference'], out['v12'], cfg.focal_gamma, cfg.focal_alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            running += float(loss.detach())

        metrics = evaluate(model, loaders['validation'], device) if 'validation' in loaders else {'f1': float('nan')}
        record = {'timestamp': utc_now(), 'epoch': epoch, 'train_loss': running / max(1, len(loaders['train'])), **{f'validation_{k}': v for k, v in metrics.items()}}
        append_jsonl(log_path, record)
        print(json.dumps(record))

        if metrics['f1'] > best['f1']:
            best = {'f1': metrics['f1'], 'epoch': epoch, 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    if best['state'] is None:
        raise SystemExit('No epoch produced a validation score')
    model.load_state_dict(best['state'])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f'oa_mae_loco_{args.held_out_city}_K{args.few_shot_k}_seed{args.seed:02d}.npz'
    torch.save({
        'config': cfg.__dict__,
        'state_dict': best['state'],
        'metadata': {
            'method': 'OA_MAE',
            'stage': 'II',
            'left_out_city': args.held_out_city,
            'few_shot_k': args.few_shot_k,
            'seed': args.seed,
            'epoch_selected': best['epoch'],
            'selection_metric': 'validation_f1',
            'selection_value': best['f1'],
            'encoders_frozen': not args.unfreeze_encoders,
            'pretrained_checkpoint': str(args.pretrained),
            'code_commit': git_commit(root),
            'config_hash': sha256_text(json.dumps(cfg.__dict__, sort_keys=True, default=str)),
            'split_hash': sha256_text(json.dumps({k: [p.stem for p in v] for k, v in splits.items()}, sort_keys=True)),
            'normalization_hash': normalization_hash,
            'training_log': str(log_path.relative_to(root)),
        },
    }, checkpoint)

    write_json(root / 'outputs' / 'logs' / f'stage2_summary_{args.held_out_city}_K{args.few_shot_k}_seed{args.seed:02d}.json', {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'epoch_selected': best['epoch'],
        'validation_f1': best['f1'],
        'train_tiles': len(splits['train']),
        'validation_tiles': len(splits['validation']),
        'test_tiles': len(splits['test']),
    })
    print(checkpoint)


if __name__ == '__main__':
    main()
