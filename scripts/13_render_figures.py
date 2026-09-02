#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from oamae_pipeline.config import load_config
from oamae_pipeline.figures import render_qf1
from oamae_pipeline.common import write_json

DESCRIPTION = 'Render figure assets from arrays and manifests.'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['reference', 'observed'], default='reference')
    args = parser.parse_args()
    root = Path('.').resolve()
    load_config(args.config)
    out_dir = root / 'outputs' / f'{args.mode}_run' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = root / 'data' / 'processed' / 'scene_bundles'
    pred_dir = root / 'data' / 'predictions' / args.mode
    selected = pd.read_csv(root / 'outputs' / f'{args.mode}_run' / 'tables' / 'selected_cases.csv')
    qf1_ids = list(selected[selected['figure_id'] == 'QF1']['sample_id'])[:4]
    rows = []
    for sid in qf1_ids:
        bundle = dict(np.load(bundle_dir / f'{sid}.npz'))
        oamae = dict(np.load(pred_dir / f'{sid}_oamae.npz'))
        croma = dict(np.load(pred_dir / f'{sid}_croma.npz'))
        rows.append({
            'row_label': sid,
            'rgb_t1': bundle['s2_rgb_t1'],
            'rgb_t2': bundle['s2_rgb_t2'],
            'sar_change': bundle['sar_change'],
            'sar_min': float(bundle['sar_change'].min()),
            'sar_max': float(bundle['sar_change'].max()),
            'mean_cloud': 0.5 * (bundle['cloud_t1'] + bundle['cloud_t2']),
            'reference': bundle['ground_truth'],
            'v12': bundle['v12'],
            'croma_probability': croma['probability'],
            'oamae_probability': oamae['probability'],
        })
    if rows:
        render_qf1(rows, out_dir)
    # Copy clean reference figures as additional comparators so the agent has all figure paths available.
    ref_fig_dir = root / 'reference_data' / 'qualitative_benchmark' / 'figures'
    for name in ['QF2_observability_abstention_diagnostics.pdf', 'QF2_observability_abstention_diagnostics.png', 'QF3_past_only_retrieval_diagnostics.pdf', 'QF3_past_only_retrieval_diagnostics.png', 'QF4_sar_optical_gate_mechanism.pdf', 'QF4_sar_optical_gate_mechanism.png', 'QF5_proposal_annotation_audit.pdf', 'QF5_proposal_annotation_audit.png', 'QF6_failure_taxonomy.pdf', 'QF6_failure_taxonomy.png', 'QF7_intercity_contact_sheet.pdf', 'QF7_intercity_contact_sheet_page1.png', 'QF7_intercity_contact_sheet_page2.png', 'S0_city_scene_atlas.pdf', 'S0_city_scene_atlas.png']:
        src = ref_fig_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    write_json(out_dir / 'figure_render_summary.json', {'rendered_qf1_rows': len(rows), 'copied_reference_figure_assets': len(list(out_dir.glob('*')))})
    print(out_dir)


if __name__ == '__main__':
    main()
