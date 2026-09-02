#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd
from PIL import Image

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--dimensions',default='FIGURE_DIMENSIONS.csv')
    ap.add_argument('--figure-dir',required=True)
    ap.add_argument('--output',default='outputs/logs/figure_dimension_validation.json')
    args=ap.parse_args()
    dims=pd.read_csv(args.dimensions).set_index('figure_id')
    root=Path(args.figure_dir)
    rows=[]; errors=[]
    for p in sorted(root.glob('*.png')):
        fid=p.stem.split('_')[0]
        if fid not in dims.index: continue
        with Image.open(p) as im: w,h=im.size; dpi=im.info.get('dpi',(0,0))
        r=dims.loc[fid]
        ok=w>=int(r.min_width_px_300dpi) and h>=int(r.min_height_px_300dpi)
        rows.append({'figure_id':fid,'file':str(p),'width_px':w,'height_px':h,'dpi':dpi,'pass':bool(ok)})
        if not ok: errors.append(str(p))
    report={'status':'PASS' if not errors else 'FAIL','records':rows,'errors':errors}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not errors else 1)
if __name__=='__main__': main()
