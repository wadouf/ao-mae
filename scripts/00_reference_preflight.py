#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--package-root',default='.')
    args=ap.parse_args()
    root=Path(args.package_root).resolve()
    ref=root/'reference_data'
    comp=root/'reference_complements'
    errors=[]
    for p in ref.rglob('*'):
        if not p.is_file(): continue
        try:
            if p.suffix=='.csv': pd.read_csv(p)
            elif p.suffix=='.json': json.loads(p.read_text(encoding='utf-8'))
            elif p.suffix=='.jsonl': [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as z:
                    for k in z.files:
                        a=z[k]
                        if a.dtype.kind in 'fc' and not np.isfinite(a).all(): errors.append(f'nonfinite:{p}:{k}')
        except Exception as e: errors.append(f'parse:{p}:{e}')
    val=json.loads((comp/'manifests/reference_completion_validation.json').read_text())
    report={'status':'PASS' if not errors else 'FAIL','errors':errors,'reference_files':sum(1 for p in ref.rglob('*') if p.is_file()),'completion_validation':val}
    out=root/'outputs/logs/reference_preflight.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__': main()
