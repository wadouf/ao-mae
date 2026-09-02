#!/usr/bin/env python3
"""Validate an empirical OA-MAE release against the locked manuscript interface."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path

REQUIRED_FIGURES = [
    "figure3_cloud_burden_gain.pdf",
    "figure4_label_noise_robustness.pdf",
    "figure5_abstention_policy.pdf",
    "figure6_severe_cloud_qualitative.pdf",
    "figure7_gate_mechanism.pdf",
    "figureS1_observability_diagnostics.pdf",
    "figureS2_past_only_retrieval.pdf",
    "figureS3_proposal_annotation_audit.pdf",
    "figureS4_failure_taxonomy.pdf",
    "figureS5_intercity_contact_sheet.pdf",
    "figureS6_city_scene_atlas_part1.pdf",
    "figureS6_city_scene_atlas_part2.pdf",
]
REQUIRED_VALUE_FILES = ["values.json", "values.tex", "value_source_map.csv"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("empirical_root", type=Path)
    ap.add_argument("--output", type=Path, default=Path("empirical_interface_validation.json"))
    args = ap.parse_args()
    root = args.empirical_root.resolve()
    missing = []
    found = []
    for name in REQUIRED_FIGURES:
        p = root / "figures" / name
        if not p.is_file():
            missing.append(str(p))
        else:
            found.append({"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    for name in REQUIRED_VALUE_FILES:
        p = root / "results" / name
        if not p.is_file():
            missing.append(str(p))
        else:
            found.append({"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    report = {
        "root": str(root),
        "status": "PASS" if not missing else "BLOCKED",
        "missing": missing,
        "found": found,
        "rule": "Passing this interface check does not validate scientific correctness; array-level and manuscript-level reconciliation remain mandatory.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
