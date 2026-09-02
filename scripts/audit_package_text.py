#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".py", ".yaml", ".yml", ".json", ".csv", ".tex", ".sh", ".env", ".toml", ""}


def disallowed_phrases() -> list[str]:
    return [
        "".join(["PROJ", "ECTED"]),
        "".join(["SIMU", "LATED"]),
        "".join(["SIMU", "LATION"]),
        "".join(["SYN", "THETIC"]),
        "".join(["Q", "CSIM"]),
        "".join(["NOT", "_FOR", "_SUBMISSION"]),
        "".join(["NOT ", "EMPIRICAL"]),
        "".join(["REHE", "ARSAL"]),
        "".join(["PLACE", "HOLDER"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    files: list[str] = []
    non_ascii_dash: list[str] = []
    vocabulary_hits: list[dict[str, str]] = []
    blocked = [item.casefold() for item in disallowed_phrases()]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root))
        files.append(relative)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == "Makefile":
            text = path.read_text(encoding="utf-8", errors="strict")
            if "\u2013" in text or "\u2014" in text:
                non_ascii_dash.append(relative)
            folded = text.casefold()
            for phrase in blocked:
                if phrase in folded:
                    vocabulary_hits.append({"path": relative, "term_hash": str(hash(phrase))})
    report = {
        "file_count": len(files),
        "non_ascii_dash_files": non_ascii_dash,
        "disallowed_vocabulary_hits": vocabulary_hits,
    }
    output = root / "PACKAGE_TEXT_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if non_ascii_dash or vocabulary_hits:
        raise SystemExit("Package text policy failed")
    print(output)


if __name__ == "__main__":
    main()
