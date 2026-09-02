from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def assert_ascii_hyphens(path: Path) -> None:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    assert "\u2013" not in text
    assert "\u2014" not in text
