"""Safe, deterministic writing of generated controller artifacts."""

from __future__ import annotations

from pathlib import Path


def write_text(path: Path, content: str) -> None:
    """Create parent directories and write UTF-8 text with a final newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8", newline="\n")

