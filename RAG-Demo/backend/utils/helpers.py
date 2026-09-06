"""
General-purpose helper utilities.
"""

import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe in file names."""
    return re.sub(r"[^\w\-.]", "_", name)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def flatten(nested: list[list]) -> list:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in sublist]
