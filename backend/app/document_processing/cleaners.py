"""Text cleaning: normalize whitespace before chunking.

Kept intentionally conservative — aggressive cleaning can destroy evidence spans that the
Phase 2 golden dataset matches against retrieved chunks.
"""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize line endings and collapse redundant whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        str: Cleaned text with normalized newlines and spacing, stripped.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    # Strip trailing spaces on each line without removing blank-line structure.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
