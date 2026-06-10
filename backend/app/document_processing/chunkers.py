"""Chunking strategies.

Three strategies are implemented because chunking is a primary experiment axis (Phase 3):

* ``fixed``     — fixed-size character windows with overlap.
* ``recursive`` — split on a priority list of separators, then greedily merge to target size.
* ``sentence``  — split into sentences, then pack sentences up to target size with overlap.

All strategies return ``list[str]`` of non-empty chunks. The vector store assigns chunk indices.
"""

from __future__ import annotations

import logging
import re

from app.config import ChunkingStrategy

logger = logging.getLogger(__name__)

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks using the given strategy.

    Args:
        text: Cleaned document text.
        strategy: One of ``"fixed"``, ``"recursive"``, ``"sentence"``.
        chunk_size: Target maximum chunk length in characters.
        overlap: Number of characters (approx) shared between consecutive chunks.

    Returns:
        list[str]: Non-empty text chunks.

    Raises:
        ValueError: If the strategy is unknown.
    """
    if not text.strip():
        return []
    if strategy == "fixed":
        chunks = _chunk_fixed(text, chunk_size, overlap)
    elif strategy == "recursive":
        chunks = _chunk_recursive(text, chunk_size, overlap)
    elif strategy == "sentence":
        chunks = _chunk_sentence(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    chunks = [c.strip() for c in chunks if c.strip()]
    logger.info("Chunked text into %d chunk(s) [%s, size=%d, overlap=%d]",
                len(chunks), strategy, chunk_size, overlap)
    return chunks


def _chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Slide a fixed-size window across the character stream."""
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
    return chunks


def _chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Recursively split to atomic units, then merge greedily with overlap."""
    units = _split_to_units(text, size, _RECURSIVE_SEPARATORS)
    return _merge_with_overlap(units, size, overlap)


def _split_to_units(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into pieces each <= ``size`` by descending separator priority.

    Args:
        text: Text to split.
        size: Maximum piece length.
        separators: Priority-ordered separators; ``""`` means hard character split.

    Returns:
        list[str]: Pieces, each at most ``size`` characters (best effort).
    """
    if len(text) <= size or not separators:
        return [text]
    sep, *rest = separators
    if sep == "":
        return [text[i:i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    units: list[str] = []
    for i, part in enumerate(parts):
        # Re-attach the separator (except after the final part) to preserve structure.
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= size:
            if piece:
                units.append(piece)
        else:
            units.extend(_split_to_units(piece, size, rest))
    return units


def _merge_with_overlap(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge units into chunks up to ``size``, carrying an overlap tail."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + unit
        else:
            current += unit
    if current.strip():
        chunks.append(current)
    return chunks


def _chunk_sentence(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks up to ``size``, with sentence-level overlap."""
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > size:
            chunks.append(" ".join(current))
            current, current_len = _overlap_tail(current, overlap)
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing sentences whose combined length is within ``overlap``."""
    if overlap <= 0:
        return [], 0
    tail: list[str] = []
    tail_len = 0
    for sentence in reversed(sentences):
        if tail_len + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        tail_len += len(sentence) + 1
    return tail, tail_len
