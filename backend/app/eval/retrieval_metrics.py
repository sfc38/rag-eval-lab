"""Retrieval metrics computed from evidence-span matching.

The pipeline:

1. For a question, find the **ground-truth relevant chunks** — every chunk whose text contains
   (or substantially overlaps) one of the question's evidence spans.
2. Compare that relevant set against the **ranked retrieved chunk indices** to compute
   recall@k, reciprocal rank (for MRR), and nDCG@k.

All functions here are pure (no I/O, no model calls) so they are fully unit-testable. Relevance
is binary, which is what evidence-span matching yields.
"""

from __future__ import annotations

import math
import re

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust span matching.

    Args:
        text: Raw text.

    Returns:
        str: Normalized text.
    """
    return _WHITESPACE.sub(" ", text.lower()).strip()


def span_matches_chunk(span: str, chunk_text: str, token_overlap_threshold: float = 0.6) -> bool:
    """Return whether an evidence span is contained in (or overlaps) a chunk.

    A match is either a normalized substring match, or — to tolerate chunk boundaries that
    split a span — a token-containment ratio at or above ``token_overlap_threshold``.

    Args:
        span: The evidence span.
        chunk_text: The chunk to test.
        token_overlap_threshold: Minimum fraction of span tokens that must appear in the chunk.

    Returns:
        bool: True if the span is considered present in the chunk.
    """
    norm_span = normalize(span)
    norm_chunk = normalize(chunk_text)
    if not norm_span:
        return False
    if norm_span in norm_chunk:
        return True
    span_tokens = norm_span.split()
    if not span_tokens:
        return False
    chunk_tokens = set(norm_chunk.split())
    overlap = sum(1 for tok in span_tokens if tok in chunk_tokens)
    return (overlap / len(span_tokens)) >= token_overlap_threshold


def relevant_chunk_indices(
    chunks: list[tuple[int, str]],
    evidence_spans: list[str],
    token_overlap_threshold: float = 0.6,
) -> set[int]:
    """Find the indices of chunks relevant to a question via evidence-span matching.

    Args:
        chunks: ``(chunk_index, chunk_text)`` pairs for the entire index.
        evidence_spans: The question's verbatim evidence spans.
        token_overlap_threshold: Passed through to :func:`span_matches_chunk`.

    Returns:
        set[int]: Indices of chunks that contain at least one evidence span.
    """
    relevant: set[int] = set()
    for chunk_index, text in chunks:
        for span in evidence_spans:
            if span_matches_chunk(span, text, token_overlap_threshold):
                relevant.add(chunk_index)
                break
    return relevant


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """Fraction of relevant chunks that appear in the top ``k`` retrieved.

    Args:
        relevant: Ground-truth relevant chunk indices.
        ranked: Retrieved chunk indices in rank order (best first).
        k: Cutoff.

    Returns:
        float: ``|relevant ∩ top_k| / |relevant|``; ``0.0`` if there are no relevant chunks.
    """
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    return len(relevant & top_k) / len(relevant)


def reciprocal_rank(relevant: set[int], ranked: list[int]) -> float:
    """Reciprocal of the rank of the first relevant chunk (the per-question term of MRR).

    Args:
        relevant: Ground-truth relevant chunk indices.
        ranked: Retrieved chunk indices in rank order.

    Returns:
        float: ``1 / rank`` of the first relevant hit, or ``0.0`` if none retrieved.
    """
    for position, chunk_index in enumerate(ranked, start=1):
        if chunk_index in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary relevance.

    Args:
        relevant: Ground-truth relevant chunk indices.
        ranked: Retrieved chunk indices in rank order.
        k: Cutoff.

    Returns:
        float: nDCG@k in [0, 1]; ``0.0`` if there are no relevant chunks.
    """
    if not relevant:
        return 0.0
    dcg = 0.0
    for position, chunk_index in enumerate(ranked[:k], start=1):
        if chunk_index in relevant:
            dcg += 1.0 / math.log2(position + 1)
    # Ideal DCG: all relevant chunks ranked first, capped at k.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
