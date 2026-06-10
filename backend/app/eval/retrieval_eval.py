"""Retrieval-only evaluation: score a config against the golden dataset.

For each question we retrieve the top ranking, derive the ground-truth relevant chunk set from
evidence spans (matched against *all* chunks in the index, not just the retrieved ones), and
compute recall@k / MRR / nDCG@k. Answerable and unanswerable questions are reported separately —
retrieval metrics only apply to answerable ones; unanswerable questions are an answer-side
(Phase 5) concern and are counted but excluded from recall/MRR/nDCG aggregates.

Generation is intentionally NOT invoked here — this is the retrieval-only path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import PipelineConfig
from app.eval import artifacts, retrieval_metrics
from app.eval.dataset import GoldenRecord
from app.embeddings.embedding_providers import embed_query
from app.vector_store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass
class EvalRunResult:
    """The persisted outcome of a retrieval evaluation run."""

    run_id: str
    summary: dict
    rows: list[dict]


def evaluate_retrieval(
    store: ChromaStore,
    document_id: str,
    index_hash: str,
    records: list[GoldenRecord],
    config: PipelineConfig,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> EvalRunResult:
    """Evaluate retrieval quality for one config over a set of golden records.

    Args:
        store: The vector store holding the document's index.
        document_id: The registered document id to retrieve against.
        index_hash: The index hash of the built index (chunking + embedding).
        records: Golden records to evaluate (already filtered to this document).
        config: The pipeline configuration under test.
        k_values: Cutoffs to report recall/nDCG at.

    Returns:
        EvalRunResult: Run id, aggregate summary, and per-question rows (also written to disk).

    Raises:
        ValueError: If the index has no chunks (the document was not indexed).
    """
    all_chunks = store.get_all_chunks(document_id, index_hash)
    if not all_chunks:
        raise ValueError("Index has no chunks; index the document before evaluating.")
    chunk_pairs = [(c.chunk_index, c.text) for c in all_chunks]
    max_k = max(k_values)

    rows: list[dict] = []
    answerable_metrics: list[dict] = []
    for record in records:
        ranked, latency_ms = _retrieve_ranked_indices(store, document_id, index_hash, record, config, max_k)
        relevant = retrieval_metrics.relevant_chunk_indices(chunk_pairs, record.evidence_spans)
        row = _build_row(record, ranked, relevant, latency_ms, k_values)
        rows.append(row)
        if not record.is_unanswerable:
            answerable_metrics.append(row)

    summary = _aggregate(answerable_metrics, rows, config, document_id, index_hash, k_values)
    run_id = artifacts.new_run_id(config.config_hash())
    summary["run_id"] = run_id
    artifacts.write_run(run_id, config.to_dict(), rows, summary)
    return EvalRunResult(run_id=run_id, summary=summary, rows=rows)


def _retrieve_ranked_indices(
    store: ChromaStore,
    document_id: str,
    index_hash: str,
    record: GoldenRecord,
    config: PipelineConfig,
    max_k: int,
) -> tuple[list[int], int]:
    """Retrieve the top ``max_k`` chunk indices for a question, timing the retrieval.

    Thresholding is bypassed here: recall/nDCG measure ranking quality over the full top-k.
    """
    start = time.perf_counter()
    query_vector = embed_query(record.question, config.embedding_model)
    results = store.query(document_id, index_hash, query_vector, top_k=max_k)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return [c.chunk_index for c in results], latency_ms


def _build_row(
    record: GoldenRecord,
    ranked: list[int],
    relevant: set[int],
    latency_ms: int,
    k_values: tuple[int, ...],
) -> dict:
    """Assemble a per-question result row with metrics flattened into columns."""
    row: dict = {
        "question_id": record.id,
        "question_type": record.question_type,
        "difficulty": record.difficulty,
        "num_relevant": len(relevant),
        "retrieved_indices": "|".join(str(i) for i in ranked),
        "reciprocal_rank": round(retrieval_metrics.reciprocal_rank(relevant, ranked), 4),
        "retrieval_ms": latency_ms,
    }
    for k in k_values:
        row[f"recall@{k}"] = round(retrieval_metrics.recall_at_k(relevant, ranked, k), 4)
        row[f"ndcg@{k}"] = round(retrieval_metrics.ndcg_at_k(relevant, ranked, k), 4)
    return row


def _aggregate(
    answerable_rows: list[dict],
    all_rows: list[dict],
    config: PipelineConfig,
    document_id: str,
    index_hash: str,
    k_values: tuple[int, ...],
) -> dict:
    """Average per-question metrics over answerable questions and add run metadata."""
    n_answerable = len(answerable_rows)
    metrics: dict = {}
    for k in k_values:
        metrics[f"recall@{k}"] = _mean(answerable_rows, f"recall@{k}")
        metrics[f"ndcg@{k}"] = _mean(answerable_rows, f"ndcg@{k}")
    metrics["mrr"] = _mean(answerable_rows, "reciprocal_rank")
    metrics["mean_retrieval_ms"] = _mean(all_rows, "retrieval_ms")

    return {
        "document_id": document_id,
        "index_hash": index_hash,
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "k_values": list(k_values),
        "n_questions": len(all_rows),
        "n_answerable": n_answerable,
        "n_unanswerable": len(all_rows) - n_answerable,
        "metrics": metrics,
    }


def _mean(rows: list[dict], key: str) -> float:
    """Mean of a numeric column over rows, or 0.0 if empty."""
    if not rows:
        return 0.0
    return round(sum(float(r[key]) for r in rows) / len(rows), 4)
