"""RAG pipeline orchestration.

Three operations, deliberately separated:

* :func:`ensure_index` — build the vector index if it is not already cached.
* :func:`retrieve_chunks` — retrieval only (powers the debugger and retrieval metrics).
* :func:`ask` — retrieve → build prompt → generate → return answer + sources + prompt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import PipelineConfig
from app.document_processing.chunkers import chunk_text
from app.embeddings.embedding_providers import embed_texts
from app.llm.factory import build_provider
from app.retrieval.retriever import retrieve
from app.vector_store.chroma_store import ChromaStore, StoredChunk

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Outcome of an indexing operation."""

    index_hash: str
    chunk_count: int
    cached: bool


@dataclass
class RetrieveResult:
    """Outcome of a retrieval operation."""

    index_hash: str
    chunks: list[StoredChunk]
    retrieval_ms: int


@dataclass
class AskResult:
    """Outcome of a full ask (retrieve + generate)."""

    answer: str
    sources: list[StoredChunk]
    prompt: str
    retrieval_ms: int
    generation_ms: int


def ensure_index(
    store: ChromaStore,
    document_id: str,
    text: str,
    config: PipelineConfig,
    chunk_count_for_hash: int | None = None,
) -> IndexResult:
    """Build the index for a document/config if not already present.

    Args:
        store: The vector store.
        document_id: Document identifier.
        text: Cleaned document text.
        config: Pipeline configuration (only chunking + embedding axes matter here).
        chunk_count_for_hash: Known chunk count if the index already exists, for reporting.

    Returns:
        IndexResult: The index hash, chunk count, and whether it was served from cache.
    """
    index_hash = config.index_hash()
    if store.has_index(document_id, index_hash):
        logger.info("Index cache hit for %s/%s", document_id, index_hash)
        return IndexResult(index_hash=index_hash, chunk_count=chunk_count_for_hash or 0, cached=True)

    chunks = chunk_text(text, config.chunking_strategy, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise ValueError("Document produced no chunks; check extraction and chunking settings.")
    embeddings = embed_texts(chunks, config.embedding_model)
    count = store.index_chunks(document_id, index_hash, chunks, embeddings)
    return IndexResult(index_hash=index_hash, chunk_count=count, cached=False)


def retrieve_chunks(
    store: ChromaStore,
    document_id: str,
    question: str,
    config: PipelineConfig,
) -> RetrieveResult:
    """Run retrieval only, measuring latency.

    Args:
        store: The vector store.
        document_id: Document identifier.
        question: The user question.
        config: Pipeline configuration.

    Returns:
        RetrieveResult: Retrieved chunks and retrieval latency in milliseconds.
    """
    start = time.perf_counter()
    chunks = retrieve(store, document_id, question, config)
    retrieval_ms = int((time.perf_counter() - start) * 1000)
    return RetrieveResult(index_hash=config.index_hash(), chunks=chunks, retrieval_ms=retrieval_ms)


def ask(
    store: ChromaStore,
    document_id: str,
    question: str,
    config: PipelineConfig,
) -> AskResult:
    """Answer a question: retrieve, build the prompt, and generate.

    Args:
        store: The vector store.
        document_id: Document identifier.
        question: The user question.
        config: Pipeline configuration.

    Returns:
        AskResult: Answer, source chunks, the exact prompt, and per-stage latencies.
    """
    from app.rag.prompt_builder import build_prompt  # local import keeps module graph shallow

    if config.rerank:
        logger.warning("Reranking requested but not implemented until Phase 4; ignoring.")

    retrieval = retrieve_chunks(store, document_id, question, config)
    prompt = build_prompt(question, retrieval.chunks)

    provider = build_provider(config)
    start = time.perf_counter()
    answer = provider.generate(prompt)
    generation_ms = int((time.perf_counter() - start) * 1000)

    return AskResult(
        answer=answer,
        sources=retrieval.chunks,
        prompt=prompt,
        retrieval_ms=retrieval.retrieval_ms,
        generation_ms=generation_ms,
    )
