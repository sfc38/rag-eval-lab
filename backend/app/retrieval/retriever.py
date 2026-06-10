"""Retrieval: embed a query and fetch the most similar chunks.

This is the retrieval-only path that powers both the debugger and the Phase 2 retrieval
metrics. Reranking (Phase 4) will wrap the output of :func:`retrieve` before it reaches the
prompt builder.
"""

from __future__ import annotations

import logging

from app.config import PipelineConfig
from app.embeddings.embedding_providers import embed_query
from app.vector_store.chroma_store import ChromaStore, StoredChunk

logger = logging.getLogger(__name__)


def retrieve(
    store: ChromaStore,
    document_id: str,
    question: str,
    config: PipelineConfig,
) -> list[StoredChunk]:
    """Retrieve the top chunks for a question under a given configuration.

    Args:
        store: The vector store holding the document's index.
        document_id: Document to query.
        question: The user question.
        config: Pipeline configuration (embedding model, top_k, threshold).

    Returns:
        list[StoredChunk]: Chunks above the similarity threshold, ranked by score.
    """
    query_vector = embed_query(question, config.embedding_model)
    chunks = store.query(
        document_id=document_id,
        index_hash=config.index_hash(),
        query_embedding=query_vector,
        top_k=config.top_k,
    )
    if config.similarity_threshold > 0.0:
        kept = [c for c in chunks if c.score >= config.similarity_threshold]
        logger.info(
            "Retrieved %d chunk(s); %d above threshold %.2f",
            len(chunks), len(kept), config.similarity_threshold,
        )
        return kept
    return chunks
