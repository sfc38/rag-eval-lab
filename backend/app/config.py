"""Pipeline configuration and config hashing.

``PipelineConfig`` captures every experiment axis of the RAG pipeline. Two hashes are
derived from it:

* ``index_hash`` — over the axes that affect the **vector index** (chunking + embedding).
  ChromaDB collections are keyed by this so a sweep over retrieval/LLM settings reuses an
  existing index instead of re-embedding.
* ``config_hash`` — over the **entire** config; identifies a full experiment run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

ChunkingStrategy = Literal["fixed", "recursive", "sentence"]
LLMProviderName = Literal["ollama", "gemini"]


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable description of one RAG pipeline configuration.

    Frozen so it is hashable and safe to use as a cache key. Defaults define the
    Phase 1 baseline pipeline.
    """

    # ── chunking ──
    chunking_strategy: ChunkingStrategy = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── embedding ──
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── retrieval ──
    top_k: int = 5
    similarity_threshold: float = 0.0  # keep all results by default

    # ── reranking (implemented in Phase 4) ──
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── llm ──
    llm_provider: LLMProviderName = "ollama"
    llm_model: str = "qwen2.5:3b"

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is internally inconsistent.

        Raises:
            ValueError: If any numeric axis is out of its valid range.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

    def index_signature(self) -> dict:
        """Return only the axes that affect the vector index.

        Returns:
            dict: Chunking and embedding fields used to key a cached index.
        """
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
        }

    def index_hash(self) -> str:
        """Return a stable short hash of the index-affecting axes."""
        payload = json.dumps(self.index_signature(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def config_hash(self) -> str:
        """Return a stable short hash of the entire configuration."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Return the configuration as a plain dict (for persistence/logging)."""
        return asdict(self)
