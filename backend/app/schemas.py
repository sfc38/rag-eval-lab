"""Pydantic models for the API boundary.

These mirror ``PipelineConfig`` and the pipeline's return values. Field names match
``PipelineConfig`` exactly so conversion is a direct unpack.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import PipelineConfig


class ConfigModel(BaseModel):
    """Request-side mirror of :class:`PipelineConfig`. All fields optional with defaults."""

    chunking_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_threshold: float = 0.0
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to a validated :class:`PipelineConfig`.

        Returns:
            PipelineConfig: The validated immutable config.
        """
        config = PipelineConfig(**self.model_dump())
        config.validate()
        return config


class IndexRequest(BaseModel):
    """Request body for ``POST /documents/index``."""

    document_id: str
    config: ConfigModel = Field(default_factory=ConfigModel)


class RetrieveRequest(BaseModel):
    """Request body for ``POST /rag/retrieve``."""

    document_id: str
    question: str
    config: ConfigModel = Field(default_factory=ConfigModel)


class AskRequest(BaseModel):
    """Request body for ``POST /rag/ask``."""

    document_id: str
    question: str
    config: ConfigModel = Field(default_factory=ConfigModel)


class RetrievedChunkModel(BaseModel):
    """A single retrieved chunk with its score and position."""

    chunk_index: int
    text: str
    score: float


class IndexResponse(BaseModel):
    """Response for ``POST /documents/index``."""

    document_id: str
    index_hash: str
    chunk_count: int
    cached: bool


class RetrieveResponse(BaseModel):
    """Response for ``POST /rag/retrieve``."""

    document_id: str
    question: str
    index_hash: str
    chunks: list[RetrievedChunkModel]
    retrieval_ms: int


class AskResponse(BaseModel):
    """Response for ``POST /rag/ask``."""

    document_id: str
    question: str
    answer: str
    sources: list[RetrievedChunkModel]
    prompt: str
    retrieval_ms: int
    generation_ms: int


class DocumentInfo(BaseModel):
    """Metadata about an uploaded document."""

    document_id: str
    filename: str
    char_count: int
    status: str
    indexes: dict[str, int]
