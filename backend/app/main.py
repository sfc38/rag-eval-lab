"""FastAPI application — RAG Eval Lab backend (Phase 1).

Endpoints:
    GET    /health
    POST   /documents/upload
    POST   /documents/index
    GET    /documents
    DELETE /documents/{document_id}
    POST   /rag/retrieve          retrieval only (powers the debugger / retrieval metrics)
    POST   /rag/ask               retrieve + prompt + generate

Eval endpoints (/eval/*) arrive in Phase 2+.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.config import PipelineConfig
from app.document_processing.cleaners import clean_text
from app.document_processing.loaders import SUPPORTED_SUFFIXES, load_text
from app.rag import rag_pipeline
from app.registry import DocumentRecord, DocumentRegistry
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    IndexRequest,
    IndexResponse,
    RetrievedChunkModel,
    RetrieveRequest,
    RetrieveResponse,
)
from app.vector_store.chroma_store import ChromaStore, StoredChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings.ensure_storage_dirs()

app = FastAPI(
    title="RAG Eval Lab",
    description="Local-first RAG pipeline + evaluation harness.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons shared across requests.
store = ChromaStore(settings.CHROMA_DIR)
registry = DocumentRegistry(settings.REGISTRY_PATH)


def _to_chunk_models(chunks: list[StoredChunk]) -> list[RetrievedChunkModel]:
    """Convert internal chunks to API response models."""
    return [
        RetrievedChunkModel(chunk_index=c.chunk_index, text=c.text, score=c.score)
        for c in chunks
    ]


def _load_document_text(record: DocumentRecord) -> str:
    """Load and clean the stored text for a document record."""
    return clean_text(load_text(Path(record.path)))


@app.get("/health")
def health() -> dict:
    """Report API health and whether Ollama is reachable.

    Returns:
        dict: Status payload including Ollama reachability.
    """
    ollama_ok = False
    try:
        import requests

        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:  # noqa: BLE001 - health check must never raise
        ollama_ok = False
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "default_model": settings.DEFAULT_OLLAMA_MODEL,
        "documents": len(registry.list()),
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Upload a document, extract and clean its text, and register it.

    Args:
        file: The uploaded PDF / TXT / MD file.

    Returns:
        DocumentInfo: Metadata for the newly registered document.

    Raises:
        HTTPException: 400 if the file type is unsupported or extraction yields no text.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    document_id = uuid.uuid4().hex[:12]
    dest = settings.UPLOADS_DIR / f"{document_id}{suffix}"
    dest.write_bytes(await file.read())

    try:
        text = clean_text(load_text(dest))
    except Exception as exc:  # noqa: BLE001 - surface extraction errors cleanly
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to read document: {exc}") from exc

    if not text.strip():
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text found in document.")

    record = DocumentRecord(
        document_id=document_id,
        filename=file.filename,
        path=str(dest),
        char_count=len(text),
    )
    registry.add(record)
    logger.info("Uploaded document %s (%s, %d chars)", document_id, file.filename, len(text))
    return DocumentInfo(**_record_to_info(record))


@app.post("/documents/index")
def index_document(request: IndexRequest) -> IndexResponse:
    """Build the vector index for a document under a configuration (cached by index hash).

    Args:
        request: Document id and pipeline configuration.

    Returns:
        IndexResponse: Index hash, chunk count, and cache status.

    Raises:
        HTTPException: 404 if the document is unknown; 400 on indexing failure.
    """
    record = registry.get(request.document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    config = request.config.to_pipeline_config()
    text = _load_document_text(record)
    try:
        result = rag_pipeline.ensure_index(
            store, request.document_id, text, config,
            chunk_count_for_hash=record.indexes.get(config.index_hash()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry.set_index(request.document_id, result.index_hash, result.chunk_count)
    return IndexResponse(
        document_id=request.document_id,
        index_hash=result.index_hash,
        chunk_count=result.chunk_count,
        cached=result.cached,
    )


@app.get("/documents")
def list_documents() -> list[DocumentInfo]:
    """List all registered documents.

    Returns:
        list[DocumentInfo]: Metadata for every document.
    """
    return [DocumentInfo(**_record_to_info(r)) for r in registry.list()]


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    """Delete a document, its file, and all its indexes.

    Args:
        document_id: Document identifier.

    Returns:
        dict: ``{"deleted": document_id}``.

    Raises:
        HTTPException: 404 if the document is unknown.
    """
    record = registry.delete(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    store.delete_document(document_id)
    Path(record.path).unlink(missing_ok=True)
    logger.info("Deleted document %s", document_id)
    return {"deleted": document_id}


@app.post("/rag/retrieve")
def rag_retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant chunks for a question (no generation).

    Auto-builds the index if it is not yet cached for the requested configuration.

    Args:
        request: Document id, question, and configuration.

    Returns:
        RetrieveResponse: Retrieved chunks and retrieval latency.

    Raises:
        HTTPException: 404 if the document is unknown.
    """
    record = registry.get(request.document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    config = request.config.to_pipeline_config()
    _ensure_indexed(record, config)
    result = rag_pipeline.retrieve_chunks(store, request.document_id, request.question, config)
    return RetrieveResponse(
        document_id=request.document_id,
        question=request.question,
        index_hash=result.index_hash,
        chunks=_to_chunk_models(result.chunks),
        retrieval_ms=result.retrieval_ms,
    )


@app.post("/rag/ask")
def rag_ask(request: AskRequest) -> AskResponse:
    """Answer a question over a document (retrieve + generate).

    Args:
        request: Document id, question, and configuration.

    Returns:
        AskResponse: Answer, sources, exact prompt, and per-stage latencies.

    Raises:
        HTTPException: 404 if the document is unknown; 502 if the LLM call fails.
    """
    record = registry.get(request.document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    config = request.config.to_pipeline_config()
    _ensure_indexed(record, config)
    try:
        result = rag_pipeline.ask(store, request.document_id, request.question, config)
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - map LLM/transport errors to 502
        logger.exception("Ask failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    return AskResponse(
        document_id=request.document_id,
        question=request.question,
        answer=result.answer,
        sources=_to_chunk_models(result.sources),
        prompt=result.prompt,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )


def _ensure_indexed(record: DocumentRecord, config: PipelineConfig) -> None:
    """Build the index for a document/config on demand if it does not exist yet."""
    if store.has_index(record.document_id, config.index_hash()):
        return
    text = _load_document_text(record)
    result = rag_pipeline.ensure_index(store, record.document_id, text, config)
    registry.set_index(record.document_id, result.index_hash, result.chunk_count)


def _record_to_info(record: DocumentRecord) -> dict:
    """Map a registry record to the API ``DocumentInfo`` fields."""
    return {
        "document_id": record.document_id,
        "filename": record.filename,
        "char_count": record.char_count,
        "status": record.status,
        "indexes": record.indexes,
    }
