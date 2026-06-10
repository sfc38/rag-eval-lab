"""ChromaDB persistent vector store.

Collections are keyed by ``(document_id, index_hash)`` so an index built for a given
chunking + embedding configuration is reused across retrieval/LLM settings (index caching).
Cosine space is used; ChromaDB returns cosine *distance*, converted here to a similarity score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    """A chunk returned from a similarity query."""

    chunk_index: int
    text: str
    score: float  # cosine similarity in [0, 1] (higher is more similar)


class ChromaStore:
    """Thin wrapper around a persistent ChromaDB client."""

    def __init__(self, persist_dir: Path) -> None:
        """Initialize the persistent client.

        Args:
            persist_dir: Directory where ChromaDB persists its data.
        """
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info("ChromaDB persistent client at %s", persist_dir)

    @staticmethod
    def _collection_name(document_id: str, index_hash: str) -> str:
        """Build the collection name for a document/index pair."""
        return f"doc_{document_id}_{index_hash}"

    def has_index(self, document_id: str, index_hash: str) -> bool:
        """Return whether an index already exists for this document/config.

        Args:
            document_id: Document identifier.
            index_hash: Index-affecting config hash.

        Returns:
            bool: True if the collection exists.
        """
        name = self._collection_name(document_id, index_hash)
        return any(c.name == name for c in self._client.list_collections())

    def index_chunks(
        self,
        document_id: str,
        index_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Create (or replace) a collection and add chunk embeddings.

        Args:
            document_id: Document identifier.
            index_hash: Index-affecting config hash.
            chunks: Chunk texts.
            embeddings: Matching embedding vectors (same length as ``chunks``).

        Returns:
            int: The number of chunks indexed.
        """
        name = self._collection_name(document_id, index_hash)
        # Rebuild from scratch to keep indexing idempotent.
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001 - collection may simply not exist yet
            pass
        collection = self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.info("Indexed %d chunk(s) into %s", len(chunks), name)
        return len(chunks)

    def query(
        self,
        document_id: str,
        index_hash: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[StoredChunk]:
        """Run a similarity query against a document's index.

        Args:
            document_id: Document identifier.
            index_hash: Index-affecting config hash.
            query_embedding: The embedded query vector.
            top_k: Number of chunks to return.

        Returns:
            list[StoredChunk]: Ranked chunks with cosine-similarity scores.
        """
        name = self._collection_name(document_id, index_hash)
        collection = self._client.get_collection(name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        chunks: list[StoredChunk] = []
        for text, distance, meta in zip(documents, distances, metadatas):
            chunks.append(
                StoredChunk(
                    chunk_index=int(meta.get("chunk_index", -1)),
                    text=text,
                    score=_distance_to_similarity(distance),
                )
            )
        return chunks

    def get_all_chunks(self, document_id: str, index_hash: str) -> list[StoredChunk]:
        """Return every chunk in a document's index (used to derive ground-truth relevance).

        Args:
            document_id: Document identifier.
            index_hash: Index-affecting config hash.

        Returns:
            list[StoredChunk]: All chunks ordered by chunk index. ``score`` is unused (0.0).
        """
        name = self._collection_name(document_id, index_hash)
        collection = self._client.get_collection(name)
        result = collection.get(include=["documents", "metadatas"])
        chunks: list[StoredChunk] = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(
                StoredChunk(chunk_index=int(meta.get("chunk_index", -1)), text=text, score=0.0)
            )
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def delete_document(self, document_id: str) -> None:
        """Delete every collection belonging to a document.

        Args:
            document_id: Document identifier whose indexes should be removed.
        """
        prefix = f"doc_{document_id}_"
        for collection in self._client.list_collections():
            if collection.name.startswith(prefix):
                self._client.delete_collection(collection.name)
                logger.info("Deleted collection %s", collection.name)


def _distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB cosine distance to a [0, 1] similarity score.

    Args:
        distance: Cosine distance (``1 - cosine_similarity``).

    Returns:
        float: Similarity score clamped to [0, 1].
    """
    similarity = 1.0 - float(distance)
    return max(0.0, min(1.0, similarity))
