"""Document registry persisted as JSON.

Tracks uploaded documents and which indexes have been built for each. Kept deliberately
simple (a JSON file guarded by a lock) — the portfolio value is in the eval harness, not in
a database layer.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """One uploaded document and its built indexes."""

    document_id: str
    filename: str
    path: str
    char_count: int
    status: str = "uploaded"
    # index_hash -> chunk_count
    indexes: dict[str, int] = field(default_factory=dict)


class DocumentRegistry:
    """Thread-safe JSON-backed registry of documents."""

    def __init__(self, registry_path: Path) -> None:
        """Load the registry from disk (or start empty).

        Args:
            registry_path: Path to the JSON registry file.
        """
        self._path = registry_path
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        """Read records from the JSON file if it exists."""
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._records = {doc_id: DocumentRecord(**data) for doc_id, data in raw.items()}
        logger.info("Loaded %d document record(s)", len(self._records))

    def _save(self) -> None:
        """Write the current records to the JSON file atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {doc_id: asdict(rec) for doc_id, rec in self._records.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def add(self, record: DocumentRecord) -> None:
        """Add a new document record and persist.

        Args:
            record: The document record to store.
        """
        with self._lock:
            self._records[record.document_id] = record
            self._save()

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return a record by id, or ``None`` if absent.

        Args:
            document_id: Document identifier.

        Returns:
            DocumentRecord | None: The record if found.
        """
        return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        """Return all document records."""
        return list(self._records.values())

    def set_index(self, document_id: str, index_hash: str, chunk_count: int) -> None:
        """Record that an index was built for a document.

        Args:
            document_id: Document identifier.
            index_hash: The index-affecting config hash.
            chunk_count: Number of chunks in the index.
        """
        with self._lock:
            record = self._records[document_id]
            record.indexes[index_hash] = chunk_count
            record.status = "indexed"
            self._save()

    def delete(self, document_id: str) -> DocumentRecord | None:
        """Remove a document record and persist.

        Args:
            document_id: Document identifier.

        Returns:
            DocumentRecord | None: The removed record, if it existed.
        """
        with self._lock:
            record = self._records.pop(document_id, None)
            if record is not None:
                self._save()
            return record
