"""Thin HTTP client for the RAG Eval Lab backend.

All Streamlit pages talk to the API through these functions so the base URL and error
handling live in one place. ``API_URL`` is overridable via env for deployment.
"""

from __future__ import annotations

import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
_TIMEOUT = 180


def health() -> dict:
    """Return the backend health payload, or an error dict if unreachable."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - surfaced in the sidebar
        return {"status": "unreachable", "error": str(exc)}


def list_documents() -> list[dict]:
    """Return all registered documents (empty list on failure)."""
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        return []


def upload_document(filename: str, data: bytes, mime: str) -> dict:
    """Upload a document file.

    Args:
        filename: Original file name (suffix determines the loader).
        data: Raw file bytes.
        mime: MIME type for the multipart upload.

    Returns:
        dict: The created document info, or ``{"error": ...}``.
    """
    try:
        resp = requests.post(
            f"{API_URL}/documents/upload",
            files={"file": (filename, data, mime)},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return {"error": resp.json().get("detail", resp.text)}
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def delete_document(document_id: str) -> bool:
    """Delete a document by id. Returns True on success."""
    try:
        resp = requests.delete(f"{API_URL}/documents/{document_id}", timeout=30)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def index_document(document_id: str, config: dict) -> dict:
    """Build (or reuse) the index for a document under a configuration."""
    return _post_json("/documents/index", {"document_id": document_id, "config": config})


def retrieve(document_id: str, question: str, config: dict) -> dict:
    """Run retrieval only for a question."""
    return _post_json(
        "/rag/retrieve",
        {"document_id": document_id, "question": question, "config": config},
    )


def ask(document_id: str, question: str, config: dict) -> dict:
    """Ask a question (retrieve + generate)."""
    return _post_json(
        "/rag/ask",
        {"document_id": document_id, "question": question, "config": config},
    )


def _post_json(path: str, body: dict) -> dict:
    """POST JSON and normalize errors into ``{"error": ...}``."""
    try:
        resp = requests.post(f"{API_URL}{path}", json=body, timeout=_TIMEOUT)
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
            return {"error": detail}
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
