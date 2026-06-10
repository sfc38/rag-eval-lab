"""Runtime settings: environment variables and storage paths.

Pipeline configuration (the experiment axes) lives in ``config.py``. This module holds
only deployment/runtime concerns so paths and provider URLs are never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolves to the ``backend/`` directory regardless of where the process is launched.
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(BACKEND_ROOT / "storage")))
UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
CHROMA_DIR: Path = STORAGE_DIR / "chroma"
REGISTRY_PATH: Path = STORAGE_DIR / "documents.json"

# Ollama (local default LLM)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def ensure_storage_dirs() -> None:
    """Create the storage directories if they do not yet exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
