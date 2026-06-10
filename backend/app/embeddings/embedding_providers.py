"""Embedding provider backed by sentence-transformers.

Models are cached per process so a sweep does not reload the same model repeatedly.
Embeddings are L2-normalized so cosine similarity equals the dot product, matching the
ChromaDB ``cosine`` space.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # avoid importing the heavy library at module load
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> "SentenceTransformer":
    """Load (and cache) a sentence-transformers model by name.

    Args:
        model_name: HuggingFace model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.

    Returns:
        SentenceTransformer: The loaded model.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a batch of texts.

    Args:
        texts: Texts to embed.
        model_name: Embedding model id.

    Returns:
        list[list[float]]: One normalized embedding vector per input text.
    """
    if not texts:
        return []
    model = _get_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str, model_name: str) -> list[float]:
    """Embed a single query string.

    Args:
        text: The query text.
        model_name: Embedding model id.

    Returns:
        list[float]: The normalized embedding vector.
    """
    return embed_texts([text], model_name)[0]
