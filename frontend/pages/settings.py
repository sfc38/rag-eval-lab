"""Settings page: edit the active pipeline configuration.

The configuration lives in ``st.session_state["config"]`` as a plain dict matching the API's
``ConfigModel``. Every page reads from it; this page is where it is edited.
"""

from __future__ import annotations

import streamlit as st

DEFAULT_CONFIG: dict = {
    "chunking_strategy": "recursive",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "similarity_threshold": 0.0,
    "rerank": False,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm_provider": "ollama",
    "llm_model": "qwen2.5:3b",
}

_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]
_CHUNKING_STRATEGIES = ["recursive", "fixed", "sentence"]


def ensure_config() -> dict:
    """Return the active config from session state, initializing defaults if needed."""
    if "config" not in st.session_state:
        st.session_state["config"] = DEFAULT_CONFIG.copy()
    return st.session_state["config"]


def render() -> None:
    """Render the settings page and persist edits to session state."""
    st.title("⚙️ Settings")
    st.caption("Changing chunking or the embedding model builds a new index on next use. "
               "Changing top_k, threshold, or the LLM reuses the existing index.")
    config = ensure_config()

    st.subheader("Chunking")
    col1, col2, col3 = st.columns(3)
    config["chunking_strategy"] = col1.selectbox(
        "Strategy", _CHUNKING_STRATEGIES,
        index=_CHUNKING_STRATEGIES.index(config["chunking_strategy"]),
        help="recursive: split on paragraphs/sentences then merge · fixed: character windows · "
             "sentence: pack whole sentences.",
    )
    config["chunk_size"] = col2.number_input(
        "Chunk size (chars)", min_value=100, max_value=2000,
        value=int(config["chunk_size"]), step=50,
    )
    config["chunk_overlap"] = col3.number_input(
        "Overlap (chars)", min_value=0, max_value=500,
        value=int(config["chunk_overlap"]), step=25,
    )

    st.subheader("Embedding")
    config["embedding_model"] = st.selectbox(
        "Model", _EMBEDDING_MODELS,
        index=_EMBEDDING_MODELS.index(config["embedding_model"])
        if config["embedding_model"] in _EMBEDDING_MODELS else 0,
        help="Both are free (Apache 2.0 / MIT). Switching rebuilds the index.",
    )

    st.subheader("Retrieval")
    col4, col5 = st.columns(2)
    config["top_k"] = col4.slider("top_k", 1, 20, int(config["top_k"]))
    config["similarity_threshold"] = col5.slider(
        "Similarity threshold", 0.0, 1.0, float(config["similarity_threshold"]), 0.05,
        help="Drop retrieved chunks below this cosine similarity. 0 keeps all.",
    )

    st.subheader("Reranking")
    config["rerank"] = st.toggle(
        "Cross-encoder reranking", value=bool(config["rerank"]),
        help="Implemented in Phase 4. Enabling it now has no effect yet.",
    )

    st.subheader("LLM")
    col6, col7 = st.columns(2)
    config["llm_provider"] = col6.selectbox(
        "Provider", ["ollama", "gemini"],
        index=["ollama", "gemini"].index(config["llm_provider"]),
        help="gemini is reserved for the Phase 8 online demo.",
    )
    config["llm_model"] = col7.text_input("Model", value=config["llm_model"])

    st.session_state["config"] = config

    if st.button("Reset to defaults"):
        st.session_state["config"] = DEFAULT_CONFIG.copy()
        st.rerun()

    with st.expander("Active config (sent with every request)"):
        st.json(config)
