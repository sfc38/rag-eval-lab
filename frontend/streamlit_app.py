"""RAG Eval Lab — Streamlit frontend shell.

Phase 1 surfaces two pages: Index & Chat, and Settings. The Benchmark and Retrieval Debugger
pages are added in later phases. The sidebar always shows the active configuration so the user
knows exactly which pipeline is in effect.
"""

from __future__ import annotations

import streamlit as st

import api_client
from pages import index_chat, settings

st.set_page_config(page_title="RAG Eval Lab", layout="wide")


def _render_sidebar() -> str:
    """Render navigation + live status. Returns the selected page name."""
    config = settings.ensure_config()
    with st.sidebar:
        st.title("🔬 RAG Eval Lab")
        page = st.radio("Navigate", ["Index & Chat", "Settings"], label_visibility="collapsed")

        st.divider()
        st.caption("Backend status")
        status = api_client.health()
        if status.get("status") == "ok":
            ollama = "🟢" if status.get("ollama_reachable") else "🔴"
            st.success(f"API online · {status.get('documents', 0)} docs")
            st.caption(f"{ollama} Ollama · default `{status.get('default_model', '?')}`")
        else:
            st.error("API unreachable")
            st.caption(f"Run: `uvicorn app.main:app` in backend/\n\n{status.get('error', '')}")

        st.divider()
        st.caption("Active config")
        st.write(
            f"**Chunking:** {config['chunking_strategy']} "
            f"({config['chunk_size']}/{config['chunk_overlap']})"
        )
        st.write(f"**Embedding:** `{config['embedding_model'].split('/')[-1]}`")
        st.write(f"**Retrieval:** top_k {config['top_k']} · thr {config['similarity_threshold']}")
        st.write(f"**Rerank:** {'on' if config['rerank'] else 'off'}")
        st.write(f"**LLM:** {config['llm_provider']} / `{config['llm_model']}`")
    return page


def main() -> None:
    """App entry point: render the sidebar and dispatch to the selected page."""
    page = _render_sidebar()
    if page == "Settings":
        settings.render()
    else:
        index_chat.render()


main()
