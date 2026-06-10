"""Index & Chat page: upload a document, index it, and ask questions with sources.

This is the demo surface. The portfolio centerpiece (benchmarks, debugger) arrives in later
phases; here we prove the pipeline works end to end.
"""

from __future__ import annotations

import streamlit as st

import api_client
from pages.settings import ensure_config


def render() -> None:
    """Render the Index & Chat page."""
    st.title("📚 Index & Chat")
    config = ensure_config()

    _render_upload()
    st.divider()
    document_id = _render_document_picker()
    if document_id is None:
        st.info("Upload and select a document to start asking questions.")
        return

    st.divider()
    _render_chat(document_id, config)


def _render_upload() -> None:
    """File uploader + upload action."""
    st.subheader("1 · Upload a document")
    uploaded = st.file_uploader("PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded is not None and st.button("Upload", type="primary"):
        with st.spinner("Extracting text…"):
            result = api_client.upload_document(uploaded.name, uploaded.getvalue(), uploaded.type)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Uploaded **{result['filename']}** "
                       f"({result['char_count']:,} chars) · id `{result['document_id']}`")
            st.rerun()


def _render_document_picker() -> str | None:
    """Document selector with delete control. Returns the selected document id."""
    st.subheader("2 · Select a document")
    documents = api_client.list_documents()
    if not documents:
        return None

    labels = {
        f"{d['filename']}  ·  {d['char_count']:,} chars  ·  {d['document_id']}": d["document_id"]
        for d in documents
    }
    col_sel, col_del = st.columns([6, 1])
    label = col_sel.selectbox("Document", list(labels.keys()), label_visibility="collapsed")
    document_id = labels[label]
    if col_del.button("🗑️", help="Delete this document and its indexes"):
        if api_client.delete_document(document_id):
            st.rerun()
        else:
            st.error("Delete failed.")
    return document_id


def _render_chat(document_id: str, config: dict) -> None:
    """Question box + answer with sources and timings."""
    st.subheader("3 · Ask a question")
    st.caption(f"Config: {config['chunking_strategy']} · size {config['chunk_size']} · "
               f"top_k {config['top_k']} · {config['llm_provider']}/{config['llm_model']}")

    question = st.text_input("Question", placeholder="What is this document about?")
    if not st.button("Ask", type="primary"):
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return

    with st.spinner("Retrieving and generating…"):
        result = api_client.ask(document_id, question, config)
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("### Answer")
    st.markdown(result["answer"] or "_(empty answer)_")

    c1, c2 = st.columns(2)
    c1.metric("Retrieval", f"{result['retrieval_ms']} ms")
    c2.metric("Generation", f"{result['generation_ms']} ms")

    with st.expander(f"Sources ({len(result['sources'])} chunks)", expanded=True):
        for i, src in enumerate(result["sources"], start=1):
            st.markdown(f"**[{i}]** chunk {src['chunk_index']} · similarity {src['score']:.3f}")
            st.text(src["text"])
            st.divider()

    with st.expander("Exact prompt sent to the LLM"):
        st.code(result["prompt"])
