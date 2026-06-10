"""Prompt construction for the RAG pipeline.

The prompt is returned (and persisted) alongside answers so the debugger can show the exact
text sent to the LLM. The abstention instruction ("say it is not in the context") lays the
groundwork for the Phase 5 unanswerable-question metric.
"""

from __future__ import annotations

from app.vector_store.chroma_store import StoredChunk

_SYSTEM_INSTRUCTION = (
    "You are a precise assistant answering questions using ONLY the provided context.\n"
    "Rules:\n"
    "1. Use only information in the context below. Do not use outside knowledge.\n"
    "2. If the answer is not in the context, reply exactly: "
    '"The answer is not in the provided context."\n'
    "3. Cite the source chunks you used with their [number].\n"
)


def build_prompt(question: str, chunks: list[StoredChunk]) -> str:
    """Assemble the final prompt from a question and retrieved chunks.

    Args:
        question: The user question.
        chunks: Retrieved context chunks, in rank order.

    Returns:
        str: The complete prompt to send to the LLM.
    """
    if chunks:
        context_blocks = "\n\n".join(
            f"[{i + 1}] (chunk {chunk.chunk_index}, score {chunk.score:.3f})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
    else:
        context_blocks = "(no context retrieved)"

    return (
        f"{_SYSTEM_INSTRUCTION}\n"
        f"=== CONTEXT ===\n{context_blocks}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== ANSWER ==="
    )
