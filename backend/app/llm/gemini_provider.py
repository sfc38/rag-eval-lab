"""Gemini LLM provider — stub for the Phase 8 online demo.

Intentionally not implemented yet. Free-tier terms and the ``google-genai`` license must be
re-verified before this is built (see CLAUDE.md hard constraints).
"""

from __future__ import annotations

from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Placeholder for the hosted Gemini provider (online demo only)."""

    def __init__(self, model: str) -> None:
        """Store the model name; construction is allowed but generation is not yet supported."""
        self._model = model

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Not implemented until Phase 8.

        Raises:
            NotImplementedError: Always, in Phase 1.
        """
        raise NotImplementedError(
            "GeminiProvider is reserved for the Phase 8 online demo. "
            "Re-verify free-tier terms before implementing."
        )
