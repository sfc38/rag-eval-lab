"""Abstract LLM provider interface.

A single ``generate`` method keeps providers swappable (the config's ``llm_provider`` axis).
Ollama is the local default; Gemini is added for the Phase 8 online demo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Generate a completion for a prompt.

        Args:
            prompt: The fully-built prompt.
            temperature: Sampling temperature (low default for factual RAG).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            str: The model's text response.
        """
        raise NotImplementedError
