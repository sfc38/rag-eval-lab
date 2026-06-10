"""Ollama LLM provider (local default).

Uses the non-streaming ``/api/generate`` endpoint. The evaluation harness needs complete
answer strings, so streaming adds no value here; the UI shows a spinner instead.
"""

from __future__ import annotations

import logging

import requests

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Calls a local Ollama server."""

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        """Initialize the provider.

        Args:
            base_url: Ollama server base URL, e.g. ``http://localhost:11434``.
            model: Model tag to use, e.g. ``qwen2.5vl:3b``.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        """Generate a completion via Ollama.

        Args:
            prompt: The fully-built prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (mapped to Ollama ``num_predict``).

        Returns:
            str: The generated text.

        Raises:
            requests.HTTPError: If Ollama returns a non-2xx status.
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(
            f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
