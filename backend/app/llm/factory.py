"""LLM provider factory: build a provider from a pipeline configuration."""

from __future__ import annotations

from app.config import PipelineConfig
from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.settings import OLLAMA_BASE_URL


def build_provider(config: PipelineConfig) -> LLMProvider:
    """Construct the LLM provider named by ``config.llm_provider``.

    Args:
        config: The pipeline configuration.

    Returns:
        LLMProvider: A ready-to-use provider.

    Raises:
        ValueError: If the provider name is unknown.
    """
    if config.llm_provider == "ollama":
        return OllamaProvider(base_url=OLLAMA_BASE_URL, model=config.llm_model)
    if config.llm_provider == "gemini":
        return GeminiProvider(model=config.llm_model)
    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")
