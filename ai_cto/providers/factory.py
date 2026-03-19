"""
factory.py — Provider factory: select and instantiate the right LLM backend.

Selection order (first match wins):
  1. AI_CTO_LLM_PROVIDER env var
       "mock"       → MockProvider  (returns pre-defined responses, no API call)
       "ollama"     → OllamaProvider (local Ollama server)
       "anthropic"  → AnthropicProvider (Anthropic API)
  2. AI_CTO_MOCK_LLM=true / 1 / yes → MockProvider
  3. ANTHROPIC_API_KEY present and non-empty → AnthropicProvider
  4. Fallback → MockProvider (no key available, treat as mock)

The factory caches the selected provider instance for the lifetime of the
process (module-level singleton), so repeated calls to get_provider() don't
create multiple SDK clients.
"""

import logging
import os

from ai_cto.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_PROVIDER_INSTANCE: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """
    Return the singleton LLMProvider for the current process.

    Call this once per agent invocation; the result is lightweight to reuse.
    """
    global _PROVIDER_INSTANCE
    if _PROVIDER_INSTANCE is None:
        _PROVIDER_INSTANCE = _create_provider()
    return _PROVIDER_INSTANCE


def _create_provider() -> LLMProvider:
    """Instantiate the correct provider based on environment variables."""
    explicit = os.environ.get("AI_CTO_LLM_PROVIDER", "").strip().lower()

    if explicit == "mock":
        logger.info("[Factory] AI_CTO_LLM_PROVIDER=mock → MockProvider")
        return _mock_provider()

    if explicit == "ollama":
        logger.info("[Factory] AI_CTO_LLM_PROVIDER=ollama → OllamaProvider")
        from ai_cto.providers.ollama import OllamaProvider
        return OllamaProvider()

    if explicit == "anthropic":
        logger.info("[Factory] AI_CTO_LLM_PROVIDER=anthropic → AnthropicProvider")
        from ai_cto.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    if explicit:
        raise ValueError(
            f"Unknown AI_CTO_LLM_PROVIDER value '{explicit}'. "
            "Valid options: mock, anthropic, ollama"
        )

    # No explicit provider set — use env heuristics
    if os.environ.get("AI_CTO_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        logger.info("[Factory] AI_CTO_MOCK_LLM=true → MockProvider")
        return _mock_provider()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        logger.info("[Factory] ANTHROPIC_API_KEY present → AnthropicProvider")
        from ai_cto.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    logger.info("[Factory] No API key / provider set → MockProvider (fallback)")
    return _mock_provider()


def _mock_provider() -> LLMProvider:
    """Return a thin MockProvider that delegates to mock_llm helpers."""
    from ai_cto.providers.mock_provider import MockProvider
    return MockProvider()


def reset_provider() -> None:
    """
    Clear the cached provider singleton.

    Call this in tests that need to swap providers between test cases,
    or after changing environment variables mid-process.
    """
    global _PROVIDER_INSTANCE
    _PROVIDER_INSTANCE = None
