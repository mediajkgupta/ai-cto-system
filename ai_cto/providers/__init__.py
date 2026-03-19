"""
ai_cto.providers — LLM provider abstraction layer.

Usage:
    from ai_cto.providers import get_provider

    provider = get_provider()
    text = provider.complete(system="You are...", user="Build a...", max_tokens=2048)

Provider selection (AI_CTO_LLM_PROVIDER env var, or heuristics):
    mock       → MockProvider      — no API call, returns static response
    anthropic  → AnthropicProvider — Anthropic Claude via official SDK
    ollama     → OllamaProvider    — local Ollama server via HTTP

See factory.py for full selection logic.
"""

from ai_cto.providers.base import LLMProvider
from ai_cto.providers.factory import get_provider, reset_provider

__all__ = ["LLMProvider", "get_provider", "reset_provider"]
