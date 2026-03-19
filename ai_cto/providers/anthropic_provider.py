"""
anthropic_provider.py — Anthropic Claude provider.

Wraps the Anthropic SDK's messages.create() into the LLMProvider interface.
Uses the model specified by AI_CTO_ANTHROPIC_MODEL (default: claude-sonnet-4-6).
"""

import os
import logging
from ai_cto.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    """LLM provider backed by Anthropic Claude via the official SDK."""

    def __init__(self) -> None:
        from anthropic import Anthropic
        self._client = Anthropic()
        self._model = os.environ.get("AI_CTO_ANTHROPIC_MODEL", _DEFAULT_MODEL)
        logger.info("[AnthropicProvider] Initialized. model=%s", self._model)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        Call the Anthropic messages API and return the response text.

        Raises:
            RuntimeError: On any Anthropic API error.
        """
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except Exception as exc:
            raise RuntimeError(f"AnthropicProvider.complete failed: {exc}") from exc
