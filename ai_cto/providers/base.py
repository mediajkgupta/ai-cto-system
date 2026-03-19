"""
base.py — LLM Provider abstract base class.

All providers must implement the `complete` method, which takes a system
prompt, a user message, and a max_tokens limit, and returns the model's
text response as a string.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        Send a completion request to the provider.

        Args:
            system:     System prompt (instructions/persona).
            user:       User message (the actual task/question).
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response as a plain string.

        Raises:
            RuntimeError: If the provider call fails.
        """
        ...
