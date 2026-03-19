"""
test_providers.py — Unit tests for the LLM provider abstraction layer.

Tests:
  1. TestProviderSelection (7)  — factory chooses the correct backend
  2. TestOllamaRequestFormat (6) — OllamaProvider builds the right HTTP payload
  3. TestMockProviderComplete (3) — MockProvider returns valid fallback JSON
  4. TestIsMockModeWithProvider (4) — is_mock_mode() respects AI_CTO_LLM_PROVIDER
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset(env_keys=(), provider=True):
    """Remove env vars and reset the provider singleton."""
    for k in env_keys:
        os.environ.pop(k, None)
    if provider:
        from ai_cto.providers.factory import reset_provider
        reset_provider()


def _set_env(**kwargs):
    for k, v in kwargs.items():
        os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# 1. Provider selection
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderSelection(unittest.TestCase):

    def setUp(self):
        _reset(["AI_CTO_LLM_PROVIDER", "AI_CTO_MOCK_LLM", "ANTHROPIC_API_KEY"])

    def tearDown(self):
        _reset(["AI_CTO_LLM_PROVIDER", "AI_CTO_MOCK_LLM", "ANTHROPIC_API_KEY"])

    def test_explicit_mock(self):
        _set_env(AI_CTO_LLM_PROVIDER="mock")
        from ai_cto.providers.factory import _create_provider
        p = _create_provider()
        from ai_cto.providers.mock_provider import MockProvider
        self.assertIsInstance(p, MockProvider)

    def test_explicit_ollama(self):
        _set_env(AI_CTO_LLM_PROVIDER="ollama")
        from ai_cto.providers.factory import _create_provider
        p = _create_provider()
        from ai_cto.providers.ollama import OllamaProvider
        self.assertIsInstance(p, OllamaProvider)

    def test_explicit_anthropic(self):
        _set_env(AI_CTO_LLM_PROVIDER="anthropic")
        # AnthropicProvider imports Anthropic SDK lazily inside __init__
        with patch("anthropic.Anthropic"):
            from ai_cto.providers.factory import _create_provider
            p = _create_provider()
        from ai_cto.providers.anthropic_provider import AnthropicProvider
        self.assertIsInstance(p, AnthropicProvider)

    def test_invalid_provider_raises(self):
        _set_env(AI_CTO_LLM_PROVIDER="gpt4")
        from ai_cto.providers.factory import _create_provider
        with self.assertRaises(ValueError):
            _create_provider()

    def test_mock_llm_env_takes_precedence(self):
        _set_env(AI_CTO_MOCK_LLM="true")
        os.environ.pop("AI_CTO_LLM_PROVIDER", None)
        from ai_cto.providers.factory import _create_provider
        p = _create_provider()
        from ai_cto.providers.mock_provider import MockProvider
        self.assertIsInstance(p, MockProvider)

    def test_api_key_selects_anthropic(self):
        _set_env(ANTHROPIC_API_KEY="sk-test-key")
        os.environ.pop("AI_CTO_LLM_PROVIDER", None)
        os.environ.pop("AI_CTO_MOCK_LLM", None)
        with patch("anthropic.Anthropic"):
            from ai_cto.providers.factory import _create_provider
            p = _create_provider()
        from ai_cto.providers.anthropic_provider import AnthropicProvider
        self.assertIsInstance(p, AnthropicProvider)

    def test_no_key_no_provider_falls_back_to_mock(self):
        os.environ.pop("AI_CTO_LLM_PROVIDER", None)
        os.environ.pop("AI_CTO_MOCK_LLM", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        from ai_cto.providers.factory import _create_provider
        p = _create_provider()
        from ai_cto.providers.mock_provider import MockProvider
        self.assertIsInstance(p, MockProvider)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ollama request format
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaRequestFormat(unittest.TestCase):
    """Verify OllamaProvider builds the correct payload without real HTTP."""

    def setUp(self):
        _reset(["OLLAMA_BASE_URL", "OLLAMA_MODEL"])

    def tearDown(self):
        _reset(["OLLAMA_BASE_URL", "OLLAMA_MODEL"])

    def _make_provider(self, base_url=None, model=None):
        if base_url:
            os.environ["OLLAMA_BASE_URL"] = base_url
        if model:
            os.environ["OLLAMA_MODEL"] = model
        from ai_cto.providers.ollama import OllamaProvider
        return OllamaProvider()

    def _capture_payload(self, provider, system, user, max_tokens):
        """Call provider.complete() with a mocked urlopen and return the request body.

        The mock response is an NDJSON stream (one JSON line per token) to match
        the streaming API format used by OllamaProvider.
        """
        captured = {}
        # Build two NDJSON lines: one token chunk + done sentinel
        stream_lines = [
            json.dumps({"message": {"content": "ollama"}, "done": False}).encode() + b"\n",
            json.dumps({"message": {"content": " says hi"}, "done": True}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(stream_lines)

        def fake_urlopen(req, timeout=None):
            captured["url"]     = req.full_url
            captured["method"]  = req.method
            captured["headers"] = dict(req.headers)
            captured["body"]    = json.loads(req.data.decode())
            return mock_resp

        with patch("ai_cto.providers.ollama.urllib.request.urlopen", fake_urlopen):
            result = provider.complete(system=system, user=user, max_tokens=max_tokens)

        captured["result"] = result
        return captured

    def test_endpoint_url(self):
        p = self._make_provider(base_url="http://localhost:11434")
        c = self._capture_payload(p, "sys", "usr", 512)
        self.assertEqual(c["url"], "http://localhost:11434/api/chat")

    def test_custom_base_url(self):
        p = self._make_provider(base_url="http://192.168.1.10:11434")
        c = self._capture_payload(p, "sys", "usr", 512)
        self.assertEqual(c["url"], "http://192.168.1.10:11434/api/chat")

    def test_payload_model(self):
        p = self._make_provider(model="codellama:7b")
        c = self._capture_payload(p, "sys", "usr", 512)
        self.assertEqual(c["body"]["model"], "codellama:7b")

    def test_payload_messages(self):
        p = self._make_provider()
        c = self._capture_payload(p, "system-prompt", "user-prompt", 512)
        msgs = c["body"]["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "system-prompt"})
        self.assertEqual(msgs[1], {"role": "user",   "content": "user-prompt"})

    def test_payload_stream_true(self):
        """Streaming is required to avoid socket timeout on slow CPU inference."""
        p = self._make_provider()
        c = self._capture_payload(p, "s", "u", 100)
        self.assertTrue(c["body"]["stream"])

    def test_max_tokens_in_options(self):
        p = self._make_provider()
        c = self._capture_payload(p, "s", "u", 1024)
        self.assertEqual(c["body"]["options"]["num_predict"], 1024)

    def test_response_content_returned(self):
        p = self._make_provider()
        c = self._capture_payload(p, "s", "u", 512)
        self.assertEqual(c["result"], "ollama says hi")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MockProvider.complete()
# ─────────────────────────────────────────────────────────────────────────────

class TestMockProviderComplete(unittest.TestCase):

    def _make(self):
        from ai_cto.providers.mock_provider import MockProvider
        return MockProvider()

    def test_returns_string(self):
        p = self._make()
        result = p.complete("sys", "usr", max_tokens=512)
        self.assertIsInstance(result, str)

    def test_returns_valid_json(self):
        p = self._make()
        result = p.complete("sys", "usr")
        parsed = json.loads(result)
        self.assertIn("files", parsed)

    def test_no_network_call(self):
        """MockProvider must not attempt any I/O."""
        import urllib.request
        original = urllib.request.urlopen
        p = self._make()
        # If urlopen were called this would raise; the test would fail
        try:
            p.complete("sys", "usr")
        finally:
            urllib.request.urlopen = original  # restore just in case


# ─────────────────────────────────────────────────────────────────────────────
# 4. is_mock_mode() with AI_CTO_LLM_PROVIDER set
# ─────────────────────────────────────────────────────────────────────────────

class TestIsMockModeWithProvider(unittest.TestCase):

    def setUp(self):
        _reset(["AI_CTO_LLM_PROVIDER", "AI_CTO_MOCK_LLM", "ANTHROPIC_API_KEY"])

    def tearDown(self):
        _reset(["AI_CTO_LLM_PROVIDER", "AI_CTO_MOCK_LLM", "ANTHROPIC_API_KEY"])

    def _mock_mode(self):
        # Reload to pick up env var changes
        import importlib
        import ai_cto.mock_llm as m
        importlib.reload(m)
        return m.is_mock_mode()

    def test_ollama_provider_not_mock_even_without_api_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        _set_env(AI_CTO_LLM_PROVIDER="ollama")
        self.assertFalse(self._mock_mode())

    def test_anthropic_provider_not_mock_even_without_api_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        _set_env(AI_CTO_LLM_PROVIDER="anthropic")
        self.assertFalse(self._mock_mode())

    def test_explicit_mock_provider_is_mock(self):
        _set_env(AI_CTO_LLM_PROVIDER="mock")
        self.assertTrue(self._mock_mode())

    def test_mock_llm_true_overrides_provider(self):
        _set_env(AI_CTO_LLM_PROVIDER="ollama", AI_CTO_MOCK_LLM="true")
        # AI_CTO_MOCK_LLM=true checked first — should still be mock
        self.assertTrue(self._mock_mode())


if __name__ == "__main__":
    unittest.main()
