"""
ollama.py — Ollama local LLM provider.

Uses stdlib urllib only — no extra dependencies required.

Configuration (env vars):
    OLLAMA_BASE_URL   Base URL of the Ollama server  (default: http://localhost:11434)
    OLLAMA_MODEL      Model tag to use               (default: deepseek-coder:latest)

Ollama API contract used here:
    POST /api/chat  (streaming mode — stream: true)
    Body: {"model": "...", "messages": [...], "stream": true,
           "options": {"num_predict": max_tokens}}
    Response: one NDJSON line per token:
        {"message": {"content": "<token>"}, "done": false}
        ...
        {"message": {"content": ""}, "done": true}

Streaming is used instead of stream:false to avoid socket timeouts on slow
CPU inference. With stream:false the server sends nothing until the entire
response is generated; on a 6.7b model that can exceed 300 s easily. With
stream:true each token arrives within seconds, keeping the socket alive.

See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
"""

import json
import logging
import os
import urllib.request
import urllib.error
from ai_cto.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL    = "deepseek-coder:latest"


class OllamaProvider(LLMProvider):
    """LLM provider backed by a locally-running Ollama server."""

    def __init__(self) -> None:
        self._base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self._model    = os.environ.get("OLLAMA_MODEL",    _DEFAULT_MODEL)
        logger.info(
            "[OllamaProvider] Initialized. base_url=%s  model=%s",
            self._base_url, self._model,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        Send a streaming chat completion request to Ollama and return the
        full response text (all token chunks concatenated).

        Uses stream:true so each token arrives within seconds, preventing
        socket timeout on slow CPU inference.  The per-read socket timeout
        is 300 s — covers prefill time for long prompts (memory context
        injection can push input to 1000+ tokens, taking 50-150 s on CPU
        before the first output token arrives).

        Raises:
            RuntimeError: If Ollama is unreachable, returns a non-200 status,
                          or produces an unexpected response format.
        """
        url = f"{self._base_url}/api/chat"
        payload = {
            "model":    self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                chunks: list[str] = []
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        chunks.append(token)
                    if chunk.get("done"):
                        break
                return "".join(chunks)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OllamaProvider: cannot reach {url} — {exc}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc

    # ── Health check helper ────────────────────────────────────────────────────

    def is_reachable(self) -> bool:
        """
        Return True if the Ollama server responds to a lightweight GET /api/tags.
        Useful for verify_ollama.py and smoke tests — never called by the pipeline.
        """
        url = f"{self._base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except Exception:  # noqa: BLE001
            return False
