"""
test_generator.py — LLM-based pytest test generation.

Given the generated source files and architecture context, asks Claude to
write a minimal but meaningful pytest test suite.

The generated tests are:
  - Written to workspace/<project_id>/tests/ on disk (for manual use)
  - Stored in VerificationResult.generated_tests (for inspection)
  - NOT executed automatically in V1 (test execution is a V2 feature)

Design decisions:
  - One LLM call per verification pass (not per file) to keep costs low
  - Returns {} gracefully if the code is a pure script with nothing to test
  - Test files are prefixed with 'test_' as pytest convention requires
"""

import json
import re
import logging
from ai_cto.providers import get_provider
from ai_cto.mock_llm import is_mock_mode, mock_generate_tests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software test engineer.
Given source code files and a project architecture description, write a minimal pytest test suite.

Return your response as valid JSON with this exact shape:
{
  "files": {
    "tests/test_<module_name>.py": "<full pytest file contents>",
    ...
  }
}

Rules:
- Write pytest-style tests only (def test_*).
- Focus on testing pure functions and critical logic — skip untestable I/O scripts.
- Mock all external dependencies: HTTP calls, database connections, file I/O.
- Each test file must be independently importable (no shared state between files).
- Keep each test file under 80 lines — prefer a few meaningful tests over exhaustive coverage.
- If there is nothing unit-testable (e.g. a pure main script with no functions), return {"files": {}}.
- Do not test __main__ blocks or CLI entry points.
"""


def generate_tests(
    files: dict[str, str],
    architecture: str,
    project_id: str,
) -> dict[str, str]:
    """
    Call Claude to generate pytest tests for the generated code.

    Args:
        files        — the {filepath: content} dict from CodingAgent
        architecture — the architecture document for context
        project_id   — used only for logging

    Returns:
        dict of {test_filepath: test_content}, may be empty if nothing is testable.
    """
    # Only include .py source files (not requirements.txt, configs, etc.)
    py_sources = {
        path: content
        for path, content in files.items()
        if path.endswith(".py") and not path.startswith("tests/")
    }

    if not py_sources:
        logger.info("[TestGenerator] No Python source files found — skipping test generation.")
        return {}

    if is_mock_mode():
        return mock_generate_tests(project_id)

    logger.info(
        "[TestGenerator] Generating tests for %d file(s) in project '%s'.",
        len(py_sources),
        project_id,
    )

    provider = get_provider()

    user_message = (
        f"Architecture:\n{architecture}\n\n"
        f"Source files:\n{json.dumps(py_sources, indent=2)}"
    )

    try:
        raw = provider.complete(
            system=SYSTEM_PROMPT,
            user=user_message,
            max_tokens=2048,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TestGenerator] API call failed: %s — skipping test generation.", exc)
        return {}
    logger.debug("[TestGenerator] Raw response:\n%s", raw)

    json_str = _extract_json(raw)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("[TestGenerator] Failed to parse response JSON: %s", exc)
        return {}

    generated: dict[str, str] = parsed.get("files", {})
    logger.info("[TestGenerator] Generated %d test file(s).", len(generated))
    return generated


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text
