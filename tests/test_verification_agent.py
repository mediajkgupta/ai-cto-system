"""
Tests for agents/verification.py — the verification node.

LLM call (test_generator) is mocked in all tests.
Filesystem writes are monkeypatched.
"""

import pytest
from unittest.mock import patch, MagicMock
from ai_cto.state import initial_state
from ai_cto.agents.verification import verification_node


def _base_state(**overrides):
    s = initial_state("Build a todo API", "test-proj")
    s["_entry_point"] = "main.py"
    s.update(overrides)
    return s


# Shared patch targets
_WRITE_FILES = "ai_cto.agents.verification.write_files"
_GEN_TESTS   = "ai_cto.agents.verification.generate_tests"


# ── Routing ────────────────────────────────────────────────────────────────────

class TestVerificationRouting:

    def test_clean_code_routes_to_execution(self):
        state = _base_state(generated_files={"main.py": "print('hello')\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["status"] == "executing"

    def test_syntax_error_routes_to_debug(self):
        state = _base_state(generated_files={"main.py": "def broken(\n    pass\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["status"] == "verifying_failed"

    def test_missing_entry_point_routes_to_debug(self):
        state = _base_state(
            generated_files={"utils.py": "def add(a, b): return a+b\n"},
            _entry_point="main.py",
        )
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["status"] == "verifying_failed"

    def test_warnings_only_routes_to_execution(self):
        # Missing dep is WARNING, not error — should still pass
        state = _base_state(generated_files={"main.py": "import requests\nprint('ok')\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["status"] == "executing"

    def test_divide_by_zero_is_error_routes_to_debug(self):
        state = _base_state(generated_files={"main.py": "x = 10 / 0\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["status"] == "verifying_failed"


# ── State mutations ────────────────────────────────────────────────────────────

class TestVerificationStateMutations:

    def test_verification_result_attached_on_pass(self):
        state = _base_state(generated_files={"main.py": "x = 1\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["verification_result"] is not None
        assert result["verification_result"]["passed"] is True

    def test_verification_result_attached_on_fail(self):
        state = _base_state(generated_files={"main.py": "def broken(\n    pass\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["verification_result"] is not None
        assert result["verification_result"]["passed"] is False

    def test_error_context_set_on_failure(self):
        state = _base_state(generated_files={"main.py": "def broken(\n    pass\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["error_context"] != ""
        assert "VERIFICATION FAILED" in result["error_context"]

    def test_error_context_cleared_on_pass(self):
        state = _base_state(
            generated_files={"main.py": "x = 1\n"},
            error_context="old error from previous attempt",
        )
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["error_context"] == ""

    def test_other_state_fields_preserved(self):
        state = _base_state(
            generated_files={"main.py": "x = 1\n"},
            idea="Build a todo API",
            project_id="test-proj",
            architecture="## Arch\nFastAPI.",
        )
        with patch(_WRITE_FILES), patch(_GEN_TESTS, return_value={}):
            result = verification_node(state)
        assert result["idea"] == "Build a todo API"
        assert result["project_id"] == "test-proj"
        assert result["architecture"] == "## Arch\nFastAPI."


# ── Test generation integration ────────────────────────────────────────────────

class TestTestGeneration:

    def test_generated_tests_stored_in_result(self):
        mock_tests = {"tests/test_main.py": "def test_add(): assert 1+1 == 2\n"}
        state = _base_state(generated_files={"main.py": "def add(a,b): return a+b\n"})
        with patch(_WRITE_FILES) as mock_write, patch(_GEN_TESTS, return_value=mock_tests):
            result = verification_node(state)
        assert result["verification_result"]["generated_tests"] == mock_tests

    def test_generated_tests_written_to_disk(self):
        mock_tests = {"tests/test_main.py": "def test_x(): pass\n"}
        state = _base_state(generated_files={"main.py": "x = 1\n"})
        with patch(_WRITE_FILES) as mock_write, patch(_GEN_TESTS, return_value=mock_tests):
            verification_node(state)
        # write_files should be called once for the test files
        mock_write.assert_called_once_with(project_id="test-proj", files=mock_tests)

    def test_test_generator_failure_is_non_fatal(self):
        """If test_generator raises, verification should still complete."""
        state = _base_state(generated_files={"main.py": "x = 1\n"})
        with patch(_WRITE_FILES), patch(_GEN_TESTS, side_effect=RuntimeError("LLM down")):
            result = verification_node(state)
        # Should still pass and not crash
        assert result["status"] == "executing"
        assert result["verification_result"]["generated_tests"] == {}

    def test_no_tests_written_when_generation_returns_empty(self):
        state = _base_state(generated_files={"main.py": "x = 1\n"})
        with patch(_WRITE_FILES) as mock_write, patch(_GEN_TESTS, return_value={}):
            verification_node(state)
        mock_write.assert_not_called()
