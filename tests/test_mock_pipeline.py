"""
test_mock_pipeline.py — Full pipeline integration tests using MOCK_LLM mode.

These tests run the complete agent graph without any API calls:
    memory_retrieve → planning → coding → verification → execution
                                                             ↓ success
                                              advance_task → memory_save → END

Two test classes:
    TestMockMode        — unit tests for is_mock_mode() and mock data contracts
    TestFullPipeline    — end-to-end graph run in mock mode, including real subprocess
                          execution of the generated main.py

The workspace is written to a tmp directory and cleaned up after each test.
"""

import os
import sys
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    """Ensure every test in this module runs with mock mode active."""
    monkeypatch.setenv("AI_CTO_MOCK_LLM", "true")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """
    Redirect WORKSPACE_ROOT to a temp directory so tests don't pollute
    the real workspace/ folder and are automatically cleaned up.

    Patching file_writer.WORKSPACE_ROOT is sufficient because:
      - write_files() reads it at call time
      - execution._workspace_path() reads it via the module reference at call time
    """
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
    return tmp_path


# ── TestMockMode ───────────────────────────────────────────────────────────────

class TestMockMode:
    """Unit tests for mock_llm.py — mode detection and data contracts."""

    def test_is_mock_mode_true_when_env_set(self, monkeypatch):
        from ai_cto.mock_llm import is_mock_mode
        monkeypatch.setenv("AI_CTO_MOCK_LLM", "true")
        assert is_mock_mode() is True

    def test_is_mock_mode_true_without_api_key(self, monkeypatch):
        from ai_cto.mock_llm import is_mock_mode
        monkeypatch.delenv("AI_CTO_MOCK_LLM", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert is_mock_mode() is True

    def test_is_mock_mode_false_with_key_and_no_flag(self, monkeypatch):
        from ai_cto.mock_llm import is_mock_mode
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
        monkeypatch.delenv("AI_CTO_MOCK_LLM", raising=False)
        assert is_mock_mode() is False

    def test_mock_tasks_are_valid_task_items(self):
        from ai_cto.mock_llm import MOCK_TASKS
        assert len(MOCK_TASKS) >= 1
        for task in MOCK_TASKS:
            assert "id" in task
            assert "title" in task
            assert "description" in task
            assert "status" in task

    def test_mock_architecture_is_non_empty_string(self):
        from ai_cto.mock_llm import MOCK_ARCHITECTURE
        assert isinstance(MOCK_ARCHITECTURE, str)
        assert len(MOCK_ARCHITECTURE) > 50

    def test_mock_main_py_is_valid_python(self):
        import ast
        from ai_cto.mock_llm import MOCK_MAIN_PY
        ast.parse(MOCK_MAIN_PY)   # raises SyntaxError if invalid

    def test_mock_generated_files_has_entry_point(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        assert MOCK_ENTRY_POINT in MOCK_GENERATED_FILES

    def test_mock_test_file_is_valid_python(self):
        import ast
        from ai_cto.mock_llm import MOCK_TEST_FILE
        ast.parse(MOCK_TEST_FILE)

    def test_mock_generated_tests_key_starts_with_tests(self):
        from ai_cto.mock_llm import MOCK_GENERATED_TESTS
        for path in MOCK_GENERATED_TESTS:
            assert path.startswith("tests/"), f"Expected test path to start with tests/, got {path}"


# ── TestMockCode ───────────────────────────────────────────────────────────────

class TestMockCode:
    """Verify the mock main.py actually passes all 5 verification checkers."""

    def test_passes_syntax_checker(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import SyntaxChecker
        issues = SyntaxChecker().check(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        assert issues == [], f"Unexpected syntax issues: {issues}"

    def test_passes_security_checker(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import SecurityChecker
        issues = SecurityChecker().check(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        errors = [i for i in issues if i["severity"] == "error"]
        assert errors == [], f"Unexpected security errors: {errors}"

    def test_passes_dependency_checker(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import DependencyChecker
        issues = DependencyChecker().check(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        assert issues == [], f"Unexpected dependency issues: {issues}"

    def test_passes_architecture_checker(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import ArchitectureChecker
        issues = ArchitectureChecker().check(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        errors = [i for i in issues if i["severity"] == "error"]
        assert errors == [], f"Unexpected architecture errors: {errors}"

    def test_passes_runtime_risk_checker(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import RuntimeRiskChecker
        issues = RuntimeRiskChecker().check(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        errors = [i for i in issues if i["severity"] == "error"]
        assert errors == [], f"Unexpected runtime errors: {errors}"

    def test_run_all_checks_passes(self):
        from ai_cto.mock_llm import MOCK_GENERATED_FILES, MOCK_ENTRY_POINT
        from ai_cto.verification.checks import run_all_checks
        result = run_all_checks(MOCK_GENERATED_FILES, MOCK_ENTRY_POINT)
        assert result["passed"] is True, f"Verification failed:\n{result['summary']}"
        assert result["error_count"] == 0

    def test_main_py_executes_successfully(self, tmp_path):
        """The mock main.py must run to completion with exit code 0."""
        from ai_cto.mock_llm import MOCK_MAIN_PY
        import subprocess
        main_file = tmp_path / "main.py"
        main_file.write_text(MOCK_MAIN_PY, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(main_file)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"main.py exited {result.returncode}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert "All operations passed" in result.stdout


# ── TestFullPipeline ───────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    End-to-end integration: run the complete LangGraph pipeline in mock mode.
    No API calls are made. Real subprocess execution of generated code.
    """

    def _run_pipeline(self, tmp_workspace, idea="Build a REST API for a todo app",
                      project_id="mock-todo"):
        """
        Build the graph and run it to completion.
        tmp_workspace fixture has already redirected WORKSPACE_ROOT.
        """
        from ai_cto.state import initial_state
        from ai_cto.graph import build_graph

        state = initial_state(idea=idea, project_id=project_id)
        graph = build_graph()

        final = None
        for event in graph.stream(state):
            node_name = list(event.keys())[0]
            final = event[node_name]
            status = final.get("status", "?")
            print(f"  [{node_name.upper()}] status={status}")

        return final

    def test_pipeline_completes_with_status_done(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        assert final is not None
        assert final["status"] == "done", (
            f"Expected status=done, got {final['status']}\n"
            f"final_output: {final.get('final_output', '')}"
        )

    def test_pipeline_architecture_is_populated(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        assert final["architecture"] != ""
        assert "Todo" in final["architecture"]

    def test_pipeline_all_tasks_completed(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        for task in final["tasks"]:
            assert task["status"] == "done", f"Task not done: {task}"

    def test_pipeline_generated_files_on_disk(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        project_dir = tmp_workspace / "mock-todo"
        assert (project_dir / "main.py").exists()
        assert (project_dir / "requirements.txt").exists()

    def test_pipeline_verification_passed(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        vr = final.get("verification_result")
        assert vr is not None
        assert vr["passed"] is True
        assert vr["error_count"] == 0

    def test_pipeline_tests_generated(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        vr = final.get("verification_result")
        assert vr is not None
        assert len(vr["generated_tests"]) > 0
        assert any("test_" in k for k in vr["generated_tests"])

    def test_pipeline_final_output_contains_summary(self, tmp_workspace):
        final = self._run_pipeline(tmp_workspace)
        output = final.get("final_output", "")
        assert "mock-todo" in output
        assert "completed" in output.lower()

    def test_pipeline_no_debug_needed_on_clean_code(self, tmp_workspace):
        """Mock code is correct so debug should never be invoked."""
        final = self._run_pipeline(tmp_workspace)
        assert final["debug_attempts"] == 0

    def test_pipeline_memory_saves_architecture(self, tmp_workspace):
        """After completion, architecture and project pattern should be stored in Chroma."""
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.memory.schema import MemoryType
        import ai_cto.agents.memory as _mem_module

        chroma_path = tmp_workspace / "test_chroma"
        store = ChromaMemoryStore(db_path=chroma_path)

        # Reset the shared store singleton and point it at our isolated chroma
        original_store = _mem_module._store
        _mem_module._store = store
        try:
            final = self._run_pipeline(tmp_workspace, project_id="mem-test")
        finally:
            _mem_module._store = original_store

        assert store.count(MemoryType.ARCHITECTURE) >= 1
        assert store.count(MemoryType.PROJECT_PATTERN) >= 1

    def test_different_ideas_dont_interfere(self, tmp_workspace):
        """Two pipeline runs with different project IDs should not conflict."""
        final_a = self._run_pipeline(tmp_workspace, project_id="proj-a")
        final_b = self._run_pipeline(tmp_workspace, project_id="proj-b")
        assert final_a["status"] == "done"
        assert final_b["status"] == "done"
        assert (tmp_workspace / "proj-a" / "main.py").exists()
        assert (tmp_workspace / "proj-b" / "main.py").exists()
