"""
test_production_fixes.py — Tests for the 5 production hardening fixes.

Fix 1: Architecture persisted and restored through task_board.json
Fix 2: CodingAgent retries on JSON parse failure; fails gracefully on exhaustion
Fix 3: CodingAgent injects patched files when debug_attempts > 0, no error_context
Fix 4: ExecutionAgent distinguishes timeout from code bug; fails after 2 consecutive
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["AI_CTO_MOCK_LLM"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_task(id=1, title="Task", status="pending", depends_on=None, retries=0, max_retries=3):
    from ai_cto.state import make_task_item
    t = make_task_item(id=id, title=title, description="", depends_on=depends_on or [])
    return {**t, "status": status, "retries": retries, "max_retries": max_retries}


def _base_state(**overrides):
    from ai_cto.state import initial_state
    s = initial_state(idea="Test project", project_id="test-proj")
    s["architecture"] = "# Architecture\nSimple REST API."
    s["tasks"] = [_make_task(1, "Build API")]
    s["active_task_id"] = 1
    s["current_task_index"] = 0
    return {**s, **overrides}


def _write_board(tmp_path: Path, board: dict) -> Path:
    project_dir = tmp_path / board["project_id"]
    project_dir.mkdir(parents=True, exist_ok=True)
    board_path = project_dir / "task_board.json"
    board_path.write_text(json.dumps(board, indent=2), encoding="utf-8")
    return board_path


# ── Fix 1: Architecture round-trips through persist / resume ──────────────────

class TestFix1ArchitecturePersistence:
    def test_architecture_saved_in_board(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import persist_task_board, load_task_board

        state = _base_state()
        state["architecture"] = "# Architecture\nREST API with PostgreSQL."
        board_path = persist_task_board(state)

        board = load_task_board(board_path)
        assert board["architecture"] == "# Architecture\nREST API with PostgreSQL."

    def test_architecture_empty_string_when_absent(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import persist_task_board, load_task_board

        state = _base_state()
        state["architecture"] = ""
        board_path = persist_task_board(state)

        board = load_task_board(board_path)
        assert board["architecture"] == ""

    def test_architecture_restored_on_resume(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board

        arch = "# Architecture\nGraphQL + Redis cache."
        board = {
            "project_id": "arch-test",
            "project_goal": "Build API",
            "architecture": arch,
            "project_status": "active",
            "active_task_id": 1,
            "tasks": [_make_task(1, "Task one", status="pending")],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "blocked_task_ids": [],
            "task_history": [],
        }
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert state["architecture"] == arch

    def test_architecture_defaults_to_empty_string_when_missing_from_board(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board

        board = {
            "project_id": "arch-missing",
            "project_goal": "Build API",
            # no "architecture" key
            "project_status": "active",
            "active_task_id": 1,
            "tasks": [_make_task(1, "Task one", status="pending")],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "blocked_task_ids": [],
            "task_history": [],
        }
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert state["architecture"] == ""

    def test_resumed_coding_sees_architecture(self, tmp_path):
        """After resume, the architecture field is non-empty in the state."""
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board

        arch = "# Architecture\nEvent-driven microservices."
        board = {
            "project_id": "arch-roundtrip",
            "project_goal": "Build pipeline",
            "architecture": arch,
            "project_status": "active",
            "active_task_id": 1,
            "tasks": [_make_task(1, "Setup", status="pending")],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "blocked_task_ids": [],
            "task_history": [],
        }
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        # Architecture is visible and non-empty — coding will see it
        assert state["architecture"] == arch
        assert len(state["architecture"]) > 0


# ── Fix 2: CodingAgent JSON parse retry / graceful failure ─────────────────────

class TestFix2CodingAgentParseRetry:
    """Tests require the real coding_node (not mock mode)."""

    def _make_mock_provider(self, responses):
        """responses: list of strings returned by successive .complete() calls."""
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = list(responses)
        return mock_provider

    def test_succeeds_on_first_attempt(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        good_json = json.dumps({
            "files": {"main.py": "print('hello')"},
            "entry_point": "main.py",
        })
        mock_provider = self._make_mock_provider([good_json])

        state = _base_state()
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        assert result["status"] == "executing"
        assert "main.py" in result["generated_files"]

    def test_succeeds_on_second_attempt_after_bad_json(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        bad = "Sorry, I can't help with that."
        good = json.dumps({
            "files": {"app.py": "x = 1"},
            "entry_point": "app.py",
        })
        mock_provider = self._make_mock_provider([bad, good])

        state = _base_state()
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        assert result["status"] == "executing"
        assert mock_provider.complete.call_count == 2

    def test_returns_verifying_failed_after_all_retries(self):
        from ai_cto.agents.coding import coding_node

        bad = "This is not JSON at all."
        mock_provider = self._make_mock_provider([bad, bad, bad])

        state = _base_state()
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        assert result["status"] == "verifying_failed"
        assert "failed to produce valid JSON" in result["error_context"].lower() or \
               "failed to produce valid JSON" in result["error_context"]

    def test_handles_markdown_fenced_json(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        fenced = "```json\n" + json.dumps({
            "files": {"main.py": "print('ok')"},
            "entry_point": "main.py",
        }) + "\n```"
        mock_provider = self._make_mock_provider([fenced])

        state = _base_state()
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        assert result["status"] == "executing"

    def test_handles_prose_before_json(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        prose_wrapped = "Here is the implementation:\n" + json.dumps({
            "files": {"svc.py": "pass"},
            "entry_point": "svc.py",
        }) + "\nHope this helps!"
        mock_provider = self._make_mock_provider([prose_wrapped])

        state = _base_state()
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        assert result["status"] == "executing"


# ── Fix 3: Patched-file injection when debug_attempts > 0 ─────────────────────

class TestFix3PatchedFileInjection:
    def test_no_injection_on_first_attempt(self):
        """When debug_attempts == 0 and no error_context, no extra context added."""
        from ai_cto.agents.coding import _build_user_message

        state = _base_state(debug_attempts=0, error_context="", generated_files={})
        task = state["tasks"][0]
        msg = _build_user_message(state, task)

        assert "Patched" not in msg
        assert "Previously patched" not in msg

    def test_error_context_injected_when_present(self):
        """When error_context is set, include it with previously generated files."""
        from ai_cto.agents.coding import _build_user_message

        state = _base_state(
            debug_attempts=1,
            error_context="NameError: x not defined",
            generated_files={"main.py": "x = undefined_var"},
        )
        task = state["tasks"][0]
        msg = _build_user_message(state, task)

        assert "NameError" in msg
        assert "main.py" in msg
        assert "Previously patched" not in msg

    def test_patched_files_injected_when_debug_attempts_positive_no_error(self):
        """When debug_attempts > 0 and error_context is empty, inject patched files."""
        from ai_cto.agents.coding import _build_user_message

        state = _base_state(
            debug_attempts=1,
            error_context="",
            generated_files={"main.py": "def foo(): pass"},
        )
        task = state["tasks"][0]
        msg = _build_user_message(state, task)

        assert "Previously patched files" in msg
        assert "main.py" in msg
        assert "def foo(): pass" in msg
        assert "preserve these fixes" in msg

    def test_patched_files_not_injected_when_generated_files_empty(self):
        """No injection if generated_files is empty even with debug_attempts > 0."""
        from ai_cto.agents.coding import _build_user_message

        state = _base_state(
            debug_attempts=2,
            error_context="",
            generated_files={},
        )
        task = state["tasks"][0]
        msg = _build_user_message(state, task)

        assert "Previously patched" not in msg

    def test_multiple_patched_files_all_injected(self):
        """All patched files appear in the message."""
        from ai_cto.agents.coding import _build_user_message

        state = _base_state(
            debug_attempts=1,
            error_context="",
            generated_files={
                "app.py": "from flask import Flask",
                "models.py": "class User: pass",
            },
        )
        task = state["tasks"][0]
        msg = _build_user_message(state, task)

        assert "app.py" in msg
        assert "models.py" in msg


# ── Fix 4: Execution timeout handling ─────────────────────────────────────────

class TestFix4TimeoutHandling:
    def _make_exec_result(self, exit_code, stderr="", stdout=""):
        return {"exit_code": exit_code, "stderr": stderr, "stdout": stdout}

    def _run_execution_node(self, state, result):
        """Run execution_node with a mocked sandbox call."""
        from ai_cto.agents.execution import execution_node
        with patch("ai_cto.agents.execution.docker_available", return_value=False):
            with patch("ai_cto.agents.execution.run_in_subprocess", return_value=result):
                return execution_node(state)

    def test_success_clears_timeout_count(self):
        state = _base_state(execution_timeout_count=1)
        result = self._make_exec_result(exit_code=0)
        out = self._run_execution_node(state, result)
        assert out["status"] == "done"
        assert out["execution_timeout_count"] == 0

    def test_first_timeout_routes_to_debug(self):
        state = _base_state(execution_timeout_count=0)
        result = self._make_exec_result(
            exit_code=1, stderr="Execution timed out after 30s"
        )
        out = self._run_execution_node(state, result)
        assert out["status"] == "debugging"
        assert out["execution_timeout_count"] == 1
        assert "[TIMEOUT]" in out["error_context"]

    def test_second_consecutive_timeout_fails_task(self):
        state = _base_state(execution_timeout_count=1)
        result = self._make_exec_result(
            exit_code=1, stderr="Execution timed out after 30s"
        )
        out = self._run_execution_node(state, result)
        assert out["status"] == "failed"
        assert out["execution_timeout_count"] == 2
        assert "[TIMEOUT]" in out["error_context"]

    def test_non_timeout_failure_resets_timeout_count(self):
        state = _base_state(execution_timeout_count=1)
        result = self._make_exec_result(
            exit_code=1, stderr="NameError: name 'x' is not defined"
        )
        out = self._run_execution_node(state, result)
        assert out["status"] == "debugging"
        assert out["execution_timeout_count"] == 0

    def test_non_timeout_failure_has_no_timeout_prefix(self):
        state = _base_state(execution_timeout_count=0)
        result = self._make_exec_result(
            exit_code=1, stderr="SyntaxError: invalid syntax"
        )
        out = self._run_execution_node(state, result)
        assert "[TIMEOUT]" not in out["error_context"]

    def test_timeout_count_reset_on_task_advance(self):
        """advance_task_node resets execution_timeout_count to 0."""
        from ai_cto.state import make_task_item
        from ai_cto.graph import advance_task_node
        import ai_cto.tools.file_writer as _fw
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _fw.WORKSPACE_ROOT = Path(tmp)
            tasks = [
                {**make_task_item(1, "Task A", ""), "status": "in_progress",
                 "retries": 0, "max_retries": 3},
                {**make_task_item(2, "Task B", ""), "status": "pending",
                 "retries": 0, "max_retries": 3},
            ]
            state = _base_state(
                tasks=tasks,
                active_task_id=1,
                current_task_index=0,
                execution_timeout_count=1,
            )
            out = advance_task_node(state)
            assert out["execution_timeout_count"] == 0

    def test_route_after_execution_returns_end_on_failed(self):
        from ai_cto.graph import route_after_execution
        state = _base_state(status="failed")
        assert route_after_execution(state) == "end"

    def test_route_after_execution_returns_debug_on_debugging(self):
        from ai_cto.graph import route_after_execution
        state = _base_state(status="debugging")
        assert route_after_execution(state) == "debug"

    def test_route_after_execution_returns_advance_on_done(self):
        from ai_cto.graph import route_after_execution
        state = _base_state(status="done")
        assert route_after_execution(state) == "advance_task"
