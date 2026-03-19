"""
test_execution_stack_aware.py — Tests for stack-aware entry point resolution
in the ExecutionAgent (_resolve_entry_point) and the updated execution_node
that passes files to the sandbox.
"""

import pytest
from unittest.mock import patch, MagicMock

from ai_cto.agents.execution import _resolve_entry_point


# ── _resolve_entry_point ────────────────────────────────────────────────────────

class TestResolveEntryPointPython:
    def test_python_entry_unchanged(self):
        files = {"main.py": "print('hi')"}
        assert _resolve_entry_point(files, "main.py") == "main.py"

    def test_custom_python_entry_unchanged(self):
        files = {"app.py": "import flask"}
        assert _resolve_entry_point(files, "app.py") == "app.py"

    def test_empty_files_keeps_entry(self):
        # unknown stack — no override
        assert _resolve_entry_point({}, "main.py") == "main.py"


class TestResolveEntryPointNode:
    def test_python_default_overridden_for_node_project(self):
        files = {"server.js": "const app = require('express')();"}
        result = _resolve_entry_point(files, "main.py")
        assert result == "server.js"

    def test_empty_entry_overridden_for_node_project(self):
        files = {"main.py": "", "app.js": "const x = 1;"}
        result = _resolve_entry_point(files, "main.py")
        assert result == "app.js"

    def test_explicit_js_entry_kept_when_non_empty(self):
        files = {"server.js": "const x = 1;", "app.js": "const y = 2;"}
        # server.js is non-empty and named explicitly → keep it
        result = _resolve_entry_point(files, "server.js")
        assert result == "server.js"

    def test_node_priority_server_js_over_app_js(self):
        files = {
            "server.js": "const s = 1;",
            "app.js": "const a = 1;",
        }
        result = _resolve_entry_point(files, "main.py")
        assert result == "server.js"

    def test_node_falls_back_to_index_js(self):
        files = {"index.js": "module.exports = {}"}
        result = _resolve_entry_point(files, "main.py")
        assert result == "index.js"

    def test_node_no_js_entry_found_keeps_original(self):
        # All JS files empty, no good entry → keep original
        files = {"server.js": "", "package.json": "{}"}
        result = _resolve_entry_point(files, "main.py")
        # detect_node_entry returns None (all JS files empty); falls back
        assert result == "main.py"


# ── execution_node passes files to sandbox ─────────────────────────────────────

class TestExecutionNodePassesFiles:
    """Verify that execution_node calls run_in_docker / run_in_subprocess with files=."""

    def _state(self, files=None):
        from ai_cto.state import initial_state
        state = initial_state("test idea", "test-proj")
        state["generated_files"] = files or {"main.py": "print(1)"}
        state["status"] = "executing"
        return state

    def _mock_result(self):
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    def test_docker_receives_files_kwarg(self, tmp_path):
        from ai_cto.agents.execution import execution_node
        state = self._state({"main.py": "print(1)"})

        with patch("ai_cto.agents.execution.docker_available", return_value=True), \
             patch("ai_cto.agents.execution.run_in_docker", return_value=self._mock_result()) as mock_docker, \
             patch("ai_cto.agents.execution._workspace_path", return_value=str(tmp_path)):
            execution_node(state)

        call_kwargs = mock_docker.call_args
        assert call_kwargs.kwargs.get("files") == {"main.py": "print(1)"}

    def test_subprocess_receives_files_kwarg(self, tmp_path):
        from ai_cto.agents.execution import execution_node
        state = self._state({"server.js": "const x=1;"})

        with patch("ai_cto.agents.execution.docker_available", return_value=False), \
             patch("ai_cto.agents.execution.run_in_subprocess", return_value=self._mock_result()) as mock_sub, \
             patch("ai_cto.agents.execution._workspace_path", return_value=str(tmp_path)):
            execution_node(state)

        call_kwargs = mock_sub.call_args
        assert call_kwargs.kwargs.get("files") == {"server.js": "const x=1;"}

    def test_node_project_entry_resolved_before_docker_call(self, tmp_path):
        from ai_cto.agents.execution import execution_node
        files = {"server.js": "const app = require('express')();"}
        state = self._state(files)
        # Default _entry_point is main.py (not in state) — should be overridden

        with patch("ai_cto.agents.execution.docker_available", return_value=True), \
             patch("ai_cto.agents.execution.run_in_docker", return_value=self._mock_result()) as mock_docker, \
             patch("ai_cto.agents.execution._workspace_path", return_value=str(tmp_path)):
            execution_node(state)

        call_kwargs = mock_docker.call_args
        assert call_kwargs.kwargs.get("entry_point") == "server.js"
