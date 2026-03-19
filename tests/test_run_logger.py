"""
test_run_logger.py — Unit tests for RunLogger.

Tests cover event recording, save(), duration computation, and error handling.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ai_cto.tools.run_logger import RunLogger, _compute_duration


# ── _compute_duration ───────────────────────────────────────────────────────────

class TestComputeDuration:
    def test_round_trip(self):
        start = "2025-01-01T00:00:00+00:00"
        end   = "2025-01-01T00:00:45+00:00"
        assert _compute_duration(start, end) == 45

    def test_returns_none_on_bad_input(self):
        assert _compute_duration("not-a-date", "also-bad") is None

    def test_zero_seconds(self):
        ts = "2025-01-01T12:00:00+00:00"
        assert _compute_duration(ts, ts) == 0


# ── RunLogger.record_event ──────────────────────────────────────────────────────

class TestRunLoggerRecordEvent:
    def _logger(self):
        return RunLogger("test-proj")

    def test_event_is_appended(self):
        rl = self._logger()
        rl.record_event("coding_node", {"status": "coding", "active_task_id": 1})
        assert len(rl._events) == 1
        assert rl._events[0]["node"] == "coding_node"
        assert rl._events[0]["status"] == "coding"
        assert rl._events[0]["active_task_id"] == 1

    def test_multiple_events_appended_in_order(self):
        rl = self._logger()
        rl.record_event("planning_node", {"status": "planning"})
        rl.record_event("coding_node",   {"status": "coding"})
        assert [e["node"] for e in rl._events] == ["planning_node", "coding_node"]

    def test_error_snippet_captured(self):
        rl = self._logger()
        rl.record_event("debug_node", {"status": "debugging", "error_context": "x" * 300})
        snippet = rl._events[0].get("error_snippet", "")
        assert len(snippet) == 200   # truncated to 200

    def test_no_error_snippet_when_no_error(self):
        rl = self._logger()
        rl.record_event("coding_node", {"status": "coding"})
        assert "error_snippet" not in rl._events[0]

    def test_timestamp_present(self):
        rl = self._logger()
        rl.record_event("coding_node", {"status": "coding"})
        assert "timestamp" in rl._events[0]


# ── RunLogger.save ──────────────────────────────────────────────────────────────

class TestRunLoggerSave:
    def _final_state(self, **overrides):
        base = {
            "project_id": "test-proj",
            "idea": "Build something",
            "status": "done",
            "project_status": "complete",
            "tasks": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "failed"},
                {"id": 3, "status": "blocked"},
            ],
            "generated_files": {"main.py": "print(1)", "utils.py": ""},
        }
        return {**base, **overrides}

    def test_writes_json_file(self, tmp_path):
        with patch("ai_cto.tools.run_logger._workspace_root", return_value=tmp_path):
            rl = RunLogger("test-proj")
            rl.record_event("coding_node", {"status": "coding"})
            out = rl.save(self._final_state())
            assert out is not None
            assert out.exists()

    def test_summary_structure(self, tmp_path):
        with patch("ai_cto.tools.run_logger._workspace_root", return_value=tmp_path):
            rl = RunLogger("test-proj")
            rl.record_event("planning_node", {"status": "planning"})
            out = rl.save(self._final_state())
            data = json.loads(out.read_text())

            assert data["project_id"] == "test-proj"
            assert data["idea"] == "Build something"
            assert data["files_generated_count"] == 2
            assert sorted(data["files_generated"]) == ["main.py", "utils.py"]
            assert len(data["events"]) == 1
            # status is now a ProjectStatus report dict
            assert data["status"]["status"] == "completed"
            assert data["status"]["tasks_total"] == 3
            assert data["status"]["tasks_completed"] == 1
            assert data["status"]["tasks_failed"] == 1
            assert data["status"]["tasks_blocked"] == 1

    def test_duration_present_and_non_negative(self, tmp_path):
        with patch("ai_cto.tools.run_logger._workspace_root", return_value=tmp_path):
            rl = RunLogger("test-proj")
            out = rl.save(self._final_state())
            data = json.loads(out.read_text())
            assert data["duration_seconds"] is not None
            assert data["duration_seconds"] >= 0

    def test_save_returns_none_on_error(self, tmp_path):
        with patch("ai_cto.tools.run_logger._workspace_root", return_value=tmp_path):
            rl = RunLogger("test-proj")
            # Force a write failure by making the workspace unwritable via mock
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                result = rl.save(self._final_state())
            assert result is None
