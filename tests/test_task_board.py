"""
test_task_board.py — Unit tests for ai_cto/tools/task_board.py

Covers all helper functions:
  - mark_task_done       partial state update
  - mark_task_failed     partial state update
  - mark_task_blocked    partial state update
  - get_next_runnable_task
  - project_is_complete / project_is_blocked
  - persist_task_board   JSON file written to workspace
"""

import json
import pytest
from pathlib import Path

from ai_cto.state import initial_state, make_task_item
from ai_cto.tools.task_board import (
    mark_task_done,
    mark_task_failed,
    mark_task_blocked,
    get_next_runnable_task,
    project_is_complete,
    project_is_blocked,
    persist_task_board,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
    return tmp_path


def _state(**overrides):
    s = initial_state("Test idea", "proj-x")
    s.update(overrides)
    return s


def _task(id, status="pending", priority=5, depends_on=None):
    t = make_task_item(id=id, title=f"T{id}", description="",
                       priority=priority, depends_on=depends_on or [])
    return {**t, "status": status}


# ── mark_task_done ─────────────────────────────────────────────────────────────

class TestMarkTaskDone:
    def test_sets_task_status_to_done(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_done(state, 1)
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        assert task["status"] == "done"

    def test_updates_updated_at(self):
        t = _task(1)
        orig_ts = t["updated_at"]
        state = _state(tasks=[t])
        updates = mark_task_done(state, 1)
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        # updated_at should change (or at worst be equal if called within same ms)
        assert task["updated_at"] >= orig_ts

    def test_adds_to_completed_task_ids(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_done(state, 1)
        assert 1 in updates["completed_task_ids"]

    def test_appends_history_event(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_done(state, 1)
        events = [e["event"] for e in updates["task_history"]]
        assert "done" in events

    def test_does_not_modify_original_state(self):
        state = _state(tasks=[_task(1)])
        mark_task_done(state, 1)
        assert state["tasks"][0]["status"] == "pending"
        assert state["completed_task_ids"] == []

    def test_preserves_other_tasks(self):
        state = _state(tasks=[_task(1), _task(2)])
        updates = mark_task_done(state, 1)
        t2 = next(t for t in updates["tasks"] if t["id"] == 2)
        assert t2["status"] == "pending"


# ── mark_task_failed ──────────────────────────────────────────────────────────

class TestMarkTaskFailed:
    def test_sets_task_status_to_failed(self):
        state = _state(tasks=[_task(1, status="in_progress")])
        updates = mark_task_failed(state, 1, "NameError")
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        assert task["status"] == "failed"

    def test_sets_error_context(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_failed(state, 1, "timeout after 30s")
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        assert "timeout" in task["error_context"]

    def test_adds_to_failed_task_ids(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_failed(state, 1)
        assert 1 in updates["failed_task_ids"]

    def test_appends_history_event(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_failed(state, 1, "crash")
        events = [e["event"] for e in updates["task_history"]]
        assert "failed" in events

    def test_history_detail_truncated(self):
        long_error = "x" * 500
        state = _state(tasks=[_task(1)])
        updates = mark_task_failed(state, 1, long_error)
        detail = updates["task_history"][-1]["detail"]
        assert len(detail) <= 200

    def test_empty_error_context_default(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_failed(state, 1)
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        assert task["error_context"] == ""


# ── mark_task_blocked ─────────────────────────────────────────────────────────

class TestMarkTaskBlocked:
    def test_sets_task_status_to_blocked(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_blocked(state, 1)
        task = next(t for t in updates["tasks"] if t["id"] == 1)
        assert task["status"] == "blocked"

    def test_adds_to_blocked_task_ids(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_blocked(state, 1)
        assert 1 in updates["blocked_task_ids"]

    def test_appends_history_event(self):
        state = _state(tasks=[_task(1)])
        updates = mark_task_blocked(state, 1, "dep 3 failed")
        events = [e["event"] for e in updates["task_history"]]
        assert "blocked" in events


# ── get_next_runnable_task ────────────────────────────────────────────────────

class TestGetNextRunnableTask:
    def test_returns_unblocked_pending(self):
        tasks = [_task(1), _task(2, depends_on=[1])]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 1

    def test_respects_priority(self):
        tasks = [_task(1, priority=5), _task(2, priority=1)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2

    def test_returns_none_when_all_done(self):
        tasks = [_task(1, status="done")]
        assert get_next_runnable_task(tasks) is None

    def test_returns_none_on_circular_deps(self):
        tasks = [_task(1, depends_on=[2]), _task(2, depends_on=[1])]
        assert get_next_runnable_task(tasks) is None

    def test_skips_blocked_status(self):
        tasks = [_task(1, status="blocked"), _task(2)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2


# ── project_is_complete / project_is_blocked ──────────────────────────────────

class TestProjectPredicates:
    def test_complete_true(self):
        tasks = [_task(1, status="done"), _task(2, status="done")]
        assert project_is_complete(tasks) is True

    def test_complete_false_with_pending(self):
        tasks = [_task(1, status="done"), _task(2)]
        assert project_is_complete(tasks) is False

    def test_complete_false_empty(self):
        assert project_is_complete([]) is False

    def test_blocked_true(self):
        tasks = [
            _task(1, status="done"),
            _task(2, depends_on=[3]),  # dep 3 missing
        ]
        assert project_is_blocked(tasks) is True

    def test_blocked_false_when_runnable(self):
        tasks = [_task(1), _task(2, depends_on=[1])]
        assert project_is_blocked(tasks) is False

    def test_blocked_false_when_all_done(self):
        tasks = [_task(1, status="done")]
        assert project_is_blocked(tasks) is False


# ── persist_task_board ────────────────────────────────────────────────────────

class TestPersistTaskBoard:
    def test_creates_file(self, tmp_workspace):
        state = _state()
        persist_task_board(state)
        assert (tmp_workspace / "proj-x" / "task_board.json").exists()

    def test_file_is_valid_json(self, tmp_workspace):
        state = _state(tasks=[_task(1)], completed_task_ids=[])
        persist_task_board(state)
        content = (tmp_workspace / "proj-x" / "task_board.json").read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_file_contains_project_id(self, tmp_workspace):
        state = _state()
        persist_task_board(state)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert data["project_id"] == "proj-x"

    def test_file_contains_tasks(self, tmp_workspace):
        state = _state(tasks=[_task(1), _task(2)])
        persist_task_board(state)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert len(data["tasks"]) == 2

    def test_file_contains_project_status(self, tmp_workspace):
        state = _state(project_status="active")
        persist_task_board(state)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert data["project_status"] == "active"

    def test_file_contains_persisted_at(self, tmp_workspace):
        state = _state()
        persist_task_board(state)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert "persisted_at" in data

    def test_file_contains_history(self, tmp_workspace):
        from ai_cto.state import task_history_event
        history = [task_history_event(1, "started")]
        state = _state(task_history=history)
        persist_task_board(state)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert len(data["task_history"]) == 1
        assert data["task_history"][0]["event"] == "started"

    def test_creates_parent_dirs(self, tmp_workspace):
        """Project dir should be created if it doesn't exist yet."""
        state = _state()
        board_path = persist_task_board(state)
        assert board_path.parent.is_dir()

    def test_overwrite_on_repeated_calls(self, tmp_workspace):
        """Second call with updated state should overwrite the file."""
        state1 = _state(tasks=[_task(1)])
        state2 = _state(tasks=[_task(1), _task(2)])
        persist_task_board(state1)
        persist_task_board(state2)
        data = json.loads((tmp_workspace / "proj-x" / "task_board.json").read_text())
        assert len(data["tasks"]) == 2
