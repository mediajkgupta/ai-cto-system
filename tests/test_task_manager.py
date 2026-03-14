"""
test_task_manager.py — Unit tests for the Task Manager Agent and state helpers.

Tests cover:
  - make_task_item factory (including V2 fields)
  - get_active_task helper
  - update_task_in_list helper
  - get_next_runnable_task / next_unblocked_task scheduling (deps, priority)
  - project_is_complete / project_is_blocked predicates
  - task_history_event helper
  - mock_task_manager_node (pipeline integration)
  - advance_task_node with dependency-aware routing
  - debug_node marking active task as failed on per-task max retries (V2)
"""

import pytest
from ai_cto.state import (
    initial_state,
    make_task_item,
    get_active_task,
    update_task_in_list,
    get_next_runnable_task,
    next_unblocked_task,
    project_is_complete,
    project_is_blocked,
    task_history_event,
    now_iso,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    monkeypatch.setenv("AI_CTO_MOCK_LLM", "true")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _state(**overrides):
    s = initial_state("Test idea", "test-proj")
    s.update(overrides)
    return s


def _task(id, status="pending", priority=5, depends_on=None,
          retries=0, max_retries=3):
    t = make_task_item(id=id, title=f"Task {id}", description="desc",
                       priority=priority, depends_on=depends_on or [],
                       max_retries=max_retries)
    return {**t, "status": status, "retries": retries}


# ── make_task_item ─────────────────────────────────────────────────────────────

class TestMakeTaskItem:
    def test_required_fields(self):
        t = make_task_item(id=1, title="T", description="D")
        assert t["id"] == 1
        assert t["title"] == "T"
        assert t["description"] == "D"
        assert t["status"] == "pending"

    def test_default_optional_fields(self):
        t = make_task_item(id=1, title="T", description="D")
        assert t["priority"] == 5
        assert t["depends_on"] == []
        assert t["outputs"] == []
        assert t["error_context"] == ""

    def test_default_v2_fields(self):
        t = make_task_item(id=1, title="T", description="D")
        assert t["retries"] == 0
        assert t["max_retries"] == 3
        assert isinstance(t["created_at"], str) and len(t["created_at"]) > 10
        assert isinstance(t["updated_at"], str) and len(t["updated_at"]) > 10

    def test_explicit_optional_fields(self):
        t = make_task_item(id=2, title="T", description="D",
                           priority=1, depends_on=[1], outputs=["main.py"],
                           max_retries=5)
        assert t["priority"] == 1
        assert t["depends_on"] == [1]
        assert t["outputs"] == ["main.py"]
        assert t["max_retries"] == 5

    def test_timestamps_are_iso_strings(self):
        t = make_task_item(id=1, title="T", description="D")
        # ISO timestamps contain "T" and "+"  or "Z" or timezone info
        assert "T" in t["created_at"]
        assert "T" in t["updated_at"]


# ── get_active_task ────────────────────────────────────────────────────────────

class TestGetActiveTask:
    def test_returns_correct_task(self):
        tasks = [_task(1), _task(2)]
        state = _state(tasks=tasks, active_task_id=2)
        result = get_active_task(state)
        assert result is not None
        assert result["id"] == 2

    def test_returns_none_when_active_task_id_is_none(self):
        state = _state(tasks=[_task(1)], active_task_id=None)
        assert get_active_task(state) is None

    def test_returns_none_when_id_not_in_list(self):
        state = _state(tasks=[_task(1)], active_task_id=99)
        assert get_active_task(state) is None


# ── update_task_in_list ────────────────────────────────────────────────────────

class TestUpdateTaskInList:
    def test_updates_matching_task(self):
        tasks = [_task(1), _task(2)]
        result = update_task_in_list(tasks, 1, status="done")
        assert result[0]["status"] == "done"
        assert result[1]["status"] == "pending"

    def test_does_not_mutate_original(self):
        tasks = [_task(1)]
        update_task_in_list(tasks, 1, status="done")
        assert tasks[0]["status"] == "pending"

    def test_supports_multiple_field_updates(self):
        tasks = [_task(1)]
        result = update_task_in_list(tasks, 1, status="failed", error_context="boom")
        assert result[0]["status"] == "failed"
        assert result[0]["error_context"] == "boom"


# ── get_next_runnable_task (V2 canonical) and next_unblocked_task (V1 alias) ───

class TestGetNextRunnableTask:
    def test_both_names_work(self):
        tasks = [_task(1)]
        assert get_next_runnable_task(tasks) is not None
        assert next_unblocked_task(tasks) is not None

    def test_returns_first_pending_no_deps(self):
        tasks = [_task(1), _task(2)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 1

    def test_skips_blocked_task_status(self):
        tasks = [_task(1, status="blocked"), _task(2)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2

    def test_skips_unmet_deps(self):
        tasks = [
            _task(1, depends_on=[2]),  # dep 2 not done
            _task(2),
        ]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2

    def test_unblocks_when_dep_is_done(self):
        tasks = [
            _task(1, depends_on=[2]),
            _task(2, status="done"),
        ]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 1

    def test_returns_none_when_all_blocked(self):
        tasks = [
            _task(1, depends_on=[2]),
            _task(2, depends_on=[1]),  # circular
        ]
        assert get_next_runnable_task(tasks) is None

    def test_returns_none_when_all_done(self):
        tasks = [_task(1, status="done"), _task(2, status="done")]
        assert get_next_runnable_task(tasks) is None

    def test_priority_ordering(self):
        tasks = [
            _task(1, priority=3),
            _task(2, priority=1),
            _task(3, priority=5),
        ]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2

    def test_skips_failed_tasks(self):
        tasks = [_task(1, status="failed"), _task(2)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2

    def test_skips_in_progress_tasks(self):
        tasks = [_task(1, status="in_progress"), _task(2)]
        result = get_next_runnable_task(tasks)
        assert result["id"] == 2


# ── project_is_complete / project_is_blocked ──────────────────────────────────

class TestProjectPredicates:
    def test_complete_when_all_done(self):
        tasks = [_task(1, status="done"), _task(2, status="done")]
        assert project_is_complete(tasks) is True

    def test_not_complete_with_pending(self):
        tasks = [_task(1, status="done"), _task(2, status="pending")]
        assert project_is_complete(tasks) is False

    def test_not_complete_empty_list(self):
        assert project_is_complete([]) is False

    def test_blocked_when_pending_with_unmet_deps(self):
        tasks = [
            _task(1, status="done"),
            _task(2, depends_on=[3]),  # dep 3 doesn't exist
        ]
        assert project_is_blocked(tasks) is True

    def test_not_blocked_when_no_pending(self):
        tasks = [_task(1, status="done")]
        assert project_is_blocked(tasks) is False

    def test_not_blocked_when_runnable_task_exists(self):
        tasks = [_task(1), _task(2, depends_on=[1])]
        assert project_is_blocked(tasks) is False


# ── task_history_event ────────────────────────────────────────────────────────

class TestTaskHistoryEvent:
    def test_shape(self):
        evt = task_history_event(1, "done")
        assert evt["task_id"] == 1
        assert evt["event"] == "done"
        assert "timestamp" in evt
        assert "detail" in evt

    def test_detail_defaults_empty(self):
        evt = task_history_event(1, "started")
        assert evt["detail"] == ""

    def test_custom_detail(self):
        evt = task_history_event(2, "retry", detail="NameError on line 5")
        assert evt["detail"] == "NameError on line 5"


# ── mock_task_manager_node ─────────────────────────────────────────────────────

class TestMockTaskManagerNode:
    def test_produces_tasks_with_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(architecture="## Arch\nsome content")
        result = task_manager_node(state)
        assert len(result["tasks"]) >= 1
        for t in result["tasks"]:
            for field in ("id", "title", "description", "status", "priority",
                          "depends_on", "outputs", "error_context",
                          "retries", "max_retries", "created_at", "updated_at"):
                assert field in t, f"Missing field: {field}"

    def test_sets_active_task_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(architecture="## Arch")
        result = task_manager_node(state)
        assert result["active_task_id"] is not None
        ids = [t["id"] for t in result["tasks"]]
        assert result["active_task_id"] in ids

    def test_sets_status_coding(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(architecture="## Arch")
        result = task_manager_node(state)
        assert result["status"] == "coding"

    def test_persists_task_board(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(architecture="## Arch")
        task_manager_node(state)
        board_path = tmp_path / "test-proj" / "task_board.json"
        assert board_path.exists()

    def test_task_history_started_event(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(architecture="## Arch")
        result = task_manager_node(state)
        events = [e["event"] for e in result.get("task_history", [])]
        assert "started" in events


# ── debug_node marks active task failed on per-task max retries (V2) ──────────

class TestDebugNodeFailure:
    def test_marks_active_task_failed_on_max_retries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.debug import debug_node
        # V2: task already at max_retries (retries=3, max_retries=3)
        # next attempt (4th) will exceed → failure
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
            debug_attempts=3,
            max_debug_attempts=3,
            error_context="some error",
            generated_files={},
        )
        result = debug_node(state)
        assert result["status"] == "failed"
        failed_task = next(t for t in result["tasks"] if t["id"] == 1)
        assert failed_task["status"] == "failed"
        assert "some error" in failed_task["error_context"]

    def test_failure_message_in_final_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
            debug_attempts=3,
            max_debug_attempts=3,
            error_context="fatal error",
            generated_files={},
        )
        result = debug_node(state)
        assert "failed" in result["final_output"].lower()

    def test_failed_task_in_failed_task_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
            debug_attempts=3,
            max_debug_attempts=3,
            error_context="fatal",
            generated_files={},
        )
        result = debug_node(state)
        assert 1 in result["failed_task_ids"]

    def test_task_board_persisted_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
            debug_attempts=3,
            max_debug_attempts=3,
            error_context="fatal",
            generated_files={},
        )
        debug_node(state)
        assert (tmp_path / "test-proj" / "task_board.json").exists()
