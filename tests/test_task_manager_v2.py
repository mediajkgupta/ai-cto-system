"""
test_task_manager_v2.py — Task Manager V2 integration tests.

These tests validate the full V2 task orchestration layer:
  - Multi-task pipeline with dependency routing
  - Per-task retry tracking (task.retries vs project-level debug_attempts)
  - Blocked-project detection and clean summary
  - Task board JSON content and updates
  - task_history log integrity
  - MOCK_LLM mode compatibility
  - Backward compatibility: debug_attempts, next_unblocked_task alias, etc.

All tests run in MOCK_LLM mode (no API calls).
"""

import json
import pytest
from pathlib import Path

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
from ai_cto.tools.task_board import (
    mark_task_done,
    mark_task_failed,
    mark_task_blocked,
    persist_task_board,
)
from ai_cto.graph import advance_task_node, route_after_advance


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    monkeypatch.setenv("AI_CTO_MOCK_LLM", "true")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture()
def tmp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
    return tmp_path


def _state(**overrides):
    s = initial_state("Build a CRM app", "crm-proj")
    s.update(overrides)
    return s


def _task(id, status="pending", priority=5, depends_on=None,
          retries=0, max_retries=3):
    t = make_task_item(id=id, title=f"Task {id}", description=f"Desc {id}",
                       priority=priority, depends_on=depends_on or [],
                       max_retries=max_retries)
    return {**t, "status": status, "retries": retries}


# ── Task schema V2 fields ─────────────────────────────────────────────────────

class TestTaskSchemaV2:
    def test_task_has_retries_field(self):
        t = make_task_item(id=1, title="T", description="D")
        assert "retries" in t
        assert t["retries"] == 0

    def test_task_has_max_retries_field(self):
        t = make_task_item(id=1, title="T", description="D")
        assert "max_retries" in t
        assert t["max_retries"] == 3

    def test_task_has_timestamps(self):
        t = make_task_item(id=1, title="T", description="D")
        assert "created_at" in t
        assert "updated_at" in t
        assert len(t["created_at"]) > 10
        assert len(t["updated_at"]) > 10

    def test_task_custom_max_retries(self):
        t = make_task_item(id=1, title="T", description="D", max_retries=5)
        assert t["max_retries"] == 5

    def test_task_status_supports_blocked(self):
        t = make_task_item(id=1, title="T", description="D")
        t2 = {**t, "status": "blocked"}
        assert t2["status"] == "blocked"


# ── Initial state V2 fields ───────────────────────────────────────────────────

class TestInitialStateV2:
    def test_completed_task_ids_empty(self):
        s = initial_state("idea", "proj")
        assert s["completed_task_ids"] == []

    def test_failed_task_ids_empty(self):
        s = initial_state("idea", "proj")
        assert s["failed_task_ids"] == []

    def test_blocked_task_ids_empty(self):
        s = initial_state("idea", "proj")
        assert s["blocked_task_ids"] == []

    def test_task_history_empty(self):
        s = initial_state("idea", "proj")
        assert s["task_history"] == []

    def test_project_status_active(self):
        s = initial_state("idea", "proj")
        assert s["project_status"] == "active"


# ── Per-task retry logic ──────────────────────────────────────────────────────

class TestPerTaskRetries:
    def test_debug_node_increments_task_retries(self, tmp_workspace):
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=0, max_retries=3)]
        state = _state(
            tasks=tasks, active_task_id=1, current_task_index=0,
            debug_attempts=0, max_debug_attempts=3,
            error_context="some error", generated_files={},
        )
        result = debug_node(state)
        t = next(t for t in result["tasks"] if t["id"] == 1)
        assert t["retries"] == 1

    def test_debug_attempts_also_incremented(self, tmp_workspace):
        """Backward compat: state.debug_attempts still increments."""
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=0, max_retries=3)]
        state = _state(
            tasks=tasks, active_task_id=1, current_task_index=0,
            debug_attempts=0, max_debug_attempts=3,
            error_context="err", generated_files={},
        )
        result = debug_node(state)
        assert result["debug_attempts"] == 1

    def test_task_fails_after_per_task_max(self, tmp_workspace):
        """Task fails when task.retries reaches task.max_retries."""
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(
            tasks=tasks, active_task_id=1, current_task_index=0,
            debug_attempts=3, max_debug_attempts=3,
            error_context="err", generated_files={},
        )
        result = debug_node(state)
        assert result["status"] == "failed"

    def test_different_tasks_have_independent_retries(self, tmp_workspace):
        """T2 retries are tracked separately from T1 retries."""
        tasks = [_task(1, status="done", retries=2), _task(2, status="in_progress", retries=0)]
        state = _state(tasks=tasks, active_task_id=2, current_task_index=1,
                       debug_attempts=0, max_debug_attempts=3,
                       error_context="err", generated_files={})
        from ai_cto.agents.debug import debug_node
        result = debug_node(state)
        t1 = next(t for t in result["tasks"] if t["id"] == 1)
        t2 = next(t for t in result["tasks"] if t["id"] == 2)
        assert t1["retries"] == 2   # unchanged
        assert t2["retries"] == 1   # incremented

    def test_advance_resets_debug_attempts(self, tmp_workspace):
        """debug_attempts resets on task advance for MemoryAgent compat."""
        tasks = [
            _task(1, status="in_progress"),
            _task(2, status="pending"),
        ]
        state = _state(tasks=tasks, active_task_id=1, current_task_index=0,
                       debug_attempts=2)
        result = advance_task_node(state)
        assert result["debug_attempts"] == 0


# ── Dependency routing ────────────────────────────────────────────────────────

class TestDependencyRouting:
    def test_chain_deps_execute_in_order(self, tmp_workspace):
        """T1 → T2 → T3 must execute strictly in that order."""
        tasks = [
            _task(1, status="in_progress"),
            _task(2, depends_on=[1]),
            _task(3, depends_on=[2]),
        ]
        state = _state(tasks=tasks, active_task_id=1)
        s1 = advance_task_node(state)
        assert s1["active_task_id"] == 2

        s1["tasks"] = update_task_in_list(s1["tasks"], 2, status="in_progress")
        s2 = advance_task_node({**s1, "active_task_id": 2})
        assert s2["active_task_id"] == 3

    def test_parallel_tasks_selected_by_priority(self, tmp_workspace):
        """Two tasks with no deps: lower priority number wins."""
        tasks = [
            _task(1, status="in_progress"),
            _task(2, priority=4),
            _task(3, priority=2),
        ]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert result["active_task_id"] == 3

    def test_project_blocked_when_dep_failed(self, tmp_workspace):
        """If required dep failed, dependent task can never run → blocked."""
        tasks = [
            _task(1, status="in_progress"),
            _task(2, depends_on=[3]),   # T3 is absent/failed
        ]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert result["status"] == "blocked"
        assert result["project_status"] == "blocked"

    def test_blocked_summary_mentions_pending_tasks(self, tmp_workspace):
        tasks = [
            _task(1, status="in_progress"),
            _task(2, depends_on=[99]),
        ]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert "blocked" in result["final_output"].lower()
        assert "Task 2" in result["final_output"]


# ── Task history log ──────────────────────────────────────────────────────────

class TestTaskHistory:
    def test_history_grows_on_advance(self, tmp_workspace):
        tasks = [_task(1, status="in_progress"), _task(2)]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert len(result["task_history"]) >= 2  # "done" + "started"

    def test_history_contains_done_event(self, tmp_workspace):
        tasks = [_task(1, status="in_progress"), _task(2)]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert any(e["event"] == "done" and e["task_id"] == 1
                   for e in result["task_history"])

    def test_history_contains_started_event(self, tmp_workspace):
        tasks = [_task(1, status="in_progress"), _task(2)]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert any(e["event"] == "started" and e["task_id"] == 2
                   for e in result["task_history"])

    def test_history_contains_retry_event(self, tmp_workspace):
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=0, max_retries=3)]
        state = _state(
            tasks=tasks, active_task_id=1, current_task_index=0,
            debug_attempts=0, max_debug_attempts=3,
            error_context="err", generated_files={},
        )
        result = debug_node(state)
        events = [e["event"] for e in result.get("task_history", [])]
        assert "retry" in events

    def test_history_event_has_timestamp(self):
        evt = task_history_event(1, "done")
        assert "timestamp" in evt
        assert "T" in evt["timestamp"]  # ISO format check

    def test_history_accumulates_across_tasks(self, tmp_workspace):
        """History from prior tasks is not lost on advance."""
        prior_history = [task_history_event(0, "started")]
        tasks = [_task(1, status="in_progress"), _task(2)]
        state = _state(tasks=tasks, active_task_id=1,
                       task_history=prior_history)
        result = advance_task_node(state)
        assert len(result["task_history"]) >= 3  # prior + done + started


# ── Task board persistence ────────────────────────────────────────────────────

class TestTaskBoardPersistence:
    def test_board_written_by_task_manager(self, tmp_workspace):
        from ai_cto.agents.task_manager import task_manager_node
        state = initial_state("Build an app", "board-test")
        state = task_manager_node(state)
        assert (tmp_workspace / "board-test" / "task_board.json").exists()

    def test_board_updated_on_advance(self, tmp_workspace):
        tasks = [_task(1, status="in_progress"), _task(2)]
        state = _state(tasks=tasks, active_task_id=1)
        advance_task_node(state)
        board_path = tmp_workspace / "crm-proj" / "task_board.json"
        assert board_path.exists()
        data = json.loads(board_path.read_text())
        assert 1 in data["completed_task_ids"]

    def test_board_reflects_project_status(self, tmp_workspace):
        tasks = [_task(1, status="in_progress")]
        state = _state(tasks=tasks, active_task_id=1)
        advance_task_node(state)
        data = json.loads(
            (tmp_workspace / "crm-proj" / "task_board.json").read_text()
        )
        assert data["project_status"] == "complete"

    def test_board_contains_all_required_keys(self, tmp_workspace):
        state = _state(tasks=[_task(1)])
        persist_task_board(state)
        data = json.loads(
            (tmp_workspace / "crm-proj" / "task_board.json").read_text()
        )
        for key in ("project_id", "project_goal", "project_status",
                    "active_task_id", "tasks",
                    "completed_task_ids", "failed_task_ids", "blocked_task_ids",
                    "task_history", "persisted_at"):
            assert key in data, f"Missing key: {key}"

    def test_board_updated_on_debug_failure(self, tmp_workspace):
        from ai_cto.agents.debug import debug_node
        tasks = [_task(1, status="in_progress", retries=3, max_retries=3)]
        state = _state(tasks=tasks, active_task_id=1, current_task_index=0,
                       debug_attempts=3, error_context="err", generated_files={})
        debug_node(state)
        data = json.loads(
            (tmp_workspace / "crm-proj" / "task_board.json").read_text()
        )
        assert 1 in data["failed_task_ids"]


# ── Full pipeline in mock mode ────────────────────────────────────────────────

class TestMockPipelineV2:
    def _run(self, tmp_workspace, idea="Build a REST API", project_id="v2-test"):
        from ai_cto.state import initial_state
        from ai_cto.graph import build_graph
        state = initial_state(idea=idea, project_id=project_id)
        graph = build_graph()
        final = None
        for event in graph.stream(state):
            node_name = list(event.keys())[0]
            final = event[node_name]
        return final

    def test_pipeline_status_done(self, tmp_workspace):
        final = self._run(tmp_workspace)
        assert final["status"] == "done"

    def test_pipeline_project_status_complete(self, tmp_workspace):
        final = self._run(tmp_workspace)
        assert final["project_status"] == "complete"

    def test_pipeline_completed_task_ids_populated(self, tmp_workspace):
        final = self._run(tmp_workspace)
        assert len(final["completed_task_ids"]) >= 1

    def test_pipeline_task_history_populated(self, tmp_workspace):
        final = self._run(tmp_workspace)
        assert len(final["task_history"]) >= 2  # started + done at minimum

    def test_pipeline_task_board_on_disk(self, tmp_workspace):
        self._run(tmp_workspace, project_id="disk-test")
        assert (tmp_workspace / "disk-test" / "task_board.json").exists()

    def test_pipeline_board_shows_complete(self, tmp_workspace):
        self._run(tmp_workspace, project_id="status-test")
        data = json.loads(
            (tmp_workspace / "status-test" / "task_board.json").read_text()
        )
        assert data["project_status"] == "complete"

    def test_pipeline_no_failed_tasks(self, tmp_workspace):
        final = self._run(tmp_workspace)
        assert final["failed_task_ids"] == []
        assert final["blocked_task_ids"] == []

    def test_pipeline_zero_retries_on_clean_code(self, tmp_workspace):
        final = self._run(tmp_workspace)
        for task in final["tasks"]:
            assert task["retries"] == 0

    def test_two_runs_dont_interfere(self, tmp_workspace):
        final_a = self._run(tmp_workspace, project_id="run-a")
        final_b = self._run(tmp_workspace, project_id="run-b")
        assert final_a["status"] == "done"
        assert final_b["status"] == "done"
        assert (tmp_workspace / "run-a" / "task_board.json").exists()
        assert (tmp_workspace / "run-b" / "task_board.json").exists()


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompat:
    def test_next_unblocked_task_alias_works(self):
        """V1 callers using next_unblocked_task still work."""
        tasks = [_task(1), _task(2, depends_on=[1])]
        result = next_unblocked_task(tasks)
        assert result["id"] == 1

    def test_debug_attempts_field_still_in_state(self):
        s = initial_state("x", "y")
        assert "debug_attempts" in s
        assert "max_debug_attempts" in s

    def test_current_task_index_kept_in_sync(self, tmp_workspace):
        tasks = [_task(1, status="in_progress"), _task(2), _task(3)]
        state = _state(tasks=tasks, active_task_id=1, current_task_index=0)
        result = advance_task_node(state)
        # current_task_index should point to the same task as active_task_id
        active_idx = result["current_task_index"]
        active_id = result["active_task_id"]
        assert result["tasks"][active_idx]["id"] == active_id

    def test_final_output_still_populated_on_success(self, tmp_workspace):
        tasks = [_task(1, status="in_progress")]
        state = _state(tasks=tasks, active_task_id=1)
        result = advance_task_node(state)
        assert result["final_output"] != ""
        assert "crm-proj" in result["final_output"]
