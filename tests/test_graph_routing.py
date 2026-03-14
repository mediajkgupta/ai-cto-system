"""Tests for graph.py — controller routing logic (V2)."""

import pytest
from ai_cto.state import initial_state, make_task_item
from ai_cto.graph import (
    route_after_verification,
    route_after_execution,
    route_after_debug,
    route_after_advance,
    advance_task_node,
)


def _make_task(id, title="T", status="pending", priority=5, depends_on=None):
    t = make_task_item(id=id, title=title, description="", priority=priority,
                       depends_on=depends_on or [])
    return {**t, "status": status}


def _make_state(**overrides):
    state = initial_state("Build a todo API", "todo-api")
    state.update(overrides)
    return state


# ── route_after_verification ──────────────────────────────────────────────────

def test_route_after_verification_passes():
    state = _make_state(status="executing")
    assert route_after_verification(state) == "execution"


def test_route_after_verification_fails():
    state = _make_state(status="verifying_failed")
    assert route_after_verification(state) == "debug"


# ── route_after_execution ──────────────────────────────────────────────────────

def test_route_after_execution_success():
    state = _make_state(status="done")
    assert route_after_execution(state) == "advance_task"


def test_route_after_execution_failure():
    state = _make_state(status="debugging")
    assert route_after_execution(state) == "debug"


# ── route_after_debug ──────────────────────────────────────────────────────────

def test_route_after_debug_retry():
    state = _make_state(status="coding")
    assert route_after_debug(state) == "coding"


def test_route_after_debug_failed():
    state = _make_state(status="failed")
    assert route_after_debug(state) == "end"


# ── route_after_advance ────────────────────────────────────────────────────────

def test_route_after_advance_memory_save():
    state = _make_state(status="saving_memory")
    assert route_after_advance(state) == "memory_save"


def test_route_after_advance_coding():
    state = _make_state(status="coding")
    assert route_after_advance(state) == "coding"


def test_route_after_advance_blocked():
    """Blocked project must route to END."""
    state = _make_state(status="blocked")
    assert route_after_advance(state) == "end"


def test_route_after_advance_failed():
    """Failed project must route to END."""
    state = _make_state(status="failed")
    assert route_after_advance(state) == "end"


# ── advance_task_node ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path, monkeypatch):
    """Redirect workspace so task_board.json writes go to a temp dir."""
    monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)


def test_advance_task_node_continues_to_next_task():
    tasks = [
        _make_task(1, "T1", status="in_progress"),
        _make_task(2, "T2", status="pending"),
    ]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)

    assert result["tasks"][0]["status"] == "done"
    assert result["active_task_id"] == 2
    assert result["current_task_index"] == 1
    assert result["status"] == "coding"
    assert result["debug_attempts"] == 0


def test_advance_task_node_finishes_when_all_done():
    tasks = [_make_task(1, "T1", status="in_progress")]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)

    assert result["status"] == "saving_memory"
    assert result["project_status"] == "complete"
    assert result["final_output"] != ""


def test_advance_task_node_populates_completed_task_ids():
    tasks = [_make_task(1, "T1", status="in_progress")]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)
    assert 1 in result["completed_task_ids"]


def test_advance_task_node_appends_task_history():
    tasks = [
        _make_task(1, "T1", status="in_progress"),
        _make_task(2, "T2", status="pending"),
    ]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)
    events = [e["event"] for e in result["task_history"]]
    assert "done" in events
    assert "started" in events


def test_advance_task_node_respects_dependencies():
    """T2 depends on T3; T3 is still pending → T3 must be next, not T2."""
    tasks = [
        _make_task(1, "T1", status="in_progress"),
        _make_task(2, "T2", status="pending", depends_on=[3]),
        _make_task(3, "T3", status="pending"),
    ]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)

    assert result["active_task_id"] == 3
    assert result["status"] == "coding"


def test_advance_task_node_blocked_project():
    """All remaining tasks have unmet deps → project halts as blocked."""
    tasks = [
        _make_task(1, "T1", status="in_progress"),
        _make_task(2, "T2", status="pending", depends_on=[3]),
        # T3 does not exist in the list → dep can never be satisfied
    ]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)

    assert result["status"] == "blocked"
    assert result["project_status"] == "blocked"
    assert route_after_advance(result) == "end"


def test_advance_task_node_priority_ordering():
    """Lower priority number selected first among eligible unblocked tasks."""
    tasks = [
        _make_task(1, "T1", status="in_progress"),
        _make_task(2, "T2", status="pending", priority=3),
        _make_task(3, "T3", status="pending", priority=1),
    ]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    result = advance_task_node(state)
    assert result["active_task_id"] == 3  # priority 1 beats priority 3


def test_advance_task_node_writes_task_board(tmp_path):
    """task_board.json must be created after advancing."""
    tasks = [_make_task(1, "T1", status="in_progress")]
    state = _make_state(tasks=tasks, active_task_id=1, current_task_index=0)
    advance_task_node(state)
    assert (tmp_path / "todo-api" / "task_board.json").exists()
