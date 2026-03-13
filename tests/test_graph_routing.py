"""Tests for graph.py — controller routing logic."""

from ai_cto.state import initial_state
from ai_cto.graph import (
    route_after_verification,
    route_after_execution,
    route_after_debug,
    route_after_advance,
    advance_task_node,
)


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


# ── advance_task_node ──────────────────────────────────────────────────────────

def test_advance_task_node_continues_to_next_task():
    tasks = [
        {"id": 1, "title": "T1", "description": "", "status": "in_progress"},
        {"id": 2, "title": "T2", "description": "", "status": "pending"},
    ]
    state = _make_state(tasks=tasks, current_task_index=0)
    result = advance_task_node(state)

    assert result["tasks"][0]["status"] == "done"
    assert result["current_task_index"] == 1
    assert result["status"] == "coding"
    assert result["debug_attempts"] == 0


def test_advance_task_node_finishes_when_all_done():
    tasks = [{"id": 1, "title": "T1", "description": "", "status": "in_progress"}]
    state = _make_state(tasks=tasks, current_task_index=0)
    result = advance_task_node(state)

    # When all tasks are done, advance_task_node signals memory_save via "saving_memory"
    assert result["status"] == "saving_memory"
    assert "final_output" in result
    assert result["final_output"] != ""


# ── route_after_advance ────────────────────────────────────────────────────────

def test_route_after_advance_end():
    # "saving_memory" is the signal that all tasks are done — routes to memory_save
    state = _make_state(status="saving_memory")
    assert route_after_advance(state) == "memory_save"


def test_route_after_advance_coding():
    state = _make_state(status="coding")
    assert route_after_advance(state) == "coding"
