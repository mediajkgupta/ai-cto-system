"""Tests for state.py — ProjectState initialisation and shape."""

from ai_cto.state import initial_state, ProjectState


def test_initial_state_defaults():
    state = initial_state("Build a todo API", "todo-api")

    assert state["idea"] == "Build a todo API"
    assert state["project_id"] == "todo-api"
    assert state["status"] == "planning"
    assert state["tasks"] == []
    assert state["generated_files"] == {}
    assert state["debug_attempts"] == 0
    assert state["max_debug_attempts"] == 3
    assert state["execution_result"] is None
    assert state["error_context"] == ""
    assert state["architecture"] == ""
    assert state["final_output"] == ""


def test_initial_state_is_dict():
    state = initial_state("An idea", "my-project")
    # ProjectState is a TypedDict — must be subscriptable
    assert isinstance(state, dict)
    assert state["project_id"] == "my-project"


def test_initial_state_current_task_index():
    state = initial_state("Some idea", "proj")
    assert state["current_task_index"] == 0
