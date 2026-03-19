"""
mock_llm.py — Mock LLM responses for pipeline testing without API credits.

Activation (either condition is sufficient):
    - ANTHROPIC_API_KEY is not set (or empty)
    - AI_CTO_MOCK_LLM=true  (explicit override, useful even with a key)

All mock responses satisfy the same JSON contracts the real agents use,
so the full pipeline runs identically:

    memory_retrieve → planning → coding → verification → execution
                                                ↓ errors
                                              debug → coding (retry)
                        advance_task → memory_save → END

Mock code properties:
    - stdlib only (no external deps, no requirements.txt entry needed)
    - passes all 5 verification checks with zero errors
    - __main__ block exits 0 so ExecutionAgent reports success
    - contains real assertions so bugs in the mock logic would surface
"""

import os
import logging
from ai_cto.state import ProjectState, TaskItem, make_task_item, next_unblocked_task
from ai_cto.tools.file_writer import write_files

logger = logging.getLogger(__name__)


# ── Mode detection ─────────────────────────────────────────────────────────────

def is_mock_mode() -> bool:
    """
    Return True if mock LLM mode is active.

    Checks (in order):
      1. AI_CTO_MOCK_LLM env var is set to a truthy value → mock
      2. AI_CTO_LLM_PROVIDER is set to a non-mock provider → NOT mock
         (ollama/anthropic are real providers even without an API key)
      3. ANTHROPIC_API_KEY is absent or empty → mock (fallback)
    """
    if os.environ.get("AI_CTO_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return True
    explicit = os.environ.get("AI_CTO_LLM_PROVIDER", "").strip().lower()
    if explicit and explicit != "mock":
        # User has explicitly chosen a real provider
        return False
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True
    return False


def log_mock_banner(agent: str) -> None:
    logger.info("[MOCK] %s — returning pre-defined response (no API call).", agent)


# ── Mock data: architecture + tasks ───────────────────────────────────────────

MOCK_ARCHITECTURE = """\
## Todo REST API — Architecture

### Components
- **main.py**: Single-module implementation with in-memory data store.
  Exposes five pure functions: `create_todo`, `get_all_todos`, `get_todo`,
  `update_todo`, `delete_todo`.

### Data Flow
1. Caller invokes a CRUD function with validated parameters.
2. Function reads/mutates the module-level `_todos` list.
3. Function returns a result dict, list, or bool.

### Tech Stack
- Python 3.11+ standard library only — zero external dependencies.
- In-memory list as the data store (resets on process restart).
- Pure functions for easy unit testing.

### Design Decisions
- Single-file for maximum portability.
- Input validation raises `ValueError` on bad input (empty title).
- IDs are auto-incremented integers starting at 1.
"""

MOCK_TASKS: list[TaskItem] = [
    make_task_item(
        id=1,
        title="Implement Todo CRUD operations",
        description=(
            "Create main.py with five functions (create_todo, get_all_todos, "
            "get_todo, update_todo, delete_todo) backed by an in-memory list. "
            "Add a __main__ block that exercises every operation and exits 0."
        ),
        priority=1,
        depends_on=[],
        outputs=["main.py", "requirements.txt"],
    )
]

# Extended 4-task mock for multi-task demonstrations and verification.
# Activated by setting AI_CTO_MOCK_EXTENDED=true.
# All tasks use the same known-good MOCK_MAIN_PY so coding/execution
# succeed on every pass — the point is to exercise the orchestration layer.
MOCK_TASKS_EXTENDED: list[TaskItem] = [
    make_task_item(
        id=1,
        title="Scaffold project structure",
        description=(
            "Create requirements.txt and the entry-point stub main.py "
            "with module docstring and import block."
        ),
        priority=1,
        depends_on=[],
        outputs=["main.py", "requirements.txt"],
    ),
    make_task_item(
        id=2,
        title="Implement in-memory data store",
        description=(
            "Add _todos list and _next_id counter with create_todo / get_all_todos."
        ),
        priority=2,
        depends_on=[1],
        outputs=["main.py"],
    ),
    make_task_item(
        id=3,
        title="Implement CRUD operations",
        description=(
            "Add get_todo, update_todo, delete_todo with full input validation."
        ),
        priority=3,
        depends_on=[2],
        outputs=["main.py"],
    ),
    make_task_item(
        id=4,
        title="Add main entry point and integration test",
        description=(
            "Add __main__ block that exercises all five CRUD functions "
            "and exits 0 on success."
        ),
        priority=4,
        depends_on=[3],
        outputs=["main.py"],
    ),
]

# Five-task mock for resume demonstrations.
# Activated by setting AI_CTO_MOCK_FIVE=true.
# Adds a 5th task (CLI layer) so the pipeline has enough tasks to stop
# partway through and show meaningful resume behaviour.
MOCK_TASKS_FIVE: list[TaskItem] = list(MOCK_TASKS_EXTENDED) + [
    make_task_item(
        id=5,
        title="Add CLI entry point with argparse",
        description=(
            "Wrap the CRUD functions in an argparse CLI with 'add', 'list', "
            "and 'delete' subcommands. Write cli.py and update main.py imports."
        ),
        priority=5,
        depends_on=[4],
        outputs=["cli.py", "main.py"],
    ),
]


# ── Mock data: generated source files ─────────────────────────────────────────

# Deliberately uses only `typing` (stdlib) so DependencyChecker raises no warnings.
# All five verification checkers will pass cleanly on this code.
MOCK_MAIN_PY = '''\
"""
Todo API — in-memory CRUD implementation.
Generated by AI CTO System (MOCK_LLM mode).
"""
from typing import Optional

_todos: list[dict] = []
_next_id: int = 1


def create_todo(title: str, description: str = "") -> dict:
    """Create a new todo item. Raises ValueError if title is blank."""
    global _next_id
    if not title.strip():
        raise ValueError("Title cannot be empty.")
    todo = {
        "id": _next_id,
        "title": title.strip(),
        "description": description,
        "done": False,
    }
    _todos.append(todo)
    _next_id += 1
    return todo


def get_all_todos() -> list[dict]:
    """Return a shallow copy of all todo items."""
    return list(_todos)


def get_todo(todo_id: int) -> Optional[dict]:
    """Return a single todo by ID, or None if not found."""
    return next((t for t in _todos if t["id"] == todo_id), None)


def update_todo(todo_id: int, done: bool) -> Optional[dict]:
    """Mark a todo done or not-done. Returns updated item or None."""
    todo = get_todo(todo_id)
    if todo is not None:
        todo["done"] = done
    return todo


def delete_todo(todo_id: int) -> bool:
    """Delete a todo by ID. Returns True if deleted, False if not found."""
    global _todos
    before = len(_todos)
    _todos = [t for t in _todos if t["id"] != todo_id]
    return len(_todos) < before


if __name__ == "__main__":
    print("=== Todo API Demo ===")

    # Create
    t1 = create_todo("Buy groceries", "Milk, eggs, bread")
    t2 = create_todo("Write documentation")
    t3 = create_todo("Deploy to production", "Use Railway or Vercel")
    assert len(get_all_todos()) == 3, "Expected 3 todos"
    print(f"Created {len(get_all_todos())} todos.")

    # Read
    found = get_todo(t1["id"])
    assert found is not None, "Todo should be found by ID"
    assert found["title"] == "Buy groceries"
    print(f"Found: {found['title']}")

    # Update
    updated = update_todo(t1["id"], done=True)
    assert updated is not None and updated["done"] is True
    print(f"Marked done: {updated['title']}")

    # Delete
    deleted = delete_todo(t2["id"])
    assert deleted is True
    assert len(get_all_todos()) == 2, "Expected 2 todos after deletion"
    print(f"After delete: {len(get_all_todos())} remaining.")

    # Edge case: delete non-existent
    assert delete_todo(9999) is False, "Deleting unknown ID should return False"

    # Edge case: empty title
    try:
        create_todo("   ")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("=== All operations passed ===")
'''

# requirements.txt intentionally empty — stdlib only, DependencyChecker sees no warnings
MOCK_REQUIREMENTS_TXT = "# No external dependencies — stdlib only.\n"

MOCK_GENERATED_FILES: dict[str, str] = {
    "main.py": MOCK_MAIN_PY,
    "requirements.txt": MOCK_REQUIREMENTS_TXT,
}
MOCK_ENTRY_POINT = "main.py"


# ── Mock data: debug patch ─────────────────────────────────────────────────────

# Returned by DebugAgent in mock mode. Uses the known-good main.py regardless
# of what error was encountered, simulating a successful fix.
MOCK_DEBUG_DIAGNOSIS = (
    "Mock debug: the generated code contained an error that was patched. "
    "Replaced with the canonical in-memory Todo implementation "
    "which uses only stdlib, has input validation, and exits cleanly."
)

MOCK_DEBUG_FILES: dict[str, str] = {
    "main.py": MOCK_MAIN_PY,
}


# ── Mock data: generated tests ─────────────────────────────────────────────────

MOCK_TEST_FILE = '''\
"""
Auto-generated tests for Todo API (MOCK_LLM mode).
Run from workspace/<project_id>/ with: pytest tests/
"""
import sys
import os
import importlib
import pytest

# Allow importing main.py from the workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def reset_todos():
    """Reset in-memory state before every test."""
    import main
    main._todos.clear()
    main._next_id = 1
    yield
    main._todos.clear()
    main._next_id = 1


def test_create_todo_returns_dict():
    import main
    todo = main.create_todo("Test task")
    assert isinstance(todo, dict)
    assert todo["title"] == "Test task"
    assert todo["done"] is False
    assert "id" in todo


def test_create_todo_increments_id():
    import main
    t1 = main.create_todo("First")
    t2 = main.create_todo("Second")
    assert t2["id"] == t1["id"] + 1


def test_create_todo_rejects_empty_title():
    import main
    with pytest.raises(ValueError):
        main.create_todo("   ")


def test_get_all_todos_returns_list():
    import main
    main.create_todo("A")
    main.create_todo("B")
    todos = main.get_all_todos()
    assert len(todos) == 2


def test_get_todo_by_id():
    import main
    todo = main.create_todo("Find me")
    found = main.get_todo(todo["id"])
    assert found is not None
    assert found["title"] == "Find me"


def test_get_todo_returns_none_for_missing():
    import main
    assert main.get_todo(9999) is None


def test_update_todo_marks_done():
    import main
    todo = main.create_todo("Mark me")
    result = main.update_todo(todo["id"], done=True)
    assert result is not None
    assert result["done"] is True


def test_update_nonexistent_returns_none():
    import main
    assert main.update_todo(9999, done=True) is None


def test_delete_todo_removes_item():
    import main
    todo = main.create_todo("Delete me")
    assert main.delete_todo(todo["id"]) is True
    assert main.get_todo(todo["id"]) is None


def test_delete_nonexistent_returns_false():
    import main
    assert main.delete_todo(9999) is False
'''

MOCK_GENERATED_TESTS: dict[str, str] = {
    "tests/test_todo.py": MOCK_TEST_FILE,
}


# ── Node-level mock functions (called by each agent) ──────────────────────────

def mock_planning_node(state: ProjectState) -> ProjectState:
    """Return a pre-defined architecture document (tasks handled by TaskManagerAgent)."""
    log_mock_banner("PlanningAgent")
    logger.info("[MOCK] Architecture plan ready.")
    return {
        **state,
        "architecture": MOCK_ARCHITECTURE,
        "status": "task_managing",
    }


def mock_task_manager_node(state: ProjectState) -> ProjectState:
    """Return the pre-defined task list, set the first active task, and persist board.

    Set AI_CTO_MOCK_FIVE=true     to use the 5-task list (resume demo).
    Set AI_CTO_MOCK_EXTENDED=true to use the 4-task list (multi-task demo).
    Default: 1-task list.
    """
    from ai_cto.state import task_history_event
    from ai_cto.tools.task_board import persist_task_board
    log_mock_banner("TaskManagerAgent")
    if os.environ.get("AI_CTO_MOCK_FIVE", "").lower() in ("1", "true", "yes"):
        tasks = list(MOCK_TASKS_FIVE)
        logger.info("[MOCK] Using FIVE-task list.")
    elif os.environ.get("AI_CTO_MOCK_EXTENDED", "").lower() in ("1", "true", "yes"):
        tasks = list(MOCK_TASKS_EXTENDED)
        logger.info("[MOCK] Using EXTENDED 4-task list.")
    else:
        tasks = list(MOCK_TASKS)
    first_task = next_unblocked_task(tasks)
    active_id = first_task["id"] if first_task else None
    first_idx = next(
        (i for i, t in enumerate(tasks) if t["id"] == active_id), 0
    ) if active_id is not None else 0
    history = list(state.get("task_history", []))
    if active_id is not None:
        history.append(task_history_event(active_id, "started"))
    logger.info("[MOCK] %d task(s) created. First active task id=%s.", len(tasks), active_id)
    new_state = {
        **state,
        "tasks": tasks,
        "active_task_id": active_id,
        "current_task_index": first_idx,
        "task_history": history,
        "status": "coding",
    }
    persist_task_board(new_state)
    return new_state


def mock_coding_node(state: ProjectState) -> ProjectState:
    """Return the pre-defined todo app source files and write them to disk.

    Mirrors the Fix-2 accumulation pattern in the real coding_node:
    current task files are merged onto existing generated_files so prior
    task outputs are never silently dropped from state.
    """
    from ai_cto.state import get_active_task, update_task_in_list
    task = get_active_task(state) or state["tasks"][state["current_task_index"]]
    log_mock_banner("CodingAgent")
    logger.info("[MOCK] Returning mock files for task: %s", task["title"])

    write_files(project_id=state["project_id"], files=MOCK_GENERATED_FILES)

    # Accumulate: merge task files onto any previously generated files
    accumulated = {**state.get("generated_files", {}), **MOCK_GENERATED_FILES}
    updated_tasks = update_task_in_list(state["tasks"], task["id"], status="in_progress")

    return {
        **state,
        "generated_files": accumulated,
        "_current_task_files": dict(MOCK_GENERATED_FILES),  # scoped for TestGenerator
        "tasks": updated_tasks,
        "_entry_point": MOCK_ENTRY_POINT,
        "status": "executing",
    }


def mock_debug_node(state: ProjectState, attempts: int) -> ProjectState:
    """Return a known-good patch regardless of the error encountered."""
    log_mock_banner("DebugAgent")
    logger.info(
        "[MOCK] Patching %d file(s). Diagnosis: %s",
        len(MOCK_DEBUG_FILES),
        MOCK_DEBUG_DIAGNOSIS[:80],
    )

    write_files(project_id=state["project_id"], files=MOCK_DEBUG_FILES)
    merged = {**state["generated_files"], **MOCK_DEBUG_FILES}

    return {
        **state,
        "generated_files": merged,
        "debug_attempts": attempts,
        "last_debug_diagnosis": MOCK_DEBUG_DIAGNOSIS,
        "_last_error_context": state.get("error_context", ""),
        "error_context": "",
        "status": "coding",
    }


def mock_generate_tests(project_id: str) -> dict[str, str]:
    """Return pre-defined pytest tests (no LLM call)."""
    log_mock_banner("TestGenerator")
    logger.info("[MOCK] Returning %d pre-defined test file(s).", len(MOCK_GENERATED_TESTS))
    return dict(MOCK_GENERATED_TESTS)
