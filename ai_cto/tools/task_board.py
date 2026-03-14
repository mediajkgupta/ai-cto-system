"""
task_board.py — Task board state helpers and disk persistence.

Provides pure functions for task state transitions and project-level tracking,
plus a persist function that writes task_board.json to the project workspace.

Design:
  - All transition helpers (mark_task_done, mark_task_failed, mark_task_blocked)
    return a PARTIAL state dict — callers merge with {**state, **updates}.
  - persist_task_board reads WORKSPACE_ROOT at call time (via the module
    reference) so test fixtures can monkeypatch it.
  - No LLM calls, no external I/O beyond the single JSON write.
"""

import json
import logging
from typing import Optional

import ai_cto.tools.file_writer as _fw
from ai_cto.state import (
    update_task_in_list,
    task_history_event,
    now_iso,
)

logger = logging.getLogger(__name__)


# ── Task state transitions ─────────────────────────────────────────────────────

def mark_task_done(state: dict, task_id: int) -> dict:
    """
    Return a partial state dict that marks a task as done.

    Updates: tasks (status + updated_at), completed_task_ids, task_history.
    Does NOT call persist_task_board — caller decides when to persist.
    """
    tasks = update_task_in_list(
        state["tasks"], task_id, status="done", updated_at=now_iso()
    )
    completed = list(state.get("completed_task_ids", [])) + [task_id]
    history = list(state.get("task_history", [])) + [
        task_history_event(task_id, "done")
    ]
    return {
        "tasks": tasks,
        "completed_task_ids": completed,
        "task_history": history,
    }


def mark_task_failed(state: dict, task_id: int, error_context: str = "") -> dict:
    """
    Return a partial state dict that marks a task as failed.

    Updates: tasks (status + error_context + updated_at), failed_task_ids,
             task_history.
    """
    tasks = update_task_in_list(
        state["tasks"], task_id,
        status="failed",
        error_context=error_context,
        updated_at=now_iso(),
    )
    failed = list(state.get("failed_task_ids", [])) + [task_id]
    history = list(state.get("task_history", [])) + [
        task_history_event(task_id, "failed", detail=error_context[:200] if error_context else "")
    ]
    return {
        "tasks": tasks,
        "failed_task_ids": failed,
        "task_history": history,
    }


def mark_task_blocked(state: dict, task_id: int, reason: str = "") -> dict:
    """
    Return a partial state dict that marks a task as blocked.

    A task is blocked when its dependencies can never be satisfied
    (e.g. a required task failed or is itself blocked).

    Updates: tasks (status + updated_at), blocked_task_ids, task_history.
    """
    tasks = update_task_in_list(
        state["tasks"], task_id, status="blocked", updated_at=now_iso()
    )
    blocked = list(state.get("blocked_task_ids", [])) + [task_id]
    history = list(state.get("task_history", [])) + [
        task_history_event(task_id, "blocked", detail=reason)
    ]
    return {
        "tasks": tasks,
        "blocked_task_ids": blocked,
        "task_history": history,
    }


# ── Project-level predicates and next-task selector ───────────────────────────
# These import from state at call time to avoid circular imports.

def project_is_complete(tasks: list) -> bool:
    """True when every task has status 'done'. False if the task list is empty."""
    from ai_cto.state import project_is_complete as _f
    return _f(tasks)


def project_is_blocked(tasks: list) -> bool:
    """True when pending tasks exist but none can run (dependency deadlock)."""
    from ai_cto.state import project_is_blocked as _f
    return _f(tasks)


def get_next_runnable_task(tasks: list):
    """Return the highest-priority pending unblocked task, or None."""
    from ai_cto.state import get_next_runnable_task as _f
    return _f(tasks)


# ── Task board persistence ─────────────────────────────────────────────────────

def persist_task_board(state: dict) -> "Path":
    """
    Write workspace/<project_id>/task_board.json from the current state.

    The file contains everything needed to inspect or resume a run:
      - project goal, status
      - full task list (with retries, timestamps, outputs, etc.)
      - completed / failed / blocked ID sets
      - chronological task history log

    Reads WORKSPACE_ROOT via module reference at call time so that test
    fixtures can monkeypatch ai_cto.tools.file_writer.WORKSPACE_ROOT.

    Returns the Path to the written file.
    """
    from pathlib import Path

    project_dir: Path = _fw.WORKSPACE_ROOT / state["project_id"]
    project_dir.mkdir(parents=True, exist_ok=True)

    board = {
        "project_id": state.get("project_id", ""),
        "project_goal": state.get("idea", ""),
        "architecture": state.get("architecture", ""),
        "project_status": state.get("project_status", "active"),
        "active_task_id": state.get("active_task_id"),
        "tasks": state.get("tasks", []),
        "completed_task_ids": state.get("completed_task_ids", []),
        "failed_task_ids": state.get("failed_task_ids", []),
        "blocked_task_ids": state.get("blocked_task_ids", []),
        "task_history": state.get("task_history", []),
        "persisted_at": now_iso(),
    }

    board_path = project_dir / "task_board.json"
    board_path.write_text(
        json.dumps(board, indent=2, default=str),
        encoding="utf-8",
    )
    logger.debug("[TaskBoard] Persisted %d task(s) to %s", len(board["tasks"]), board_path)
    return board_path


# ── Resume helpers ─────────────────────────────────────────────────────────────

def load_task_board(board_path: "Path") -> dict:
    """
    Read and parse task_board.json from disk.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the JSON is missing required keys.
    """
    from pathlib import Path

    path = Path(board_path)
    if not path.exists():
        raise FileNotFoundError(f"task_board.json not found: {path}")

    board = json.loads(path.read_text(encoding="utf-8"))

    for key in ("project_id", "project_goal", "tasks"):
        if key not in board:
            raise ValueError(f"task_board.json is missing required key '{key}'")

    return board


def resume_state_from_board(board_path: "Path") -> "ProjectState":
    """
    Reconstruct a ProjectState from a persisted task_board.json so that
    the pipeline can continue from where it left off.

    What this does:
      1. Reads task_board.json from disk.
      2. Resets any 'in_progress' tasks back to 'pending' — they were
         interrupted and must re-run from the coding step.
      3. Resets their 'retries' counter (the interrupted run didn't count).
      4. Determines the next runnable task via get_next_runnable_task().
      5. Scans the workspace directory for generated files already on disk.
      6. Returns a full ProjectState ready to feed into the resume graph
         (which starts at 'coding' rather than 'memory_retrieve').

    Args:
        board_path: Path to the task_board.json file.

    Returns:
        A ProjectState dict with status='coding'.

    Raises:
        FileNotFoundError: if board_path does not exist.
        ValueError: if the board is missing required fields or is not resumable.
    """
    from pathlib import Path
    from ai_cto.state import (
        initial_state,
        get_next_runnable_task,
        task_history_event,
        project_is_complete,
        project_is_blocked,
    )

    board_path = Path(board_path)
    board = load_task_board(board_path)

    project_status = board.get("project_status", "active")
    if project_status == "complete":
        raise ValueError(
            f"Project '{board['project_id']}' is already complete — nothing to resume."
        )

    # ── 1. Reset interrupted in_progress tasks back to pending ────────────────
    tasks = []
    for t in board["tasks"]:
        if t.get("status") == "in_progress":
            tasks.append({**t, "status": "pending", "retries": 0})
            logger.info(
                "[Resume] Task id=%d '%s' was in_progress; reset to pending.",
                t["id"], t["title"],
            )
        else:
            tasks.append(dict(t))

    # ── 2. Determine the next runnable task ────────────────────────────────────
    if project_is_complete(tasks):
        raise ValueError(
            f"Project '{board['project_id']}' is already complete — nothing to resume."
        )
    if project_is_blocked(tasks):
        raise ValueError(
            f"Project '{board['project_id']}' is blocked — cannot resume. "
            "Fix the dependency deadlock first."
        )

    next_task = get_next_runnable_task(tasks)
    if next_task is None:
        raise ValueError(
            f"No runnable task found in '{board['project_id']}'. "
            "All pending tasks have unmet dependencies."
        )

    next_idx = next(
        (i for i, t in enumerate(tasks) if t["id"] == next_task["id"]), 0
    )

    # ── 3. Scan workspace for files already generated ─────────────────────────
    project_dir = board_path.parent
    generated_files = _scan_workspace_files(project_dir)
    logger.info(
        "[Resume] Found %d file(s) already on disk in %s.",
        len(generated_files), project_dir,
    )

    # ── 4. Build history entry for the resumed task ───────────────────────────
    history = list(board.get("task_history", [])) + [
        task_history_event(next_task["id"], "started", detail="(resumed)")
    ]

    # ── 5. Assemble full state ─────────────────────────────────────────────────
    base = initial_state(
        idea=board["project_goal"],
        project_id=board["project_id"],
    )
    return {
        **base,
        "architecture": board.get("architecture", ""),
        "tasks": tasks,
        "active_task_id": next_task["id"],
        "current_task_index": next_idx,
        "completed_task_ids": list(board.get("completed_task_ids", [])),
        "failed_task_ids": list(board.get("failed_task_ids", [])),
        "blocked_task_ids": list(board.get("blocked_task_ids", [])),
        "task_history": history,
        "project_status": "active",
        "generated_files": generated_files,
        "status": "coding",
    }


def _scan_workspace_files(project_dir: "Path") -> dict:
    """
    Walk project_dir and return a dict of {relative_path: content} for
    all non-binary, non-metadata files found.

    Skips: task_board.json, tests/ directory, __pycache__/, .pyc files.
    """
    from pathlib import Path

    result = {}
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        return result

    _SKIP_NAMES = {"task_board.json", "__pycache__"}
    _SKIP_EXTS = {".pyc", ".pyo"}
    _SKIP_DIRS = {"__pycache__", ".git", "tests"}

    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip by name or extension
        if path.name in _SKIP_NAMES or path.suffix in _SKIP_EXTS:
            continue
        # Skip if any parent directory is in the skip list
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(project_dir))
        try:
            result[rel] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            pass  # binary or unreadable — skip silently

    return result
