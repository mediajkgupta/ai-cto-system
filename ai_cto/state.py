"""
state.py — Shared project state for the AI CTO System.

All agents read from and write to a single ProjectState object.
LangGraph passes this state through the graph on every node invocation.

Design principle: agents are pure functions (state_in → state_out).
The graph decides routing; agents only transform state.
"""

from datetime import datetime, timezone
from typing import TypedDict, Optional


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TaskItem(TypedDict):
    """A single unit of development work."""
    id: int
    title: str
    description: str
    status: str      # pending | in_progress | done | failed | blocked
    priority: int    # 1 = highest; lower number runs first; default 5
    depends_on: list # list of task IDs that must be done before this one starts
    outputs: list    # relative file paths expected to be produced by this task
    error_context: str  # last error message if this task reached 'failed' status
    retries: int        # how many debug cycles this task has consumed
    max_retries: int    # per-task retry cap; default 3
    created_at: str     # ISO-8601 UTC, set once at creation
    updated_at: str     # ISO-8601 UTC, updated on every status transition


class ExecutionResult(TypedDict):
    """Output from running generated code in the sandbox."""
    stdout: str
    stderr: str
    exit_code: int


class ProjectState(TypedDict):
    """
    Central shared state that flows through the entire agent graph.

    Lifecycle:
        1. User provides `idea`
        2. PlanningAgent populates `architecture`
        3. TaskManagerAgent populates `tasks`, `active_task_id`, `task_history`
        4. CodingAgent populates `generated_files` for the active task
        5. VerificationAgent + ExecutionAgent validate and run the code
        6. DebugAgent patches `generated_files` on failure (per-task retries)
        7. Controller routes back to coding or advances to the next task
        8. MemorySave persists learnings on success
    """

    # ── Input ─────────────────────────────────────────────────────────────
    idea: str
    """The raw user idea or product description."""

    project_id: str
    """Unique slug used for the workspace directory (e.g. 'my-saas-app')."""

    # ── Planning phase ────────────────────────────────────────────────────
    architecture: str
    """Markdown architecture document produced by the PlanningAgent."""

    tasks: list
    """Ordered task list produced by the TaskManagerAgent."""

    current_task_index: int
    """Index into `tasks` pointing at the active task (kept in sync with active_task_id)."""

    active_task_id: Optional[int]
    """ID of the task currently being worked on. Primary pointer after task_manager runs."""

    # ── Task orchestration ────────────────────────────────────────────────
    completed_task_ids: list
    """IDs of tasks that have reached status 'done'."""

    failed_task_ids: list
    """IDs of tasks that exhausted their retries and reached status 'failed'."""

    blocked_task_ids: list
    """IDs of tasks whose dependencies can never be satisfied (status 'blocked')."""

    task_history: list
    """
    Append-only chronological log of task state transitions.
    Each entry is a dict: {task_id, event, detail, timestamp}.
    Events: started | done | failed | blocked | retry
    """

    project_status: str
    """
    Coarse project lifecycle status. One of:
        active    → tasks are being executed
        complete  → all tasks done, memory saved
        failed    → a task exhausted retries, project halted
        blocked   → pending tasks exist but none can run (dep deadlock)
    """

    # ── Coding phase ──────────────────────────────────────────────────────
    generated_files: dict
    """
    Map of relative file paths to file contents.
    Example: {"main.py": "print('hello')", "utils/helper.py": "..."}
    Written to disk by file_writer before execution.
    """

    # ── Verification phase ────────────────────────────────────────────────
    verification_result: Optional[dict]
    """
    VerificationResult dict populated by the VerificationAgent.
    Contains: passed, issues, generated_tests, error_count, warning_count, summary.
    """

    # ── Execution phase ───────────────────────────────────────────────────
    execution_result: Optional[ExecutionResult]
    """Populated after the ExecutionAgent runs the sandbox."""

    # ── Debug phase ───────────────────────────────────────────────────────
    error_context: str
    """Combined error message fed to the DebugAgent (stderr + traceback)."""

    debug_attempts: int
    """
    Debug attempts for the currently active task in this execution pass.
    Resets to 0 when advancing to a new task.
    Kept for backward compatibility with the MemoryAgent which uses it to
    decide whether to persist a debug solution.
    Per-task retry tracking uses TaskItem.retries instead.
    """

    max_debug_attempts: int
    """
    Fallback retry cap used when the active task has no max_retries field.
    Default: 3. Per-task cap is TaskItem.max_retries (also defaults to 3).
    """

    # ── Memory phase ──────────────────────────────────────────────────────
    memory_context: str
    """
    Formatted string of relevant past experiences retrieved from the vector DB.
    Injected into the PlanningAgent and DebugAgent prompts.
    Empty string when no relevant memories exist.
    """

    last_debug_diagnosis: str
    """
    The most recent diagnosis text produced by the DebugAgent.
    Persisted to memory on project completion so future projects can
    recognise and fix similar errors faster.
    """

    _last_error_context: str
    """
    The error context that triggered the last successful debug.
    Stored alongside last_debug_diagnosis in memory.
    """

    # ── Output ────────────────────────────────────────────────────────────
    status: str
    """
    Graph routing status. One of:
        planning      → waiting for architecture
        task_managing → architecture done, building task list
        coding        → generating source files
        executing     → running in sandbox
        verifying_failed → static checks found errors
        debugging     → analysing errors
        saving_memory → all tasks done, persisting to vector DB
        blocked       → dep deadlock; no runnable tasks remain
        done          → project completed successfully
        failed        → exceeded retry limit or unrecoverable error
    """

    final_output: str
    """Human-readable summary of what was built (populated on completion or failure)."""


# ── Default factory ────────────────────────────────────────────────────────────

def initial_state(idea: str, project_id: str) -> ProjectState:
    """
    Return a clean ProjectState for a new project run.

    Usage:
        state = initial_state("Build a todo API", "todo-api")
    """
    return ProjectState(
        idea=idea,
        project_id=project_id,
        architecture="",
        tasks=[],
        current_task_index=0,
        active_task_id=None,
        completed_task_ids=[],
        failed_task_ids=[],
        blocked_task_ids=[],
        task_history=[],
        project_status="active",
        generated_files={},
        verification_result=None,
        execution_result=None,
        error_context="",
        debug_attempts=0,
        max_debug_attempts=3,
        status="planning",
        final_output="",
        memory_context="",
        last_debug_diagnosis="",
        _last_error_context="",
    )


# ── Task factories and helpers ─────────────────────────────────────────────────

def make_task_item(
    id: int,
    title: str,
    description: str,
    priority: int = 5,
    depends_on: Optional[list] = None,
    outputs: Optional[list] = None,
    max_retries: int = 3,
) -> TaskItem:
    """Factory for TaskItem with sensible defaults for all optional fields."""
    ts = now_iso()
    return TaskItem(
        id=id,
        title=title,
        description=description,
        status="pending",
        priority=priority,
        depends_on=depends_on or [],
        outputs=outputs or [],
        error_context="",
        retries=0,
        max_retries=max_retries,
        created_at=ts,
        updated_at=ts,
    )


def get_active_task(state: ProjectState) -> Optional[TaskItem]:
    """Return the task pointed to by active_task_id, or None."""
    task_id = state.get("active_task_id")
    if task_id is None:
        return None
    return next((t for t in state["tasks"] if t["id"] == task_id), None)


def update_task_in_list(tasks: list, task_id: int, **updates) -> list:
    """Return a new list with the specified task's fields updated."""
    return [{**t, **updates} if t["id"] == task_id else t for t in tasks]


def get_next_runnable_task(tasks: list) -> Optional[TaskItem]:
    """
    Return the highest-priority pending task whose dependencies are all done.

    Selection rules:
        1. task.status must be "pending"
        2. All IDs in task.depends_on must belong to tasks with status "done"
        3. Among eligible candidates, lowest priority number wins (1 = highest)
        4. Ties broken by insertion order
    """
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    candidates = [
        t for t in tasks
        if t["status"] == "pending"
        and all(dep in done_ids for dep in t.get("depends_on", []))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t.get("priority", 5))


# V1 backward-compatibility alias — do not remove; imported by graph.py, task_manager.py,
# mock_llm.py, and existing tests.
next_unblocked_task = get_next_runnable_task


def project_is_complete(tasks: list) -> bool:
    """True when every task has status 'done'. False if the list is empty."""
    return bool(tasks) and all(t["status"] == "done" for t in tasks)


def project_is_blocked(tasks: list) -> bool:
    """
    True when at least one task is still pending but no task can run.

    Distinguishes between:
        - "all done" (project_is_complete) → should save memory
        - "stuck"    (project_is_blocked)  → dependency deadlock, must halt
    """
    pending = [t for t in tasks if t["status"] == "pending"]
    if not pending:
        return False
    return get_next_runnable_task(tasks) is None


def task_history_event(task_id: int, event: str, detail: str = "") -> dict:
    """
    Create a single task history log entry.

    Args:
        task_id — the task this event refers to
        event   — one of: started | done | failed | blocked | retry
        detail  — optional human-readable context (e.g. error snippet)
    """
    return {
        "task_id": task_id,
        "event": event,
        "detail": detail,
        "timestamp": now_iso(),
    }
