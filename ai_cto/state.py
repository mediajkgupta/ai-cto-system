"""
state.py — Shared project state for the AI CTO System.

All agents read from and write to a single ProjectState object.
LangGraph passes this state through the graph on every node invocation.

Design principle: agents are pure functions (state_in → state_out).
The graph decides routing; agents only transform state.
"""

from typing import TypedDict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_cto.verification.checks import VerificationResult


class TaskItem(TypedDict):
    """A single unit of development work."""
    id: int
    title: str
    description: str
    status: str  # pending | in_progress | done | failed


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
        2. PlanningAgent populates `architecture` and `tasks`
        3. CodingAgent populates `generated_files` for current task
        4. ExecutionAgent populates `execution_result`
        5. DebugAgent patches `generated_files` on failure
        6. Controller routes back to coding or advances to done
    """

    # ── Input ─────────────────────────────────────────────────────────────
    idea: str
    """The raw user idea or product description."""

    project_id: str
    """Unique slug used for the workspace directory (e.g. 'my-saas-app')."""

    # ── Planning phase ────────────────────────────────────────────────────
    architecture: str
    """Markdown architecture document produced by the PlanningAgent."""

    tasks: list[TaskItem]
    """Ordered task list produced by the PlanningAgent."""

    current_task_index: int
    """Index into `tasks` pointing at the task currently being worked on."""

    # ── Coding phase ──────────────────────────────────────────────────────
    generated_files: dict[str, str]
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
    Typed as dict to avoid circular import; cast to VerificationResult when needed.
    """

    # ── Execution phase ───────────────────────────────────────────────────
    execution_result: Optional[ExecutionResult]
    """Populated after the ExecutionAgent runs the sandbox."""

    # ── Debug phase ───────────────────────────────────────────────────────
    error_context: str
    """Combined error message fed to the DebugAgent (stderr + traceback)."""

    debug_attempts: int
    """How many times the DebugAgent has been invoked for the current task."""

    max_debug_attempts: int
    """Hard cap on debug retries to prevent infinite loops. Default: 3."""

    # ── Memory phase ──────────────────────────────────────────────────────────
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
    Current pipeline status. One of:
        planning   → waiting for architecture
        coding     → generating source files
        executing  → running in sandbox
        debugging  → analysing errors
        done       → task completed successfully
        failed     → exceeded retry limit or unrecoverable error
    """

    final_output: str
    """Human-readable summary of what was built (populated on completion)."""


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
