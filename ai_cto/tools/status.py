"""
status.py — Machine-readable project status at project, agent, and task level.

Maps internal LangGraph routing states to a clean, user-facing status vocabulary:

  Internal           →  User-facing
  ──────────────────────────────────
  project_status:
    active           →  in_progress
    complete         →  completed
    failed           →  failed
    blocked          →  blocked

  routing status (status field):
    planning         →  planned
    task_managing    →  planned
    coding           →  in_progress
    verifying*       →  in_progress
    executing        →  in_progress
    debugging        →  in_progress
    saving_memory    →  in_progress
    done             →  completed
    failed           →  failed
    blocked          →  blocked

  task status:
    pending          →  planned
    in_progress      →  in_progress
    done             →  completed
    failed           →  failed
    blocked          →  blocked

Usage:
    report = get_status_report(state)
    print(format_status_report(report))
"""

from typing import TypedDict, Optional
from ai_cto.state import now_iso


# ── User-facing vocabulary ─────────────────────────────────────────────────────

# These are the only values that appear in the public status report.
# Internal LangGraph states ("saving_memory", "verifying_failed") are never exposed.
STATUS_PLANNED     = "planned"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED   = "completed"
STATUS_FAILED      = "failed"
STATUS_BLOCKED     = "blocked"


# ── TypedDicts for the status report ──────────────────────────────────────────

class TaskStatus(TypedDict):
    id: int
    title: str
    status: str       # planned | in_progress | completed | failed | blocked
    priority: int
    retries: int
    max_retries: int
    outputs: list
    depends_on: list


class AgentStatus(TypedDict):
    current_stage: str    # the active graph node name
    pipeline_status: str  # planned | in_progress | completed | failed | blocked
    active_task_id: Optional[int]
    debug_attempts: int
    execution_timeout_count: int


class ProjectStatus(TypedDict):
    project_id: str
    idea: str
    status: str           # planned | in_progress | completed | failed | blocked
    tasks_total: int
    tasks_planned: int
    tasks_in_progress: int
    tasks_completed: int
    tasks_failed: int
    tasks_blocked: int
    files_generated_count: int
    agent: AgentStatus
    tasks: list           # list[TaskStatus]
    generated_at: str     # ISO-8601 UTC


# ── Mapping functions ──────────────────────────────────────────────────────────

_PROJECT_STATUS_MAP = {
    "active":   STATUS_IN_PROGRESS,
    "complete": STATUS_COMPLETED,
    "failed":   STATUS_FAILED,
    "blocked":  STATUS_BLOCKED,
}

_ROUTING_STATUS_MAP = {
    "planning":          STATUS_PLANNED,
    "task_managing":     STATUS_PLANNED,
    "coding":            STATUS_IN_PROGRESS,
    "verifying":         STATUS_IN_PROGRESS,
    "verifying_failed":  STATUS_IN_PROGRESS,
    "executing":         STATUS_IN_PROGRESS,
    "debugging":         STATUS_IN_PROGRESS,
    "saving_memory":     STATUS_IN_PROGRESS,
    "done":              STATUS_COMPLETED,
    "failed":            STATUS_FAILED,
    "blocked":           STATUS_BLOCKED,
}

_TASK_STATUS_MAP = {
    "pending":     STATUS_PLANNED,
    "in_progress": STATUS_IN_PROGRESS,
    "done":        STATUS_COMPLETED,
    "failed":      STATUS_FAILED,
    "blocked":     STATUS_BLOCKED,
}


def _map_project_status(state: dict) -> str:
    """Return user-facing project status from state.

    Routing status is always the authoritative source when it maps to a known
    user-facing value.  project_status is only used as a fallback for states
    that the routing map does not cover (defensive).
    """
    routing = state.get("status", "")
    if routing in _ROUTING_STATUS_MAP:
        return _ROUTING_STATUS_MAP[routing]
    project = state.get("project_status", "active")
    return _PROJECT_STATUS_MAP.get(project, STATUS_IN_PROGRESS)


def _map_pipeline_status(routing_status: str) -> str:
    return _ROUTING_STATUS_MAP.get(routing_status, STATUS_IN_PROGRESS)


def _map_task_status(task_status: str) -> str:
    return _TASK_STATUS_MAP.get(task_status, STATUS_PLANNED)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_status_report(state: dict) -> ProjectStatus:
    """
    Build a machine-readable ProjectStatus from the current pipeline state.

    Can be called at any point during or after a run. Safe to call with
    partial state (e.g. right after initial_state() is created).
    """
    routing_status = state.get("status", "planning")
    tasks_raw = state.get("tasks", [])
    generated_files = state.get("generated_files", {})

    # ── Task statuses ──────────────────────────────────────────────────────────
    task_statuses: list[TaskStatus] = []
    counts = {s: 0 for s in [STATUS_PLANNED, STATUS_IN_PROGRESS, STATUS_COMPLETED,
                               STATUS_FAILED, STATUS_BLOCKED]}

    for t in tasks_raw:
        mapped = _map_task_status(t.get("status", "pending"))
        counts[mapped] += 1
        task_statuses.append(TaskStatus(
            id=t["id"],
            title=t.get("title", ""),
            status=mapped,
            priority=t.get("priority", 5),
            retries=t.get("retries", 0),
            max_retries=t.get("max_retries", 3),
            outputs=t.get("outputs", []),
            depends_on=t.get("depends_on", []),
        ))

    # ── Agent status ───────────────────────────────────────────────────────────
    agent = AgentStatus(
        current_stage=routing_status,
        pipeline_status=_map_pipeline_status(routing_status),
        active_task_id=state.get("active_task_id"),
        debug_attempts=state.get("debug_attempts", 0),
        execution_timeout_count=state.get("execution_timeout_count", 0),
    )

    return ProjectStatus(
        project_id=state.get("project_id", ""),
        idea=state.get("idea", ""),
        status=_map_project_status(state),
        tasks_total=len(tasks_raw),
        tasks_planned=counts[STATUS_PLANNED],
        tasks_in_progress=counts[STATUS_IN_PROGRESS],
        tasks_completed=counts[STATUS_COMPLETED],
        tasks_failed=counts[STATUS_FAILED],
        tasks_blocked=counts[STATUS_BLOCKED],
        files_generated_count=len(generated_files),
        agent=agent,
        tasks=task_statuses,
        generated_at=now_iso(),
    )


def format_status_report(report: ProjectStatus) -> str:
    """
    Return a human-readable multi-line summary of a ProjectStatus report.

    Suitable for CLI output, logging, or the run_summary.json.
    """
    lines = [
        f"Project:  {report['project_id']}",
        f"Idea:     {report['idea']}",
        f"Status:   {report['status'].upper()}",
        f"",
        f"Tasks:    {report['tasks_total']} total",
        f"          {report['tasks_planned']} planned",
        f"          {report['tasks_in_progress']} in_progress",
        f"          {report['tasks_completed']} completed",
        f"          {report['tasks_failed']} failed",
        f"          {report['tasks_blocked']} blocked",
        f"Files:    {report['files_generated_count']} generated",
        f"",
        f"Agent pipeline:",
        f"  stage:            {report['agent']['current_stage']}",
        f"  pipeline_status:  {report['agent']['pipeline_status']}",
        f"  active_task_id:   {report['agent']['active_task_id']}",
        f"  debug_attempts:   {report['agent']['debug_attempts']}",
    ]

    if report["tasks"]:
        lines += ["", "Task detail:"]
        for t in report["tasks"]:
            retry_str = f"  retries={t['retries']}/{t['max_retries']}" if t["retries"] > 0 else ""
            dep_str = f"  deps={t['depends_on']}" if t["depends_on"] else ""
            lines.append(
                f"  [{t['id']:>3}] {t['status']:<12}  {t['title']}{retry_str}{dep_str}"
            )

    return "\n".join(lines)
