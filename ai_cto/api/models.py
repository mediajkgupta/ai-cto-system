"""
models.py — Pydantic request/response models for the AI CTO API.

All response models are derived from existing internal TypedDicts but
re-declared as Pydantic BaseModels for automatic JSON serialisation,
OpenAPI schema generation, and input validation.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Request bodies ─────────────────────────────────────────────────────────────

class StartProjectRequest(BaseModel):
    idea: str = Field(..., min_length=3, max_length=2000,
                      description="Natural language description of the software to build.")
    project_id: Optional[str] = Field(
        None,
        pattern=r"^[a-z0-9][a-z0-9\-]{0,38}[a-z0-9]$",
        description=(
            "Optional project slug (used as workspace directory name). "
            "Auto-generated from idea if omitted. "
            "Must be lowercase alphanumeric + hyphens, 2–40 chars."
        ),
    )


# ── Response bodies ────────────────────────────────────────────────────────────

class StartProjectResponse(BaseModel):
    project_id: str
    status: str          # "started" | "already_running" | "queued"
    workspace: str       # relative path to workspace directory


class AgentStatusResponse(BaseModel):
    current_stage: str
    pipeline_status: str
    active_task_id: Optional[int]
    debug_attempts: int
    execution_timeout_count: int


class TaskStatusResponse(BaseModel):
    id: int
    title: str
    status: str          # planned | in_progress | completed | failed | blocked
    priority: int
    retries: int
    max_retries: int
    outputs: list
    depends_on: list


class ProjectStatusResponse(BaseModel):
    project_id: str
    idea: str
    status: str          # planned | in_progress | completed | failed | blocked
    tasks_total: int
    tasks_planned: int
    tasks_in_progress: int
    tasks_completed: int
    tasks_failed: int
    tasks_blocked: int
    files_generated_count: int
    agent: AgentStatusResponse
    tasks: list[TaskStatusResponse]
    generated_at: str


class RunSummaryResponse(BaseModel):
    project_id: str
    idea: str
    started_at: str
    finished_at: str
    duration_seconds: Optional[int]
    status: ProjectStatusResponse
    files_generated: list[str]
    files_generated_count: int
    events: list[dict]


class TaskBoardResponse(BaseModel):
    project_id: str
    project_goal: str
    architecture: str
    project_status: str
    active_task_id: Optional[int]
    tasks: list[dict]
    completed_task_ids: list
    failed_task_ids: list
    blocked_task_ids: list
    task_history: list
    debug_retries: int
    generated_files_count: int
    generated_files: list[str]
    status_report: Any
    persisted_at: str


class ProjectSummaryResponse(BaseModel):
    """Lightweight per-project entry returned by GET /projects."""
    project_id: str
    idea: str
    status: str          # planned | in_progress | completed | failed | blocked | cancelled
    tasks_total: int
    tasks_completed: int
    files_generated_count: int
    persisted_at: str
    is_active: bool      # True only if this process is currently running the pipeline


class ProjectListResponse(BaseModel):
    projects: list[ProjectSummaryResponse]
    total: int


class CancelProjectResponse(BaseModel):
    project_id: str
    cancelled: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
