"""
routes/projects.py — Project endpoints for the AI CTO API.

Endpoints:
    GET    /projects                           List all known projects (from workspace)
    POST   /projects                           Start a new project run
    DELETE /projects/{project_id}              Cancel an active run
    GET    /projects/{project_id}/status       Current project status (from disk)
    GET    /projects/{project_id}/run_summary  Run audit log (from disk)
    GET    /projects/{project_id}/task_board   Full task board (from disk)

All GET endpoints read from disk artefacts written by the pipeline:
    workspace/<project_id>/task_board.json   — written by persist_task_board()
    workspace/<project_id>/run_summary.json  — written by RunLogger.save()

This means status is accurate even if the API process was restarted between
the POST and the GET.
"""

import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, status as http_status

import ai_cto.tools.file_writer as _fw
from ai_cto.api.models import (
    StartProjectRequest,
    StartProjectResponse,
    ProjectStatusResponse,
    ProjectSummaryResponse,
    ProjectListResponse,
    CancelProjectResponse,
    RunSummaryResponse,
    TaskBoardResponse,
)
from ai_cto.api.runner import (
    start_project_run, cancel_project_run,
    is_running, get_run_state, RunState,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _workspace(project_id: str) -> Path:
    return _fw.WORKSPACE_ROOT / project_id


def _read_json(path: Path, label: str) -> dict:
    """Read and parse a JSON file; raise 404 if it doesn't exist."""
    if not path.exists():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found for project '{path.parent.name}'. "
                   f"The project may still be running or has not started.",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{label} is corrupt: {exc}",
        )


def _slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:40] or "project"


# ── GET /projects ──────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=ProjectListResponse,
    summary="List all known projects",
    description=(
        "Discovers projects by scanning the workspace directory for subdirectories "
        "that contain a task_board.json. Returns a lightweight summary per project. "
        "The is_active flag is True only when this API process is currently running "
        "the pipeline for that project. Results are sorted by persisted_at descending "
        "(most recently updated first). Projects with no task_board.json are not listed."
    ),
)
def list_projects() -> ProjectListResponse:
    workspace_root = _fw.WORKSPACE_ROOT
    summaries: list[ProjectSummaryResponse] = []

    if workspace_root.is_dir():
        for project_dir in sorted(workspace_root.iterdir()):
            if not project_dir.is_dir():
                continue
            board_path = project_dir / "task_board.json"
            if not board_path.exists():
                continue

            try:
                board = json.loads(board_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # skip unreadable boards silently

            project_id = board.get("project_id", project_dir.name)
            status_report = board.get("status_report", {})

            # If the project is actively running in this process, override status
            live_status = status_report.get("status", "in_progress")
            if is_running(project_id):
                live_status = "in_progress"

            summaries.append(ProjectSummaryResponse(
                project_id=project_id,
                idea=board.get("project_goal", ""),
                status=live_status,
                tasks_total=len(board.get("tasks", [])),
                tasks_completed=len(board.get("completed_task_ids", [])),
                files_generated_count=board.get("generated_files_count", 0),
                persisted_at=board.get("persisted_at", ""),
                is_active=is_running(project_id),
            ))

    # Sort most-recently-updated first
    summaries.sort(key=lambda s: s.persisted_at, reverse=True)
    return ProjectListResponse(projects=summaries, total=len(summaries))


# ── POST /projects ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=StartProjectResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Start a new project run",
    description=(
        "Accepts an idea and optional project_id. "
        "Launches the AI CTO pipeline in the background and returns immediately. "
        "Returns 409 if a run for this project_id is already active."
    ),
)
def start_project(body: StartProjectRequest) -> StartProjectResponse:
    project_id = body.project_id or _slugify(body.idea)

    if is_running(project_id):
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"A run for project '{project_id}' is already in progress.",
        )

    started = start_project_run(idea=body.idea, project_id=project_id)
    if not started:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"A run for project '{project_id}' is already in progress.",
        )

    workspace_rel = f"workspace/{project_id}"
    logger.info("[API] Started project '%s'.", project_id)
    return StartProjectResponse(
        project_id=project_id,
        status="started",
        workspace=workspace_rel,
    )


# ── DELETE /projects/{project_id} ─────────────────────────────────────────────

@router.delete(
    "/{project_id}",
    response_model=CancelProjectResponse,
    summary="Cancel an active project run",
    description=(
        "Requests cancellation of a running project. "
        "The pipeline thread will stop after its current graph node completes "
        "and write a partial run_summary.json. "
        "Returns 404 if the project workspace does not exist. "
        "Returns 409 if the project has already completed, failed, or been cancelled."
    ),
)
def cancel_project(project_id: str) -> CancelProjectResponse:
    # 404 if no workspace at all
    board_path = _workspace(project_id) / "task_board.json"
    if not board_path.exists():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found. No task_board.json in workspace.",
        )

    # Attempt cancellation via in-process registry
    cancelled = cancel_project_run(project_id)
    if cancelled:
        return CancelProjectResponse(
            project_id=project_id,
            cancelled=True,
            message=f"Cancellation requested for '{project_id}'. "
                    "The pipeline will stop after the current node completes.",
        )

    # Not running in this process — determine why from disk + registry
    run_state = get_run_state(project_id)
    if run_state == RunState.CANCELLED:
        terminal_status = "already cancelled"
    elif run_state == RunState.COMPLETE:
        terminal_status = "already completed"
    elif run_state == RunState.FAILED:
        terminal_status = "already failed"
    else:
        # Process restarted — read persisted status from disk
        try:
            board = json.loads(board_path.read_text(encoding="utf-8"))
            sr = board.get("status_report", {})
            terminal_status = f"not running (last known status: {sr.get('status', 'unknown')})"
        except (json.JSONDecodeError, OSError):
            terminal_status = "not running"

    raise HTTPException(
        status_code=http_status.HTTP_409_CONFLICT,
        detail=f"Cannot cancel project '{project_id}': {terminal_status}.",
    )


# ── GET /projects/{project_id}/status ─────────────────────────────────────────

@router.get(
    "/{project_id}/status",
    response_model=ProjectStatusResponse,
    summary="Get current project status",
    description=(
        "Returns the machine-readable status at project, agent, and task level. "
        "Reads the status_report embedded in task_board.json. "
        "Returns 404 if the project has not yet written its first task_board."
    ),
)
def get_project_status(project_id: str) -> ProjectStatusResponse:
    board_path = _workspace(project_id) / "task_board.json"
    board = _read_json(board_path, "task_board.json")

    status_report = board.get("status_report")
    if status_report is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No status_report found in task_board.json for '{project_id}'.",
        )

    # If the project is running and has completed, reflect that in the response
    run_state = get_run_state(project_id)
    if run_state == RunState.RUNNING and status_report.get("status") not in (
        "completed", "failed", "blocked"
    ):
        status_report["status"] = "in_progress"

    return ProjectStatusResponse(**status_report)


# ── GET /projects/{project_id}/run_summary ────────────────────────────────────

@router.get(
    "/{project_id}/run_summary",
    response_model=RunSummaryResponse,
    summary="Get run audit log",
    description=(
        "Returns the full run summary including duration, event trace, and file list. "
        "Only available after the run completes (RunLogger.save() is called on completion). "
        "Returns 404 if the run has not finished."
    ),
)
def get_run_summary(project_id: str) -> RunSummaryResponse:
    summary_path = _workspace(project_id) / "run_summary.json"
    data = _read_json(summary_path, "run_summary.json")
    return RunSummaryResponse(**data)


# ── GET /projects/{project_id}/task_board ─────────────────────────────────────

@router.get(
    "/{project_id}/task_board",
    response_model=TaskBoardResponse,
    summary="Get full task board",
    description=(
        "Returns the full task board including all tasks, history, and metadata. "
        "Updated by the pipeline after each task transition. "
        "Returns 404 if the pipeline has not yet reached the first task."
    ),
)
def get_task_board(project_id: str) -> TaskBoardResponse:
    board_path = _workspace(project_id) / "task_board.json"
    data = _read_json(board_path, "task_board.json")
    return TaskBoardResponse(**data)
