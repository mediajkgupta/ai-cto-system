"""
test_api.py — Tests for the FastAPI backend.

Uses Starlette's synchronous TestClient.  The pipeline runner is monkeypatched
so tests never call real LLM APIs or start background threads.

Coverage:
  - GET    /projects                 (list all — disk-based discovery)
  - POST   /projects
  - DELETE /projects/{id}            (cancel)
  - GET    /projects/{id}/status
  - GET    /projects/{id}/run_summary
  - GET    /projects/{id}/task_board
  - GET    /health
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from ai_cto.api.app import create_app
from ai_cto.state import initial_state, make_task_item
from ai_cto.tools.status import get_status_report


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Fresh TestClient per test (fresh app to reset runner state)."""
    return TestClient(create_app(), raise_server_exceptions=True)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Redirect WORKSPACE_ROOT to a temp dir; return workspace root Path."""
    import ai_cto.tools.file_writer as _fw
    monkeypatch.setattr(_fw, "WORKSPACE_ROOT", tmp_path)

    # Also patch the reference used inside routes/projects.py
    import ai_cto.api.routes.projects as _routes
    monkeypatch.setattr(_routes, "_fw", _fw)

    return tmp_path


def _write_task_board(workspace: Path, project_id: str, state: dict) -> Path:
    """Write a task_board.json in the style persist_task_board() produces."""
    from ai_cto.tools.status import get_status_report
    from ai_cto.state import now_iso
    proj_dir = workspace / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    generated_files = state.get("generated_files", {})
    board = {
        "project_id": project_id,
        "project_goal": state.get("idea", ""),
        "architecture": state.get("architecture", ""),
        "project_status": state.get("project_status", "active"),
        "active_task_id": state.get("active_task_id"),
        "tasks": state.get("tasks", []),
        "completed_task_ids": state.get("completed_task_ids", []),
        "failed_task_ids": state.get("failed_task_ids", []),
        "blocked_task_ids": state.get("blocked_task_ids", []),
        "task_history": [],
        "debug_retries": state.get("debug_attempts", 0),
        "generated_files_count": len(generated_files),
        "generated_files": sorted(generated_files.keys()),
        "status_report": get_status_report(state),
        "persisted_at": now_iso(),
    }
    p = proj_dir / "task_board.json"
    p.write_text(json.dumps(board), encoding="utf-8")
    return p


def _write_run_summary(workspace: Path, project_id: str, state: dict) -> Path:
    """Write a run_summary.json in the style RunLogger.save() produces."""
    from ai_cto.tools.status import get_status_report
    proj_dir = workspace / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    generated_files = state.get("generated_files", {})
    summary = {
        "project_id": project_id,
        "idea": state.get("idea", ""),
        "started_at": "2026-03-15T00:00:00+00:00",
        "finished_at": "2026-03-15T00:01:00+00:00",
        "duration_seconds": 60,
        "status": get_status_report(state),
        "files_generated": sorted(generated_files.keys()),
        "files_generated_count": len(generated_files),
        "events": [
            {"node": "planning", "timestamp": "2026-03-15T00:00:01+00:00",
             "status": "planning", "active_task_id": None},
        ],
    }
    p = proj_dir / "run_summary.json"
    p.write_text(json.dumps(summary), encoding="utf-8")
    return p


def _state_done(project_id: str) -> dict:
    state = initial_state("Build a REST API", project_id)
    state["status"] = "done"
    state["project_status"] = "complete"
    state["tasks"] = [
        {**make_task_item(1, "Foundation", "desc"), "status": "done"},
        {**make_task_item(2, "Routes", "desc"), "status": "done"},
    ]
    state["completed_task_ids"] = [1, 2]
    state["generated_files"] = {"main.py": "x", "routes.py": "y"}
    return state


# ── GET /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── POST /projects ───────────────────────────────────────────────────────────────

class TestStartProject:
    def test_returns_202_and_project_id(self, client):
        with patch("ai_cto.api.routes.projects.start_project_run", return_value=True):
            r = client.post("/projects", json={"idea": "Build a todo app"})
        assert r.status_code == 202
        body = r.json()
        assert "project_id" in body
        assert body["status"] == "started"

    def test_explicit_project_id_used(self, client):
        with patch("ai_cto.api.routes.projects.start_project_run", return_value=True):
            r = client.post("/projects", json={"idea": "todo app", "project_id": "my-todo"})
        assert r.status_code == 202
        assert r.json()["project_id"] == "my-todo"

    def test_workspace_path_in_response(self, client):
        with patch("ai_cto.api.routes.projects.start_project_run", return_value=True):
            r = client.post("/projects", json={"idea": "todo", "project_id": "ws-test"})
        assert r.json()["workspace"] == "workspace/ws-test"

    def test_duplicate_run_returns_409(self, client):
        with patch("ai_cto.api.routes.projects.is_running", return_value=True):
            r = client.post("/projects", json={"idea": "todo", "project_id": "dup"})
        assert r.status_code == 409

    def test_missing_idea_returns_422(self, client):
        r = client.post("/projects", json={})
        assert r.status_code == 422

    def test_idea_too_short_returns_422(self, client):
        r = client.post("/projects", json={"idea": "ab"})
        assert r.status_code == 422

    def test_invalid_project_id_returns_422(self, client):
        r = client.post("/projects", json={"idea": "build something", "project_id": "MY UPPER CASE"})
        assert r.status_code == 422

    def test_auto_slug_from_idea(self, client):
        with patch("ai_cto.api.routes.projects.start_project_run", return_value=True):
            r = client.post("/projects", json={"idea": "Build a Weather Dashboard!"})
        assert r.status_code == 202
        pid = r.json()["project_id"]
        assert pid == "build-a-weather-dashboard"


# ── GET /projects/{id}/status ────────────────────────────────────────────────────

class TestGetStatus:
    def test_returns_status_report(self, client, workspace):
        state = _state_done("status-test")
        _write_task_board(workspace, "status-test", state)
        r = client.get("/projects/status-test/status")
        assert r.status_code == 200
        body = r.json()
        assert body["project_id"] == "status-test"
        assert body["status"] == "completed"
        assert body["tasks_total"] == 2
        assert body["tasks_completed"] == 2

    def test_404_when_no_task_board(self, client, workspace):
        r = client.get("/projects/nonexistent/status")
        assert r.status_code == 404

    def test_agent_section_present(self, client, workspace):
        state = _state_done("agent-test")
        _write_task_board(workspace, "agent-test", state)
        r = client.get("/projects/agent-test/status")
        body = r.json()
        assert "agent" in body
        assert "current_stage" in body["agent"]
        assert "pipeline_status" in body["agent"]

    def test_tasks_section_present(self, client, workspace):
        state = _state_done("tasks-test")
        _write_task_board(workspace, "tasks-test", state)
        r = client.get("/projects/tasks-test/status")
        body = r.json()
        assert len(body["tasks"]) == 2
        assert all(t["status"] == "completed" for t in body["tasks"])

    def test_in_progress_project(self, client, workspace):
        state = initial_state("build a thing", "wip-proj")
        state["status"] = "coding"
        state["project_status"] = "active"
        state["tasks"] = [
            {**make_task_item(1, "Foundation", "desc"), "status": "done"},
            {**make_task_item(2, "Routes", "desc"), "status": "pending"},
        ]
        _write_task_board(workspace, "wip-proj", state)
        r = client.get("/projects/wip-proj/status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_progress"
        assert body["tasks_completed"] == 1
        assert body["tasks_planned"] == 1


# ── GET /projects/{id}/run_summary ───────────────────────────────────────────────

class TestGetRunSummary:
    def test_returns_summary(self, client, workspace):
        state = _state_done("summary-test")
        _write_run_summary(workspace, "summary-test", state)
        r = client.get("/projects/summary-test/run_summary")
        assert r.status_code == 200
        body = r.json()
        assert body["project_id"] == "summary-test"
        assert body["duration_seconds"] == 60
        assert body["files_generated_count"] == 2

    def test_404_when_no_run_summary(self, client, workspace):
        r = client.get("/projects/nonexistent/run_summary")
        assert r.status_code == 404

    def test_status_block_in_summary(self, client, workspace):
        state = _state_done("summary-status")
        _write_run_summary(workspace, "summary-status", state)
        r = client.get("/projects/summary-status/run_summary")
        body = r.json()
        assert "status" in body
        assert body["status"]["status"] == "completed"

    def test_events_list_in_summary(self, client, workspace):
        state = _state_done("events-test")
        _write_run_summary(workspace, "events-test", state)
        r = client.get("/projects/events-test/run_summary")
        body = r.json()
        assert isinstance(body["events"], list)
        assert len(body["events"]) == 1


# ── GET /projects/{id}/task_board ────────────────────────────────────────────────

class TestGetTaskBoard:
    def test_returns_task_board(self, client, workspace):
        state = _state_done("board-test")
        _write_task_board(workspace, "board-test", state)
        r = client.get("/projects/board-test/task_board")
        assert r.status_code == 200
        body = r.json()
        assert body["project_id"] == "board-test"
        assert body["project_goal"] == "Build a REST API"
        assert len(body["tasks"]) == 2

    def test_404_when_no_board(self, client, workspace):
        r = client.get("/projects/no-board/task_board")
        assert r.status_code == 404

    def test_board_contains_completed_ids(self, client, workspace):
        state = _state_done("complete-ids")
        _write_task_board(workspace, "complete-ids", state)
        r = client.get("/projects/complete-ids/task_board")
        body = r.json()
        assert body["completed_task_ids"] == [1, 2]

    def test_board_contains_generated_files(self, client, workspace):
        state = _state_done("files-check")
        _write_task_board(workspace, "files-check", state)
        r = client.get("/projects/files-check/task_board")
        body = r.json()
        assert body["generated_files_count"] == 2
        assert sorted(body["generated_files"]) == ["main.py", "routes.py"]

    def test_board_contains_status_report(self, client, workspace):
        state = _state_done("sr-check")
        _write_task_board(workspace, "sr-check", state)
        r = client.get("/projects/sr-check/task_board")
        body = r.json()
        assert "status_report" in body
        assert body["status_report"]["status"] == "completed"


# ── GET /projects ────────────────────────────────────────────────────────────────

class TestListProjects:
    def test_empty_workspace_returns_empty_list(self, client, workspace):
        r = client.get("/projects")
        assert r.status_code == 200
        body = r.json()
        assert body["projects"] == []
        assert body["total"] == 0

    def test_lists_projects_with_task_boards(self, client, workspace):
        _write_task_board(workspace, "alpha", _state_done("alpha"))
        _write_task_board(workspace, "beta",  _state_done("beta"))
        r = client.get("/projects")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        ids = {p["project_id"] for p in body["projects"]}
        assert ids == {"alpha", "beta"}

    def test_project_without_task_board_not_listed(self, client, workspace):
        # Create a workspace dir with no task_board.json
        (workspace / "ghost-proj").mkdir()
        _write_task_board(workspace, "real-proj", _state_done("real-proj"))
        r = client.get("/projects")
        body = r.json()
        ids = [p["project_id"] for p in body["projects"]]
        assert "ghost-proj" not in ids
        assert "real-proj" in ids

    def test_summary_fields_present(self, client, workspace):
        _write_task_board(workspace, "field-check", _state_done("field-check"))
        r = client.get("/projects")
        p = r.json()["projects"][0]
        assert p["project_id"] == "field-check"
        assert p["idea"] == "Build a REST API"
        assert p["status"] == "completed"
        assert p["tasks_total"] == 2
        assert p["tasks_completed"] == 2
        assert p["files_generated_count"] == 2
        assert "persisted_at" in p
        assert p["is_active"] is False

    def test_sorted_most_recent_first(self, client, workspace):
        # Write two boards with different persisted_at times embedded
        import time
        _write_task_board(workspace, "older", _state_done("older"))
        time.sleep(0.01)
        _write_task_board(workspace, "newer", _state_done("newer"))
        r = client.get("/projects")
        ids = [p["project_id"] for p in r.json()["projects"]]
        assert ids[0] == "newer"

    def test_is_active_true_when_running(self, client, workspace):
        _write_task_board(workspace, "live-proj", _state_done("live-proj"))
        with patch("ai_cto.api.routes.projects.is_running", return_value=True):
            r = client.get("/projects")
        body = r.json()
        proj = next(p for p in body["projects"] if p["project_id"] == "live-proj")
        assert proj["is_active"] is True
        assert proj["status"] == "in_progress"

    def test_in_progress_status_when_running(self, client, workspace):
        # Even if disk says "completed", is_running overrides status to in_progress
        _write_task_board(workspace, "run-proj", _state_done("run-proj"))
        with patch("ai_cto.api.routes.projects.is_running", return_value=True):
            r = client.get("/projects")
        p = next(x for x in r.json()["projects"] if x["project_id"] == "run-proj")
        assert p["status"] == "in_progress"

    def test_corrupt_board_skipped_silently(self, client, workspace):
        # Write a valid board and a corrupt one
        _write_task_board(workspace, "good-proj", _state_done("good-proj"))
        bad_dir = workspace / "bad-proj"
        bad_dir.mkdir()
        (bad_dir / "task_board.json").write_text("NOT JSON", encoding="utf-8")
        r = client.get("/projects")
        ids = [p["project_id"] for p in r.json()["projects"]]
        assert "good-proj" in ids
        assert "bad-proj" not in ids


# ── DELETE /projects/{id} ────────────────────────────────────────────────────────

class TestCancelProject:
    def test_cancel_running_project_returns_200(self, client, workspace):
        _write_task_board(workspace, "to-cancel", _state_done("to-cancel"))
        with patch("ai_cto.api.routes.projects.cancel_project_run", return_value=True):
            r = client.delete("/projects/to-cancel")
        assert r.status_code == 200
        body = r.json()
        assert body["cancelled"] is True
        assert "to-cancel" in body["message"]

    def test_cancel_nonexistent_project_returns_404(self, client, workspace):
        r = client.delete("/projects/does-not-exist")
        assert r.status_code == 404

    def test_cancel_completed_project_returns_409(self, client, workspace):
        _write_task_board(workspace, "done-proj", _state_done("done-proj"))
        with patch("ai_cto.api.routes.projects.cancel_project_run", return_value=False), \
             patch("ai_cto.api.routes.projects.get_run_state",
                   return_value=__import__("ai_cto.api.runner", fromlist=["RunState"]).RunState.COMPLETE):
            r = client.delete("/projects/done-proj")
        assert r.status_code == 409
        assert "already completed" in r.json()["detail"]

    def test_cancel_already_cancelled_returns_409(self, client, workspace):
        from ai_cto.api.runner import RunState
        _write_task_board(workspace, "cancelled-proj", _state_done("cancelled-proj"))
        with patch("ai_cto.api.routes.projects.cancel_project_run", return_value=False), \
             patch("ai_cto.api.routes.projects.get_run_state", return_value=RunState.CANCELLED):
            r = client.delete("/projects/cancelled-proj")
        assert r.status_code == 409
        assert "already cancelled" in r.json()["detail"]

    def test_cancel_post_restart_reads_disk_status(self, client, workspace):
        # Simulate a process restart: project not in _RUNS, but board exists on disk
        _write_task_board(workspace, "restarted", _state_done("restarted"))
        with patch("ai_cto.api.routes.projects.cancel_project_run", return_value=False), \
             patch("ai_cto.api.routes.projects.get_run_state", return_value=None):
            r = client.delete("/projects/restarted")
        assert r.status_code == 409
        assert "not running" in r.json()["detail"]
        # disk status should be embedded: "completed" from the state_done fixture
        assert "completed" in r.json()["detail"]

    def test_cancel_failed_project_returns_409(self, client, workspace):
        from ai_cto.api.runner import RunState
        _write_task_board(workspace, "failed-proj", _state_done("failed-proj"))
        with patch("ai_cto.api.routes.projects.cancel_project_run", return_value=False), \
             patch("ai_cto.api.routes.projects.get_run_state", return_value=RunState.FAILED):
            r = client.delete("/projects/failed-proj")
        assert r.status_code == 409
        assert "already failed" in r.json()["detail"]
