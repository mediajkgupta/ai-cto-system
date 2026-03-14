"""
test_resume.py — Tests for resume-from-task_board.json functionality.

Covers:
  - load_task_board: reads JSON, validates required keys
  - resume_state_from_board: reconstructs ProjectState from board
  - in_progress task reset to pending on resume
  - retries reset on interrupted task
  - generated files scanned from workspace
  - ValueError when project is complete or blocked
  - build_resume_graph: entry point is 'coding', full pipeline runs
"""

import json
import os
import pytest
from pathlib import Path

os.environ["AI_CTO_MOCK_LLM"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_board(tmp_path: Path, board: dict) -> Path:
    project_dir = tmp_path / board["project_id"]
    project_dir.mkdir(parents=True, exist_ok=True)
    board_path = project_dir / "task_board.json"
    board_path.write_text(json.dumps(board, indent=2), encoding="utf-8")
    return board_path


def _make_task(id, title, status="pending", depends_on=None, retries=0, max_retries=3):
    from ai_cto.state import make_task_item, now_iso
    t = make_task_item(id=id, title=title, description="", depends_on=depends_on or [])
    return {**t, "status": status, "retries": retries, "max_retries": max_retries}


def _minimal_board(project_id="test-proj", project_status="active", tasks=None):
    return {
        "project_id": project_id,
        "project_goal": "Build something",
        "project_status": project_status,
        "active_task_id": 1,
        "tasks": tasks or [_make_task(1, "Task one", status="pending")],
        "completed_task_ids": [],
        "failed_task_ids": [],
        "blocked_task_ids": [],
        "task_history": [],
    }


# ── load_task_board ────────────────────────────────────────────────────────────

class TestLoadTaskBoard:
    def test_reads_valid_board(self, tmp_path):
        from ai_cto.tools.task_board import load_task_board
        board_path = _write_board(tmp_path, _minimal_board())
        board = load_task_board(board_path)
        assert board["project_id"] == "test-proj"

    def test_raises_if_file_missing(self, tmp_path):
        from ai_cto.tools.task_board import load_task_board
        with pytest.raises(FileNotFoundError):
            load_task_board(tmp_path / "no-such-project" / "task_board.json")

    def test_raises_if_missing_project_id(self, tmp_path):
        from ai_cto.tools.task_board import load_task_board
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"project_goal": "x", "tasks": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="project_id"):
            load_task_board(p)

    def test_raises_if_missing_tasks(self, tmp_path):
        from ai_cto.tools.task_board import load_task_board
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"project_id": "x", "project_goal": "y"}), encoding="utf-8")
        with pytest.raises(ValueError, match="tasks"):
            load_task_board(p)


# ── resume_state_from_board ────────────────────────────────────────────────────

class TestResumeStateFromBoard:
    def test_returns_project_state(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        board = _minimal_board()
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert state["project_id"] == "test-proj"
        assert state["idea"] == "Build something"
        assert state["status"] == "coding"
        assert state["project_status"] == "active"

    def test_in_progress_task_reset_to_pending(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        tasks = [_make_task(1, "Auth module", status="in_progress", retries=1)]
        board = _minimal_board(tasks=tasks)
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        t = next(t for t in state["tasks"] if t["id"] == 1)
        assert t["status"] == "pending"

    def test_in_progress_retries_reset_to_zero(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        tasks = [_make_task(1, "Auth module", status="in_progress", retries=2)]
        board = _minimal_board(tasks=tasks)
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        t = next(t for t in state["tasks"] if t["id"] == 1)
        assert t["retries"] == 0

    def test_completed_ids_preserved(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        tasks = [
            _make_task(1, "Done task", status="done"),
            _make_task(2, "Next task", status="pending", depends_on=[1]),
        ]
        board = _minimal_board(tasks=tasks)
        board["completed_task_ids"] = [1]
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert 1 in state["completed_task_ids"]
        assert state["active_task_id"] == 2

    def test_next_runnable_task_selected(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        tasks = [
            _make_task(1, "Done task", status="done"),
            _make_task(2, "Next task", status="pending", depends_on=[1]),
            _make_task(3, "Later task", status="pending", depends_on=[2]),
        ]
        board = _minimal_board(tasks=tasks)
        board["completed_task_ids"] = [1]
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert state["active_task_id"] == 2  # not 3 — depends on 2

    def test_task_history_preserved_and_extended(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        board = _minimal_board()
        board["task_history"] = [
            {"task_id": 1, "event": "started", "detail": "", "timestamp": "2026-01-01T00:00:00"}
        ]
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        # Original event preserved + new "started (resumed)" event added
        events = [e["event"] for e in state["task_history"]]
        assert events.count("started") == 2
        details = [e["detail"] for e in state["task_history"]]
        assert "(resumed)" in details[-1]

    def test_raises_if_complete(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        tasks = [_make_task(1, "Done task", status="done")]
        board = _minimal_board(project_status="complete", tasks=tasks)
        board["completed_task_ids"] = [1]
        board_path = _write_board(tmp_path, board)
        with pytest.raises(ValueError, match="already complete"):
            resume_state_from_board(board_path)

    def test_raises_if_blocked(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        # T1 done, T2 depends on T3, T3 depends on T4 (doesn't exist)
        tasks = [
            _make_task(1, "Done", status="done"),
            _make_task(2, "Blocked A", status="pending", depends_on=[3]),
            _make_task(3, "Blocked B", status="pending", depends_on=[4]),
        ]
        board = _minimal_board(tasks=tasks)
        board["completed_task_ids"] = [1]
        board_path = _write_board(tmp_path, board)
        with pytest.raises(ValueError, match="blocked"):
            resume_state_from_board(board_path)

    def test_scans_generated_files_from_workspace(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        # Write a file to the project workspace
        project_dir = tmp_path / "test-proj"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        board = _minimal_board()
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert "main.py" in state["generated_files"]
        assert state["generated_files"]["main.py"] == "print('hello')"

    def test_task_board_json_not_in_generated_files(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.tools.task_board import resume_state_from_board
        board = _minimal_board()
        board_path = _write_board(tmp_path, board)
        state = resume_state_from_board(board_path)
        assert "task_board.json" not in state["generated_files"]


# ── build_resume_graph integration ────────────────────────────────────────────

class TestBuildResumeGraph:
    def test_resume_graph_completes_remaining_tasks(self, tmp_path):
        """Full integration: resume from a 2-task board where task 1 is done."""
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        os.environ["AI_CTO_MOCK_EXTENDED"] = "true"
        try:
            from ai_cto.tools.task_board import resume_state_from_board
            from ai_cto.graph import build_resume_graph
            from ai_cto.state import make_task_item, now_iso

            # Simulate a board where tasks 1 and 2 are done, 3 and 4 are pending
            tasks = [
                {**make_task_item(1, "Scaffold project structure", "", priority=1, depends_on=[]), "status": "done"},
                {**make_task_item(2, "Implement in-memory data store", "", priority=2, depends_on=[1]), "status": "done"},
                {**make_task_item(3, "Implement CRUD operations", "", priority=3, depends_on=[2]), "status": "pending"},
                {**make_task_item(4, "Add main entry point and integration test", "", priority=4, depends_on=[3]), "status": "pending"},
            ]
            board = {
                "project_id": "resume-test",
                "project_goal": "Build a SaaS CRM with contacts and deals",
                "project_status": "active",
                "active_task_id": 3,
                "tasks": tasks,
                "completed_task_ids": [1, 2],
                "failed_task_ids": [],
                "blocked_task_ids": [],
                "task_history": [],
            }
            board_path = _write_board(tmp_path, board)

            state = resume_state_from_board(board_path)
            assert state["active_task_id"] == 3

            graph = build_resume_graph()
            final = None
            for event in graph.stream(state):
                node = list(event.keys())[0]
                final = event[node]

            assert final["project_status"] == "complete"
            assert final["status"] == "done"
            assert set(final["completed_task_ids"]) == {1, 2, 3, 4}
        finally:
            os.environ.pop("AI_CTO_MOCK_EXTENDED", None)
