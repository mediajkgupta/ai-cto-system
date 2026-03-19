"""
test_status.py — Unit tests for the machine-readable project status system.
"""

import pytest
from ai_cto.state import initial_state, make_task_item
from ai_cto.tools.status import (
    get_status_report,
    format_status_report,
    STATUS_PLANNED, STATUS_IN_PROGRESS, STATUS_COMPLETED, STATUS_FAILED, STATUS_BLOCKED,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _state_with_tasks(routing_status, project_status, task_specs):
    """task_specs: list of (task_status_internal, retries)"""
    state = initial_state("build a thing", "test-proj")
    state["status"] = routing_status
    state["project_status"] = project_status
    state["tasks"] = [
        {**make_task_item(i+1, f"Task {i+1}", "desc"),
         "status": spec[0], "retries": spec[1]}
        for i, spec in enumerate(task_specs)
    ]
    return state


# ── Project-level status mapping ────────────────────────────────────────────────

class TestProjectStatusMapping:
    def test_initial_state_is_planned(self):
        state = initial_state("idea", "proj")
        report = get_status_report(state)
        assert report["status"] == STATUS_PLANNED

    def test_coding_status_is_in_progress(self):
        state = initial_state("idea", "proj")
        state["status"] = "coding"
        state["project_status"] = "active"
        report = get_status_report(state)
        assert report["status"] == STATUS_IN_PROGRESS

    def test_done_routing_is_completed(self):
        state = initial_state("idea", "proj")
        state["status"] = "done"
        state["project_status"] = "complete"
        report = get_status_report(state)
        assert report["status"] == STATUS_COMPLETED

    def test_failed_routing_is_failed(self):
        state = initial_state("idea", "proj")
        state["status"] = "failed"
        state["project_status"] = "failed"
        report = get_status_report(state)
        assert report["status"] == STATUS_FAILED

    def test_blocked_routing_is_blocked(self):
        state = initial_state("idea", "proj")
        state["status"] = "blocked"
        state["project_status"] = "blocked"
        report = get_status_report(state)
        assert report["status"] == STATUS_BLOCKED

    def test_all_internal_routing_statuses_map_without_error(self):
        statuses = [
            "planning", "task_managing", "coding", "verifying", "verifying_failed",
            "executing", "debugging", "saving_memory", "done", "failed", "blocked",
        ]
        state = initial_state("idea", "proj")
        for s in statuses:
            state["status"] = s
            report = get_status_report(state)
            assert report["status"] in (
                STATUS_PLANNED, STATUS_IN_PROGRESS, STATUS_COMPLETED,
                STATUS_FAILED, STATUS_BLOCKED,
            ), f"Unexpected status for routing={s}: {report['status']}"


# ── Task-level status mapping ───────────────────────────────────────────────────

class TestTaskStatusMapping:
    def test_pending_maps_to_planned(self):
        state = _state_with_tasks("coding", "active", [("pending", 0)])
        report = get_status_report(state)
        assert report["tasks"][0]["status"] == STATUS_PLANNED

    def test_in_progress_maps_to_in_progress(self):
        state = _state_with_tasks("coding", "active", [("in_progress", 1)])
        report = get_status_report(state)
        assert report["tasks"][0]["status"] == STATUS_IN_PROGRESS

    def test_done_maps_to_completed(self):
        state = _state_with_tasks("done", "complete", [("done", 0)])
        report = get_status_report(state)
        assert report["tasks"][0]["status"] == STATUS_COMPLETED

    def test_failed_maps_to_failed(self):
        state = _state_with_tasks("failed", "failed", [("failed", 3)])
        report = get_status_report(state)
        assert report["tasks"][0]["status"] == STATUS_FAILED

    def test_blocked_maps_to_blocked(self):
        state = _state_with_tasks("blocked", "blocked", [("blocked", 0)])
        report = get_status_report(state)
        assert report["tasks"][0]["status"] == STATUS_BLOCKED

    def test_task_counts_are_accurate(self):
        state = _state_with_tasks("coding", "active", [
            ("done", 0), ("done", 0),
            ("failed", 3),
            ("pending", 0), ("pending", 0),
            ("blocked", 0),
        ])
        report = get_status_report(state)
        assert report["tasks_total"] == 6
        assert report["tasks_completed"] == 2
        assert report["tasks_failed"] == 1
        assert report["tasks_planned"] == 2
        assert report["tasks_blocked"] == 1


# ── Agent-level status ───────────────────────────────────────────────────────────

class TestAgentStatus:
    def test_agent_stage_matches_routing_status(self):
        state = initial_state("idea", "proj")
        state["status"] = "debugging"
        state["debug_attempts"] = 2
        state["active_task_id"] = 3
        report = get_status_report(state)
        assert report["agent"]["current_stage"] == "debugging"
        assert report["agent"]["pipeline_status"] == STATUS_IN_PROGRESS
        assert report["agent"]["active_task_id"] == 3
        assert report["agent"]["debug_attempts"] == 2

    def test_agent_execution_timeout_count(self):
        state = initial_state("idea", "proj")
        state["execution_timeout_count"] = 1
        report = get_status_report(state)
        assert report["agent"]["execution_timeout_count"] == 1


# ── Report fields ───────────────────────────────────────────────────────────────

class TestReportFields:
    def test_project_id_and_idea_present(self):
        state = initial_state("build an API", "my-api")
        report = get_status_report(state)
        assert report["project_id"] == "my-api"
        assert report["idea"] == "build an API"

    def test_files_generated_count(self):
        state = initial_state("idea", "proj")
        state["generated_files"] = {"a.py": "x", "b.py": "y"}
        report = get_status_report(state)
        assert report["files_generated_count"] == 2

    def test_generated_at_is_present(self):
        report = get_status_report(initial_state("idea", "proj"))
        assert "generated_at" in report
        assert "T" in report["generated_at"]   # ISO format


# ── format_status_report ────────────────────────────────────────────────────────

class TestFormatStatusReport:
    def test_output_is_string(self):
        state = initial_state("build a REST API", "rest-api")
        state["tasks"] = [make_task_item(1, "Foundation", "desc")]
        report = get_status_report(state)
        text = format_status_report(report)
        assert isinstance(text, str)
        assert "rest-api" in text
        assert "build a REST API" in text

    def test_output_includes_task_section(self):
        state = _state_with_tasks("coding", "active", [("done", 0), ("pending", 0)])
        report = get_status_report(state)
        text = format_status_report(report)
        assert "completed" in text
        assert "planned" in text

    def test_output_includes_agent_section(self):
        state = initial_state("idea", "proj")
        state["status"] = "debugging"
        report = get_status_report(state)
        text = format_status_report(report)
        assert "debugging" in text
