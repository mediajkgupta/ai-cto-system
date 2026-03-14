"""
test_task_manager_llm.py — Tests for the real LLM-backed TaskManagerAgent.

All LLM calls are monkeypatched — no API credits consumed.

Coverage:
  - Task selection: active_task_id respects priority and dependency order
  - Dependency-aware planning: chain deps, completed deps unlock next task
  - No duplicate completed tasks: LLM duplicates silently dropped
  - Completed task metadata (retries, timestamps) never reset
  - Failed / blocked tasks also protected
  - Resume compatibility: new IDs continue from max existing ID
  - User message context: completed/failed tasks injected into prompt
  - Merge logic: protected tasks first, LLM tasks appended
  - Validation warnings: missing deps, duplicate IDs
  - Retry loop: succeeds on second attempt after bad JSON
  - Raises RuntimeError after all retries exhausted
  - Board persisted to disk after node completes
"""

import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from ai_cto.state import initial_state, make_task_item


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _llm_response(tasks: list) -> MagicMock:
    """Return a mock Anthropic Message containing tasks as JSON text."""
    msg = MagicMock()
    msg.content = [MagicMock()]
    msg.content[0].text = json.dumps({"tasks": tasks})
    return msg


def _mock_anthropic(monkeypatch, tasks: list) -> MagicMock:
    """Patch Anthropic() in task_manager so it returns a fixed task list."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _llm_response(tasks)
    monkeypatch.setattr("ai_cto.agents.task_manager.Anthropic", lambda: mock_client)
    monkeypatch.delenv("AI_CTO_MOCK_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    return mock_client


def _state(**overrides):
    s = initial_state("Build a CRM app", "crm-test")
    s.update(overrides)
    return s


def _task(id, title=None, status="pending", depends_on=None, priority=5, retries=0):
    t = make_task_item(
        id=id,
        title=title or f"Task {id}",
        description=f"Description for task {id}",
        depends_on=depends_on or [],
        priority=priority,
    )
    return {**t, "status": status, "retries": retries}


def _llm_task(id, title=None, priority=5, depends_on=None, outputs=None):
    return {
        "id": id,
        "title": title or f"LLM Task {id}",
        "description": f"LLM-generated description for task {id}",
        "priority": priority,
        "depends_on": depends_on or [],
        "outputs": outputs or [],
    }


# ── Task selection ───────────────────────────────────────────────────────────────

class TestTaskSelection:
    def test_active_task_is_first_with_no_deps(self, tmp_path, monkeypatch):
        """T1 (no deps, priority 1) should be selected over T2 (depends on T1)."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [
            _llm_task(1, "Foundation", priority=1, depends_on=[]),
            _llm_task(2, "Feature",    priority=2, depends_on=[1]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        assert result["active_task_id"] == 1

    def test_lowest_priority_number_wins(self, tmp_path, monkeypatch):
        """Priority 1 beats priority 3 when neither has deps."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [
            _llm_task(1, "Low prio",  priority=3, depends_on=[]),
            _llm_task(2, "High prio", priority=1, depends_on=[]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        assert result["active_task_id"] == 2

    def test_current_task_index_in_sync_with_active_id(self, tmp_path, monkeypatch):
        """tasks[current_task_index].id must equal active_task_id."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [
            _llm_task(1, priority=2, depends_on=[]),
            _llm_task(2, priority=1, depends_on=[]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        idx = result["current_task_index"]
        assert result["tasks"][idx]["id"] == result["active_task_id"]

    def test_status_set_to_coding(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1)])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        assert result["status"] == "coding"

    def test_started_history_event_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1)])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        events = [e["event"] for e in result["task_history"]]
        assert "started" in events
        assert result["task_history"][-1]["task_id"] == result["active_task_id"]


# ── Dependency-aware planning ────────────────────────────────────────────────────

class TestDependencyAwarePlanning:
    def test_chain_deps_selects_root_task(self, tmp_path, monkeypatch):
        """T1 -> T2 -> T3: T1 (no deps) must be active first."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [
            _llm_task(1, "Root", priority=1, depends_on=[]),
            _llm_task(2, "Mid",  priority=2, depends_on=[1]),
            _llm_task(3, "Leaf", priority=3, depends_on=[2]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        assert result["active_task_id"] == 1
        assert len(result["tasks"]) == 3

    def test_completed_dep_unlocks_dependent_task(self, tmp_path, monkeypatch):
        """T2 depends on T1 (already completed): T2 should become active."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = {**_task(1, status="done"), "title": "T1 Done"}
        _mock_anthropic(monkeypatch, [
            _llm_task(2, "T2", priority=2, depends_on=[1]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], completed_task_ids=[1])
        result = task_manager_node(state)
        assert result["active_task_id"] == 2

    def test_all_tasks_have_required_v2_fields(self, tmp_path, monkeypatch):
        """All tasks produced must have V2 schema fields."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1)])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        for t in result["tasks"]:
            for field in (
                "id", "title", "description", "status", "priority",
                "depends_on", "outputs", "error_context",
                "retries", "max_retries", "created_at", "updated_at",
            ):
                assert field in t, f"Task missing field: {field}"


# ── No duplicate completed tasks ─────────────────────────────────────────────────

class TestNoDuplicateCompletedTasks:
    def test_llm_duplicate_of_completed_task_dropped(self, tmp_path, monkeypatch):
        """LLM returns T1 (already done) + T2: merge keeps existing T1, adds T2."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = {**_task(1, title="Original Done Task", status="done")}
        _mock_anthropic(monkeypatch, [
            _llm_task(1, title="DUPLICATE — should be dropped"),
            _llm_task(2, title="New Task"),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], completed_task_ids=[1])
        result = task_manager_node(state)

        t1 = next(t for t in result["tasks"] if t["id"] == 1)
        assert t1["title"] == "Original Done Task"   # LLM title rejected
        assert t1["status"] == "done"                # status untouched
        # No duplicated id=1
        assert len([t for t in result["tasks"] if t["id"] == 1]) == 1

    def test_completed_task_retries_not_reset(self, tmp_path, monkeypatch):
        """Completed task metadata (retries, timestamps) must survive the merge."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = {**_task(1, status="done", retries=2)}
        _mock_anthropic(monkeypatch, [
            _llm_task(1, title="OVERWRITE ATTEMPT"),
            _llm_task(2, title="Real new work"),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], completed_task_ids=[1])
        result = task_manager_node(state)
        t1 = next(t for t in result["tasks"] if t["id"] == 1)
        assert t1["retries"] == 2      # never reset by LLM output

    def test_failed_task_also_protected(self, tmp_path, monkeypatch):
        """Failed tasks (exhausted retries) are protected just like completed ones."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = {
            **_task(1, status="failed", retries=3),
            "error_context": "NameError: jwt undefined",
        }
        _mock_anthropic(monkeypatch, [
            _llm_task(1, title="OVERWRITE ATTEMPT"),
            _llm_task(2, title="Replacement task"),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], failed_task_ids=[1])
        result = task_manager_node(state)
        t1 = next(t for t in result["tasks"] if t["id"] == 1)
        assert t1["error_context"] == "NameError: jwt undefined"
        assert t1["status"] == "failed"

    def test_total_task_count_correct_after_dedup(self, tmp_path, monkeypatch):
        """1 completed + LLM returns (1 dup + 2 new) = 3 total, not 4."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = _task(1, status="done")
        _mock_anthropic(monkeypatch, [
            _llm_task(1, title="DUP"),   # protected — dropped
            _llm_task(2, title="T2"),
            _llm_task(3, title="T3"),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], completed_task_ids=[1])
        result = task_manager_node(state)
        assert len(result["tasks"]) == 3   # T1 (protected) + T2 + T3

    def test_completed_ids_unchanged_in_output_state(self, tmp_path, monkeypatch):
        """task_manager_node must not modify completed_task_ids."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        existing_t1 = _task(1, status="done")
        _mock_anthropic(monkeypatch, [_llm_task(2)])
        from ai_cto.agents.task_manager import task_manager_node
        state = _state(tasks=[existing_t1], completed_task_ids=[1])
        result = task_manager_node(state)
        assert result["completed_task_ids"] == [1]


# ── Resume compatibility ──────────────────────────────────────────────────────────

class TestResumeCompatibility:
    def test_new_ids_continue_from_max_completed(self, tmp_path, monkeypatch):
        """LLM uses IDs 3, 4 (continuing after completed 1, 2) — no collision."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        t1 = {**_task(1, status="done"), "title": "Step 1"}
        t2 = {**_task(2, status="done"), "title": "Step 2"}
        _mock_anthropic(monkeypatch, [
            _llm_task(3, "Step 3", priority=3, depends_on=[2]),
            _llm_task(4, "Step 4", priority=4, depends_on=[3]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state(tasks=[t1, t2], completed_task_ids=[1, 2]))
        ids = sorted(t["id"] for t in result["tasks"])
        assert ids == [1, 2, 3, 4]

    def test_active_task_skips_completed_after_resume(self, tmp_path, monkeypatch):
        """After T1+T2 done, active_task_id must be T3, not T1 or T2."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        t1 = _task(1, status="done")
        t2 = _task(2, status="done")
        _mock_anthropic(monkeypatch, [
            _llm_task(3, "T3", priority=3, depends_on=[2]),
        ])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state(tasks=[t1, t2], completed_task_ids=[1, 2]))
        assert result["active_task_id"] == 3

    def test_prior_history_preserved(self, tmp_path, monkeypatch):
        """Existing task_history entries must be kept; new started event appended."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        from ai_cto.state import task_history_event
        prior = [task_history_event(1, "done")]
        _mock_anthropic(monkeypatch, [_llm_task(2)])
        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state(task_history=prior))
        assert len(result["task_history"]) >= 2
        assert result["task_history"][0]["event"] == "done"   # prior preserved


# ── User message context ──────────────────────────────────────────────────────────

class TestUserMessageContext:
    def test_completed_task_title_in_message(self):
        from ai_cto.agents.task_manager import _build_user_message
        t1 = {**_task(1, title="Auth Module", status="done")}
        state = _state(tasks=[t1], completed_task_ids=[1])
        msg = _build_user_message(state)
        assert "Auth Module" in msg

    def test_completed_ids_listed_as_taken(self):
        from ai_cto.agents.task_manager import _build_user_message
        t1 = {**_task(1, status="done")}
        state = _state(tasks=[t1], completed_task_ids=[1])
        msg = _build_user_message(state)
        assert "1" in msg            # ID 1 must appear
        assert "Completed" in msg or "completed" in msg

    def test_no_task_state_section_when_empty(self):
        from ai_cto.agents.task_manager import _build_user_message
        state = _state()   # no existing tasks
        msg = _build_user_message(state)
        assert "Current task state" not in msg
        assert "Completed tasks" not in msg

    def test_failed_task_shown_with_error_snippet(self):
        from ai_cto.agents.task_manager import _build_user_message
        t1 = {**_task(1, status="failed"), "error_context": "NameError on line 5"}
        state = _state(tasks=[t1], failed_task_ids=[1])
        msg = _build_user_message(state)
        assert "Failed" in msg or "failed" in msg
        assert "NameError on line 5" in msg

    def test_memory_context_prepended(self):
        from ai_cto.agents.task_manager import _build_user_message
        state = _state(memory_context="## Past Architecture\nsome useful context")
        msg = _build_user_message(state)
        assert msg.startswith("Past relevant")

    def test_architecture_always_included(self):
        from ai_cto.agents.task_manager import _build_user_message
        state = _state(architecture="## My Architecture\nComponents: X, Y")
        msg = _build_user_message(state)
        assert "## My Architecture" in msg


# ── Merge logic (unit) ────────────────────────────────────────────────────────────

class TestMergeLogic:
    def test_protected_tasks_come_first_in_order(self):
        from ai_cto.agents.task_manager import _merge_tasks
        existing = [_task(1, status="done"), _task(2, status="done")]
        llm = [_llm_task(3)]
        result = _merge_tasks(existing, llm, protected_ids={1, 2})
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert result[2]["id"] == 3

    def test_llm_task_with_protected_id_dropped(self):
        from ai_cto.agents.task_manager import _merge_tasks
        existing = [_task(1, title="Original", status="done")]
        llm = [
            _llm_task(1, title="OVERWRITE"),
            _llm_task(2, title="New"),
        ]
        result = _merge_tasks(existing, llm, protected_ids={1})
        assert len(result) == 2
        assert result[0]["title"] == "Original"   # not "OVERWRITE"
        assert result[1]["id"] == 2

    def test_no_existing_tasks_returns_all_llm_tasks(self):
        from ai_cto.agents.task_manager import _merge_tasks
        llm = [_llm_task(1), _llm_task(2)]
        result = _merge_tasks([], llm, protected_ids=set())
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_empty_llm_output_preserves_protected(self):
        from ai_cto.agents.task_manager import _merge_tasks
        existing = [_task(1, status="done")]
        result = _merge_tasks(existing, [], protected_ids={1})
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_new_llm_tasks_are_fresh_task_items(self):
        """LLM tasks should be converted to full TaskItems via make_task_item."""
        from ai_cto.agents.task_manager import _merge_tasks
        llm = [_llm_task(1)]
        result = _merge_tasks([], llm, protected_ids=set())
        t = result[0]
        assert "retries" in t and t["retries"] == 0
        assert "max_retries" in t
        assert "created_at" in t
        assert "status" in t and t["status"] == "pending"


# ── Validation warnings (no-raise) ───────────────────────────────────────────────

class TestValidation:
    def test_missing_dep_logs_warning(self, caplog):
        import logging
        from ai_cto.agents.task_manager import _validate_tasks
        tasks = [
            make_task_item(id=1, title="T1", description="d", depends_on=[99]),
        ]
        with caplog.at_level(logging.WARNING, logger="ai_cto.agents.task_manager"):
            _validate_tasks(tasks, protected_ids=set())
        assert any("99" in r.message for r in caplog.records)

    def test_duplicate_id_logs_warning(self, caplog):
        import logging
        from ai_cto.agents.task_manager import _validate_tasks
        t = make_task_item(id=1, title="T", description="d")
        with caplog.at_level(logging.WARNING, logger="ai_cto.agents.task_manager"):
            _validate_tasks([t, {**t}], protected_ids=set())
        assert any("Duplicate" in r.message for r in caplog.records)

    def test_valid_tasks_produce_no_warnings(self, caplog):
        import logging
        from ai_cto.agents.task_manager import _validate_tasks
        tasks = [
            make_task_item(id=1, title="T1", description="d", depends_on=[]),
            make_task_item(id=2, title="T2", description="d", depends_on=[1]),
        ]
        with caplog.at_level(logging.WARNING, logger="ai_cto.agents.task_manager"):
            _validate_tasks(tasks, protected_ids=set())
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0


# ── Retry on parse failure ────────────────────────────────────────────────────────

class TestRetryOnParseFailure:
    def test_succeeds_on_second_attempt_after_bad_json(self, tmp_path, monkeypatch):
        """First LLM response is garbage; second is valid JSON → should succeed."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        monkeypatch.delenv("AI_CTO_MOCK_LLM", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        bad = MagicMock()
        bad.content = [MagicMock()]
        bad.content[0].text = "not json at all !!!"

        good = _llm_response([_llm_task(1)])

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [bad, good]
        monkeypatch.setattr("ai_cto.agents.task_manager.Anthropic", lambda: mock_client)

        from ai_cto.agents.task_manager import task_manager_node
        result = task_manager_node(_state())
        assert result["active_task_id"] == 1
        assert mock_client.messages.create.call_count == 2

    def test_raises_runtime_error_after_all_retries(self, tmp_path, monkeypatch):
        """If every attempt returns bad JSON, RuntimeError is raised."""
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        monkeypatch.delenv("AI_CTO_MOCK_LLM", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        bad = MagicMock()
        bad.content = [MagicMock()]
        bad.content[0].text = "still bad json"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = bad
        monkeypatch.setattr("ai_cto.agents.task_manager.Anthropic", lambda: mock_client)

        from ai_cto.agents.task_manager import task_manager_node
        with pytest.raises(RuntimeError, match="failed to produce valid JSON"):
            task_manager_node(_state())


# ── Board persistence ─────────────────────────────────────────────────────────────

class TestBoardPersistence:
    def test_board_written_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1, "My Task")])
        from ai_cto.agents.task_manager import task_manager_node
        task_manager_node(_state())
        assert (tmp_path / "crm-test" / "task_board.json").exists()

    def test_board_contains_new_tasks(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1, "My Task")])
        from ai_cto.agents.task_manager import task_manager_node
        task_manager_node(_state())
        data = json.loads((tmp_path / "crm-test" / "task_board.json").read_text())
        titles = [t["title"] for t in data["tasks"]]
        assert "My Task" in titles

    def test_board_project_status_active(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_cto.tools.file_writer.WORKSPACE_ROOT", tmp_path)
        _mock_anthropic(monkeypatch, [_llm_task(1)])
        from ai_cto.agents.task_manager import task_manager_node
        task_manager_node(_state())
        data = json.loads((tmp_path / "crm-test" / "task_board.json").read_text())
        assert data["project_status"] == "active"
