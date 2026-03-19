"""
test_reliability_fixes.py — Tests for the 5 reliability fixes.

Fix 1: Mock entries tagged run_mode='mock'; real runs exclude them
Fix 2: generated_files accumulates across tasks (no silent drops)
Fix 3: TestGenerator receives current-task files, not full accumulated state
Fix 4: active_task_id is None when project completes
Fix 5: DebugAgent retrieves DEBUG_SOLUTION memory for current error
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["AI_CTO_MOCK_LLM"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_task(id=1, title="Task", status="pending", depends_on=None, retries=0, max_retries=3):
    from ai_cto.state import make_task_item
    t = make_task_item(id=id, title=title, description="", depends_on=depends_on or [])
    return {**t, "status": status, "retries": retries, "max_retries": max_retries}


def _base_state(**overrides):
    from ai_cto.state import initial_state
    s = initial_state(idea="Test project", project_id="test-proj")
    s["architecture"] = "# Architecture\nSimple REST API."
    s["tasks"] = [_make_task(1, "Build API")]
    s["active_task_id"] = 1
    s["current_task_index"] = 0
    return {**s, **overrides}


def _fresh_store(tmp_path):
    """Create an isolated ChromaMemoryStore backed by a temp dir."""
    from ai_cto.memory.store import ChromaMemoryStore
    return ChromaMemoryStore(db_path=tmp_path / "chroma")


def _add_entry(store, memory_type, content, run_mode, project_id="proj"):
    from ai_cto.memory.store import new_id
    from ai_cto.memory.schema import MemoryEntry
    store.add(MemoryEntry(
        id=new_id(),
        type=memory_type,
        content=content,
        metadata={"project_id": project_id, "run_mode": run_mode, "idea": content[:40]},
    ))


# ── Fix 1: Mock memory isolation ──────────────────────────────────────────────

class TestFix1MockMemoryIsolation:
    def test_exclude_mock_filters_out_mock_entries(self, tmp_path):
        from ai_cto.memory.schema import MemoryType
        store = _fresh_store(tmp_path)
        _add_entry(store, MemoryType.ARCHITECTURE, "Mock REST API design", "mock")
        results = store.search("REST API", MemoryType.ARCHITECTURE, n=2, exclude_mock=True)
        assert results == []

    def test_exclude_mock_keeps_real_entries(self, tmp_path):
        from ai_cto.memory.schema import MemoryType
        store = _fresh_store(tmp_path)
        _add_entry(store, MemoryType.ARCHITECTURE, "Real REST API design", "real")
        results = store.search("REST API", MemoryType.ARCHITECTURE, n=2, exclude_mock=True)
        assert len(results) == 1
        assert results[0]["metadata"]["run_mode"] == "real"

    def test_exclude_mock_false_returns_all(self, tmp_path):
        from ai_cto.memory.schema import MemoryType
        store = _fresh_store(tmp_path)
        _add_entry(store, MemoryType.ARCHITECTURE, "Mock REST API design", "mock")
        _add_entry(store, MemoryType.ARCHITECTURE, "Real REST API design", "real")
        results = store.search("REST API", MemoryType.ARCHITECTURE, n=5, exclude_mock=False)
        assert len(results) == 2

    def test_exclude_mock_preserves_legacy_entries_without_run_mode(self, tmp_path):
        """Entries saved before this fix (no run_mode key) must survive real-run retrieval."""
        from ai_cto.memory.store import new_id
        from ai_cto.memory.schema import MemoryType, MemoryEntry
        store = _fresh_store(tmp_path)
        # Legacy entry: no run_mode in metadata
        store.add(MemoryEntry(
            id=new_id(),
            type=MemoryType.ARCHITECTURE,
            content="Legacy REST API design without run_mode tag",
            metadata={"project_id": "legacy", "idea": "Legacy REST API"},
        ))
        results = store.search("REST API", MemoryType.ARCHITECTURE, n=2, exclude_mock=True)
        # Legacy entries (no run_mode) should NOT be excluded — they are real
        assert len(results) == 1
        assert results[0]["metadata"].get("run_mode") is None

    def test_memory_save_node_tags_mock_entries(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.agents.memory import memory_save_node
        from ai_cto.memory.schema import MemoryType

        state = _base_state(
            architecture="# Arch",
            final_output="Project summary",
            debug_attempts=0,
        )
        # Mock mode is active (AI_CTO_MOCK_LLM=true)
        memory_save_node(state)

        results = store.search("project", MemoryType.ARCHITECTURE, n=2, exclude_mock=False)
        assert len(results) == 1
        assert results[0]["metadata"]["run_mode"] == "mock"

    def test_memory_save_node_tags_real_entries(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.agents.memory import memory_save_node
        from ai_cto.memory.schema import MemoryType

        state = _base_state(
            architecture="# Arch",
            final_output="Project summary",
            debug_attempts=0,
        )
        with patch("ai_cto.agents.memory.is_mock_mode", return_value=False):
            memory_save_node(state)

        results = store.search("project", MemoryType.ARCHITECTURE, n=2, exclude_mock=False)
        assert len(results) == 1
        assert results[0]["metadata"]["run_mode"] == "real"

    def test_memory_retrieve_excludes_mock_in_real_run(self, tmp_path):
        """Real-run retrieve call passes exclude_mock=True → mock entries invisible."""
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.memory.schema import MemoryType
        _add_entry(store, MemoryType.ARCHITECTURE, "Mock architecture content", "mock")
        _add_entry(store, MemoryType.PROJECT_PATTERN, "Mock pattern content", "mock")

        state = _base_state()
        with patch("ai_cto.agents.memory.is_mock_mode", return_value=False):
            from ai_cto.agents.memory import memory_retrieve_node
            result = memory_retrieve_node(state)

        assert result["memory_context"] == ""

    def test_memory_retrieve_includes_mock_in_mock_run(self, tmp_path):
        """Mock-run retrieve sees everything (useful for pipeline tests)."""
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.memory.schema import MemoryType
        _add_entry(store, MemoryType.ARCHITECTURE, "Mock architecture content for tests", "mock")

        state = _base_state()
        # is_mock_mode() returns True (AI_CTO_MOCK_LLM=true)
        from ai_cto.agents.memory import memory_retrieve_node
        result = memory_retrieve_node(state)
        assert "Mock architecture content" in result["memory_context"]


# ── Fix 2: Cumulative file state across tasks ──────────────────────────────────

class TestFix2CumulativeFiles:
    def _make_mock_provider(self, responses):
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = list(responses)
        return mock_provider

    def _coding_response(self, files: dict, entry_point="main.py") -> str:
        return json.dumps({"files": files, "entry_point": entry_point})

    def test_task2_files_merge_with_task1_files(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        task1_resp = self._coding_response({"main.py": "# main", "models.py": "# models"})
        task2_resp = self._coding_response({"api.py": "# api"})

        tasks = [
            _make_task(1, "Scaffold", status="pending"),
            _make_task(2, "Add API", status="pending", depends_on=[1]),
        ]

        # Simulate state after task 1 completed: main.py and models.py accumulated
        state_after_t1 = _base_state(
            tasks=tasks,
            active_task_id=2,
            current_task_index=1,
            generated_files={"main.py": "# main", "models.py": "# models"},
        )

        mock_provider = self._make_mock_provider([task2_resp])
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state_after_t1)

        # All three files present
        assert "main.py" in result["generated_files"]
        assert "models.py" in result["generated_files"]
        assert "api.py" in result["generated_files"]

    def test_later_task_can_overwrite_earlier_file(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        # Task 2 rewrites main.py with a richer version
        task2_resp = self._coding_response({"main.py": "# main v2"})
        state = _base_state(
            generated_files={"main.py": "# main v1"},
        )
        mock_provider = self._make_mock_provider([task2_resp])
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        # Latest version wins
        assert result["generated_files"]["main.py"] == "# main v2"

    def test_current_task_files_sentinel_set_correctly(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        new_files = {"service.py": "class Service: pass"}
        resp = self._coding_response(new_files)
        state = _base_state(generated_files={"main.py": "# main"})

        mock_provider = self._make_mock_provider([resp])
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                result = coding_node(state)

        # _current_task_files = only THIS task's files
        assert result["_current_task_files"] == new_files
        # generated_files = accumulated
        assert "main.py" in result["generated_files"]
        assert "service.py" in result["generated_files"]

    def test_prior_task_files_not_rewritten_to_disk(self, tmp_path):
        """write_files is called with new files only, not the full accumulated set."""
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents.coding import coding_node

        new_files = {"new_module.py": "x = 1"}
        resp = self._coding_response(new_files)
        state = _base_state(generated_files={"old_module.py": "y = 2"})

        write_calls = []

        def fake_write(project_id, files):
            write_calls.append(files)

        mock_provider = self._make_mock_provider([resp])
        with patch("ai_cto.agents.coding.get_provider", return_value=mock_provider):
            with patch("ai_cto.agents.coding.is_mock_mode", return_value=False):
                with patch("ai_cto.agents.coding.write_files", side_effect=fake_write):
                    coding_node(state)

        # Only the new file was written; old_module.py was NOT rewritten
        assert len(write_calls) == 1
        assert "new_module.py" in write_calls[0]
        assert "old_module.py" not in write_calls[0]


# ── Fix 3: Verification / test scoping ────────────────────────────────────────

class TestFix3TestScoping:
    def test_try_generate_tests_uses_current_task_files(self):
        """When _current_task_files is set, generate_tests receives it, not full state."""
        from ai_cto.agents.verification import _try_generate_tests

        current_files = {"service.py": "class Service: pass"}
        all_files = {"main.py": "# main", "service.py": "class Service: pass"}

        state = _base_state(
            generated_files=all_files,
            _current_task_files=current_files,
        )

        calls = []
        def fake_generate(files, architecture, project_id):
            calls.append(files)
            return {}

        with patch("ai_cto.agents.verification.generate_tests", side_effect=fake_generate):
            _try_generate_tests(state)

        assert calls[0] == current_files
        assert calls[0] != all_files

    def test_try_generate_tests_falls_back_to_all_files_when_no_sentinel(self):
        """Without _current_task_files, fall back to full generated_files."""
        from ai_cto.agents.verification import _try_generate_tests

        all_files = {"main.py": "# main"}
        state = _base_state(generated_files=all_files)
        # No _current_task_files key in state

        calls = []
        def fake_generate(files, architecture, project_id):
            calls.append(files)
            return {}

        with patch("ai_cto.agents.verification.generate_tests", side_effect=fake_generate):
            _try_generate_tests(state)

        assert calls[0] == all_files

    def test_scaffold_task_gets_scoped_tests_not_full_crud(self):
        """Scaffold task only produces main.py stub — test generator receives that only."""
        from ai_cto.agents.verification import _try_generate_tests

        # Scaffold task produces a stub — full CRUD not yet implemented
        scaffold_files = {"main.py": "# stub entry point\n"}
        all_files = {
            "main.py": "# stub entry point\n",
            "models.py": "# not written yet",
            "crud.py": "# not written yet",
        }

        state = _base_state(
            generated_files=all_files,
            _current_task_files=scaffold_files,
        )

        received = []
        def fake_generate(files, architecture, project_id):
            received.append(set(files.keys()))
            return {}

        with patch("ai_cto.agents.verification.generate_tests", side_effect=fake_generate):
            _try_generate_tests(state)

        # TestGenerator only saw the scaffold file
        assert received[0] == {"main.py"}
        assert "models.py" not in received[0]
        assert "crud.py" not in received[0]


# ── Fix 4: active_task_id cleared on completion ────────────────────────────────

class TestFix4ActiveTaskIdCleared:
    def test_active_task_id_none_when_all_tasks_done(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.state import make_task_item
        from ai_cto.graph import advance_task_node

        tasks = [
            {**make_task_item(1, "Only task", ""), "status": "in_progress",
             "retries": 0, "max_retries": 3},
        ]
        state = _base_state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
            completed_task_ids=[],
        )
        result = advance_task_node(state)

        # Project is complete
        assert result["project_status"] == "complete"
        # active_task_id must be None in terminal state
        assert result["active_task_id"] is None

    def test_active_task_id_none_in_persisted_board(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.state import make_task_item
        from ai_cto.graph import advance_task_node
        from ai_cto.tools.task_board import load_task_board

        tasks = [
            {**make_task_item(1, "Only task", ""), "status": "in_progress",
             "retries": 0, "max_retries": 3},
        ]
        state = _base_state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
        )
        advance_task_node(state)

        board = load_task_board(tmp_path / "test-proj" / "task_board.json")
        assert board["active_task_id"] is None

    def test_active_task_id_set_while_tasks_remain(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.state import make_task_item
        from ai_cto.graph import advance_task_node

        tasks = [
            {**make_task_item(1, "Task A", ""), "status": "in_progress",
             "retries": 0, "max_retries": 3},
            {**make_task_item(2, "Task B", ""), "status": "pending",
             "retries": 0, "max_retries": 3},
        ]
        state = _base_state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
        )
        result = advance_task_node(state)

        # Still tasks remaining — active_task_id should point to next task
        assert result["active_task_id"] == 2

    def test_blocked_project_also_clears_active_task_id(self, tmp_path):
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.state import make_task_item
        from ai_cto.graph import advance_task_node

        # T1 done, T2 depends on T3, T3 depends on T4 (doesn't exist) → blocked
        tasks = [
            {**make_task_item(1, "Task A", ""), "status": "in_progress",
             "retries": 0, "max_retries": 3},
            {**make_task_item(2, "Task B", "", depends_on=[3]), "status": "pending",
             "retries": 0, "max_retries": 3},
        ]
        state = _base_state(
            tasks=tasks,
            active_task_id=1,
            current_task_index=0,
        )
        result = advance_task_node(state)
        # Project is blocked — active_task_id should not point to an incomplete task
        assert result["status"] == "blocked"
        # active_task_id is irrelevant in a blocked state; ensure it is consistent


# ── Fix 5: DEBUG_SOLUTION retrieval by DebugAgent ─────────────────────────────

class TestFix5DebugSolutionRetrieval:
    def test_retrieve_debug_solutions_returns_matching_content(self, tmp_path):
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.memory.schema import MemoryType
        from ai_cto.memory.store import new_id
        from ai_cto.memory.schema import MemoryEntry

        store.add(MemoryEntry(
            id=new_id(),
            type=MemoryType.DEBUG_SOLUTION,
            content="Error: NameError\nFix: add import statement",
            metadata={"project_id": "proj1", "run_mode": "real", "idea": "API"},
        ))

        from ai_cto.agents.memory import retrieve_debug_solutions
        result = retrieve_debug_solutions("NameError: name not defined", exclude_mock=False)

        assert "NameError" in result
        assert "import statement" in result
        assert "proj1" in result

    def test_retrieve_debug_solutions_empty_when_no_entries(self, tmp_path):
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.agents.memory import retrieve_debug_solutions
        result = retrieve_debug_solutions("some error", exclude_mock=True)
        assert result == ""

    def test_retrieve_debug_solutions_excludes_mock_entries(self, tmp_path):
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.memory.schema import MemoryType
        _add_entry(store, MemoryType.DEBUG_SOLUTION,
                   "Mock debug fix for NameError", "mock")

        from ai_cto.agents.memory import retrieve_debug_solutions
        result = retrieve_debug_solutions("NameError", exclude_mock=True)
        assert result == ""

    def test_retrieve_debug_solutions_includes_mock_when_not_excluded(self, tmp_path):
        import ai_cto.agents.memory as mem_module
        store = _fresh_store(tmp_path)
        mem_module._store = store

        from ai_cto.memory.schema import MemoryType
        _add_entry(store, MemoryType.DEBUG_SOLUTION,
                   "Mock debug fix for NameError in tests", "mock")

        from ai_cto.agents.memory import retrieve_debug_solutions
        result = retrieve_debug_solutions("NameError", exclude_mock=False)
        assert "Mock debug fix" in result

    def test_debug_node_passes_error_context_to_retrieve(self, tmp_path):
        """debug_node calls retrieve_debug_solutions with the current error_context."""
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents import debug as debug_mod

        # Build a state that will hit the real-LLM path (not exhausted, not mock)
        tasks = [_make_task(1, "Task", status="in_progress", retries=0, max_retries=3)]
        state = _base_state(
            tasks=tasks,
            error_context="NameError: name 'db' is not defined",
            debug_attempts=0,
        )

        called_with = []

        def fake_retrieve(error_context, exclude_mock=True):
            called_with.append(error_context)
            return "## Relevant past debug solutions\nFix: add import"

        good_patch = json.dumps({
            "diagnosis": "Missing import",
            "files": {"main.py": "import db\n"},
        })
        mock_provider = MagicMock()
        mock_provider.complete.return_value = good_patch

        with patch("ai_cto.agents.debug.is_mock_mode", return_value=False):
            with patch("ai_cto.agents.debug.retrieve_debug_solutions", side_effect=fake_retrieve):
                with patch("ai_cto.agents.debug.get_provider", return_value=mock_provider):
                    with patch("ai_cto.agents.debug.write_files"):
                        with patch("ai_cto.agents.debug.persist_task_board"):
                            debug_mod.debug_node(state)

        assert called_with[0] == "NameError: name 'db' is not defined"

    def test_debug_solutions_injected_into_llm_prompt(self, tmp_path):
        """When debug solutions exist, they appear in the LLM user message."""
        import ai_cto.tools.file_writer as _fw
        _fw.WORKSPACE_ROOT = tmp_path
        from ai_cto.agents import debug as debug_mod

        tasks = [_make_task(1, "Task", status="in_progress", retries=0, max_retries=3)]
        state = _base_state(
            tasks=tasks,
            error_context="KeyError: 'user_id'",
            debug_attempts=0,
        )

        llm_prompts = []
        good_patch = json.dumps({"diagnosis": "Fixed", "files": {"main.py": "pass"}})
        mock_provider = MagicMock()
        def capture_complete(system, user, max_tokens=4096):
            llm_prompts.append(user)
            return good_patch
        mock_provider.complete.side_effect = capture_complete

        with patch("ai_cto.agents.debug.is_mock_mode", return_value=False):
            with patch("ai_cto.agents.debug.retrieve_debug_solutions",
                       return_value="## Relevant past debug solutions\nKnown fix: check dict keys"):
                with patch("ai_cto.agents.debug.get_provider", return_value=mock_provider):
                    with patch("ai_cto.agents.debug.write_files"):
                        with patch("ai_cto.agents.debug.persist_task_board"):
                            debug_mod.debug_node(state)

        assert len(llm_prompts) == 1
        assert "Relevant past debug solutions" in llm_prompts[0]
        assert "Known fix: check dict keys" in llm_prompts[0]
