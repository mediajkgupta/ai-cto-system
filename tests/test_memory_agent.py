"""Tests for agents/memory.py — retrieve and save nodes."""

import pytest
from unittest.mock import patch, MagicMock
from ai_cto.state import initial_state
from ai_cto.memory.schema import MemoryEntry, MemoryType
from ai_cto.memory.store import new_id


def _base_state(**overrides):
    s = initial_state("Build a REST API", "rest-api")
    s.update(overrides)
    return s


# ── memory_retrieve_node ───────────────────────────────────────────────────────

class TestMemoryRetrieveNode:

    def test_empty_store_sets_empty_context(self, tmp_path):
        """When the store has no entries, memory_context should be empty."""
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_retrieve_node
            result = memory_retrieve_node(_base_state())

        assert result["memory_context"] == ""

    def test_populated_store_sets_context(self, tmp_path):
        """With relevant entries, memory_context should be non-empty."""
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        store.add(MemoryEntry(
            id=new_id(),
            type=MemoryType.ARCHITECTURE,
            content="FastAPI + PostgreSQL REST API architecture.",
            metadata={"project_id": "past-api", "idea": "rest api", "timestamp": "2026-01-01"},
        ))

        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_retrieve_node
            result = memory_retrieve_node(_base_state())

        assert len(result["memory_context"]) > 0
        assert "past-api" in result["memory_context"]

    def test_preserves_all_other_state_fields(self, tmp_path):
        """memory_retrieve_node must not overwrite unrelated state fields."""
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_retrieve_node
            state = _base_state(idea="My idea", project_id="my-proj")
            result = memory_retrieve_node(state)

        assert result["idea"] == "My idea"
        assert result["project_id"] == "my-proj"


# ── memory_save_node ───────────────────────────────────────────────────────────

class TestMemorySaveNode:

    def test_saves_architecture_and_pattern(self, tmp_path):
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_save_node
            state = _base_state(
                architecture="## Architecture\nFastAPI backend.",
                final_output="Project completed.\nFiles: main.py",
                debug_attempts=0,
            )
            memory_save_node(state)

        assert store.count(MemoryType.ARCHITECTURE) == 1
        assert store.count(MemoryType.PROJECT_PATTERN) == 1
        assert store.count(MemoryType.DEBUG_SOLUTION) == 0

    def test_saves_debug_solution_when_debugging_occurred(self, tmp_path):
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_save_node
            state = _base_state(
                architecture="## Architecture",
                final_output="Done.",
                debug_attempts=2,
                last_debug_diagnosis="Missing import: added `import os`.",
                _last_error_context="ModuleNotFoundError: No module named 'os'",
            )
            memory_save_node(state)

        assert store.count(MemoryType.DEBUG_SOLUTION) == 1

    def test_no_debug_solution_when_no_debugging(self, tmp_path):
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_save_node
            state = _base_state(
                architecture="## Architecture",
                final_output="Done.",
                debug_attempts=0,
                last_debug_diagnosis="",
            )
            memory_save_node(state)

        assert store.count(MemoryType.DEBUG_SOLUTION) == 0

    def test_returns_status_done(self, tmp_path):
        from ai_cto.memory.store import ChromaMemoryStore
        from ai_cto.agents import memory as memory_module

        store = ChromaMemoryStore(db_path=tmp_path / "chroma")
        with patch.object(memory_module, "_get_store", return_value=store):
            from ai_cto.agents.memory import memory_save_node
            result = memory_save_node(_base_state(architecture="a", final_output="f"))

        assert result["status"] == "done"
