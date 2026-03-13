"""Tests for memory/store.py — ChromaMemoryStore operations."""

import pytest
from pathlib import Path
from ai_cto.memory.schema import MemoryEntry, MemoryType
from ai_cto.memory.store import ChromaMemoryStore, new_id


@pytest.fixture
def store(tmp_path):
    """Isolated store using a temp directory — never touches .chroma_db/."""
    s = ChromaMemoryStore(db_path=tmp_path / "test_chroma")
    yield s
    s.clear()


# ── add / count ────────────────────────────────────────────────────────────────

def test_add_and_count(store):
    assert store.count(MemoryType.ARCHITECTURE) == 0

    store.add(MemoryEntry(
        id=new_id(),
        type=MemoryType.ARCHITECTURE,
        content="FastAPI + PostgreSQL architecture for a todo app.",
        metadata={"project_id": "todo-api", "idea": "todo app", "timestamp": "2026-01-01"},
    ))

    assert store.count(MemoryType.ARCHITECTURE) == 1


def test_add_many(store):
    entries = [
        MemoryEntry(
            id=new_id(),
            type=MemoryType.PROJECT_PATTERN,
            content=f"Pattern {i}",
            metadata={"project_id": f"proj-{i}", "idea": f"idea {i}", "timestamp": "2026-01-01"},
        )
        for i in range(3)
    ]
    store.add_many(entries)
    assert store.count(MemoryType.PROJECT_PATTERN) == 3


def test_upsert_same_id(store):
    """Adding the same id twice should upsert, not duplicate."""
    entry_id = new_id()
    for content in ("first content", "second content"):
        store.add(MemoryEntry(
            id=entry_id,
            type=MemoryType.DEBUG_SOLUTION,
            content=content,
            metadata={"project_id": "p", "idea": "i", "timestamp": "2026-01-01"},
        ))
    assert store.count(MemoryType.DEBUG_SOLUTION) == 1


# ── search ─────────────────────────────────────────────────────────────────────

def test_search_empty_collection_returns_empty(store):
    results = store.search("anything", MemoryType.ARCHITECTURE, n=3)
    assert results == []


def test_search_returns_results(store):
    store.add(MemoryEntry(
        id=new_id(),
        type=MemoryType.ARCHITECTURE,
        content="FastAPI REST API with JWT authentication and PostgreSQL.",
        metadata={"project_id": "auth-api", "idea": "auth api", "timestamp": "2026-01-01"},
    ))
    store.add(MemoryEntry(
        id=new_id(),
        type=MemoryType.ARCHITECTURE,
        content="React dashboard with chart visualisations and CSV export.",
        metadata={"project_id": "dashboard", "idea": "dashboard", "timestamp": "2026-01-01"},
    ))

    results = store.search("REST API authentication", MemoryType.ARCHITECTURE, n=1)
    assert len(results) == 1
    assert results[0]["type"] == MemoryType.ARCHITECTURE
    assert "content" in results[0]
    assert "metadata" in results[0]


def test_search_n_capped_at_collection_size(store):
    """Requesting n=10 when only 2 entries exist should not raise."""
    for i in range(2):
        store.add(MemoryEntry(
            id=new_id(),
            type=MemoryType.PROJECT_PATTERN,
            content=f"Project {i} summary",
            metadata={"project_id": f"p{i}", "idea": f"idea {i}", "timestamp": "2026-01-01"},
        ))
    results = store.search("project", MemoryType.PROJECT_PATTERN, n=10)
    assert len(results) == 2


# ── clear ──────────────────────────────────────────────────────────────────────

def test_clear_single_type(store):
    store.add(MemoryEntry(
        id=new_id(), type=MemoryType.ARCHITECTURE,
        content="arch", metadata={"project_id": "p", "idea": "i", "timestamp": "t"},
    ))
    store.add(MemoryEntry(
        id=new_id(), type=MemoryType.PROJECT_PATTERN,
        content="pattern", metadata={"project_id": "p", "idea": "i", "timestamp": "t"},
    ))

    store.clear(MemoryType.ARCHITECTURE)

    assert store.count(MemoryType.ARCHITECTURE) == 0
    assert store.count(MemoryType.PROJECT_PATTERN) == 1


def test_clear_all(store):
    for mt in MemoryType:
        store.add(MemoryEntry(
            id=new_id(), type=mt,
            content="some content", metadata={"project_id": "p", "idea": "i", "timestamp": "t"},
        ))
    store.clear()
    for mt in MemoryType:
        assert store.count(mt) == 0
