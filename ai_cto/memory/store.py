"""
store.py — ChromaDB-backed vector memory store.

Uses a local persistent Chroma database (no server required).
One collection per MemoryType keeps queries fast and avoids cross-type noise.

Storage location: <project_root>/.chroma_db/
Embeddings: Chroma's built-in default (all-MiniLM-L6-v2 via ONNX runtime).
            No API key or external service required.

Usage:
    store = ChromaMemoryStore()
    store.add(MemoryEntry(id=..., type=MemoryType.ARCHITECTURE, content=..., metadata=...))
    results = store.search("REST API with FastAPI", memory_type=MemoryType.ARCHITECTURE, n=3)
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from ai_cto.memory.schema import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)

# Persistent DB stored at project root / .chroma_db
_DB_PATH = Path(__file__).parent.parent.parent / ".chroma_db"


class ChromaMemoryStore:
    """
    Thin wrapper around ChromaDB providing add/search/clear operations.

    Each MemoryType maps to its own Chroma collection, preventing
    irrelevant cross-type results from polluting searches.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = str(db_path or _DB_PATH)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("[MemoryStore] Initialised Chroma at: %s", path)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, entry: MemoryEntry) -> None:
        """
        Store a MemoryEntry in the appropriate collection.

        If an entry with the same id already exists it is silently overwritten.
        """
        collection = self._collection(entry["type"])
        collection.upsert(
            ids=[entry["id"]],
            documents=[entry["content"]],
            metadatas=[entry["metadata"]],
        )
        logger.debug(
            "[MemoryStore] Saved %s entry (id=%s, %d chars)",
            entry["type"],
            entry["id"],
            len(entry["content"]),
        )

    def add_many(self, entries: list[MemoryEntry]) -> None:
        """Bulk-insert a list of MemoryEntry objects."""
        for entry in entries:
            self.add(entry)

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        memory_type: MemoryType,
        n: int = 3,
        exclude_mock: bool = False,
    ) -> list[MemoryEntry]:
        """
        Semantic search within a single memory type collection.

        Args:
            query        — Natural language query string.
            memory_type  — Which collection to search.
            n            — Maximum number of results to return.
            exclude_mock — When True, skip entries tagged run_mode='mock'.
                           Fetches extra candidates then post-filters in Python
                           so legacy entries (no run_mode key) are preserved.

        Returns:
            List of MemoryEntry objects, best match first.
            Returns [] if the collection is empty or no results found.
        """
        collection = self._collection(memory_type)

        # Guard: empty collection — Chroma raises on query with n > count
        count = collection.count()
        if count == 0:
            logger.debug("[MemoryStore] Collection '%s' is empty.", memory_type)
            return []

        # Fetch extra candidates when filtering so we can still return n results
        # after mock entries are removed.
        fetch_n = min(n * 4, count) if exclude_mock else min(n, count)
        results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas"],
        )

        entries: list[MemoryEntry] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for entry_id, doc, meta in zip(ids, docs, metas):
            if exclude_mock and meta.get("run_mode") == "mock":
                continue
            entries.append(
                MemoryEntry(
                    id=entry_id,
                    type=memory_type,
                    content=doc,
                    metadata=meta,
                )
            )

        entries = entries[:n]  # cap at requested n after filtering

        logger.debug(
            "[MemoryStore] Search '%s' in %s → %d result(s) (exclude_mock=%s).",
            query[:60],
            memory_type,
            len(entries),
            exclude_mock,
        )
        return entries

    # ── Utility ────────────────────────────────────────────────────────────────

    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """
        Delete all entries from one collection (or all collections).

        Primarily used in tests.
        """
        types = [memory_type] if memory_type else list(MemoryType)
        for mt in types:
            name = _collection_name(mt)
            try:
                self._client.delete_collection(name)
                logger.debug("[MemoryStore] Cleared collection: %s", name)
            except Exception:  # noqa: BLE001
                pass  # collection may not exist yet

    def count(self, memory_type: MemoryType) -> int:
        """Return the number of entries in a collection."""
        return self._collection(memory_type).count()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _collection(self, memory_type: MemoryType):
        """Get-or-create the Chroma collection for a MemoryType."""
        return self._client.get_or_create_collection(
            name=_collection_name(memory_type),
            metadata={"hnsw:space": "cosine"},
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_id() -> str:
    """Generate a new random UUID string for a MemoryEntry."""
    return str(uuid.uuid4())


def _collection_name(memory_type: MemoryType) -> str:
    return f"ai_cto_{memory_type.value}"
