"""
memory.py — Memory Agent nodes.

Two LangGraph nodes:

  memory_retrieve_node
      Runs BEFORE the PlanningAgent.
      Queries the vector store for memories relevant to the current idea.
      Writes a formatted context string into state.memory_context.
      The PlanningAgent injects this context into its prompt.

  memory_save_node
      Runs AFTER all tasks complete successfully.
      Persists three types of knowledge for future projects:
        1. ARCHITECTURE  — the architecture document for this project
        2. PROJECT_PATTERN — a summary of what was built
        3. DEBUG_SOLUTION  — error→fix pairs if any debugging occurred

Neither node makes an LLM call — memory is retrieved and stored directly.
"""

import logging
from datetime import datetime, timezone

from ai_cto.state import ProjectState
from ai_cto.memory.schema import MemoryEntry, MemoryType
from ai_cto.memory.store import ChromaMemoryStore, new_id

logger = logging.getLogger(__name__)

# Shared store instance (one per process; Chroma is thread-safe)
_store: ChromaMemoryStore | None = None


def _get_store() -> ChromaMemoryStore:
    """Lazily initialise the shared ChromaMemoryStore."""
    global _store
    if _store is None:
        _store = ChromaMemoryStore()
    return _store


# ── Retrieve node ──────────────────────────────────────────────────────────────

def memory_retrieve_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: retrieve relevant memories before planning.

    Searches for:
      - Past architectures similar to the current idea
      - Completed project patterns similar to the current idea

    The results are formatted into a single `memory_context` string
    which the PlanningAgent appends to its prompt.
    """
    idea = state["idea"]
    logger.info("[MemoryAgent] Retrieving memories for: %s", idea)

    store = _get_store()
    sections: list[str] = []

    # Retrieve past architecture decisions
    arch_results = store.search(idea, MemoryType.ARCHITECTURE, n=2)
    if arch_results:
        section = "## Relevant past architectures\n"
        for r in arch_results:
            project = r["metadata"].get("project_id", "unknown")
            section += f"\n### Project: {project}\n{r['content']}\n"
        sections.append(section)

    # Retrieve completed project patterns
    pattern_results = store.search(idea, MemoryType.PROJECT_PATTERN, n=2)
    if pattern_results:
        section = "## Relevant past project summaries\n"
        for r in pattern_results:
            project = r["metadata"].get("project_id", "unknown")
            section += f"\n### Project: {project}\n{r['content']}\n"
        sections.append(section)

    if sections:
        memory_context = (
            "# Memory: relevant experience from past projects\n\n"
            + "\n\n".join(sections)
        )
        logger.info(
            "[MemoryAgent] Injecting %d memory section(s) (%d chars).",
            len(sections),
            len(memory_context),
        )
    else:
        memory_context = ""
        logger.info("[MemoryAgent] No relevant memories found.")

    return {**state, "memory_context": memory_context}


# ── Save node ──────────────────────────────────────────────────────────────────

def memory_save_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: persist learnings after successful project completion.

    Saves up to three entries:
      1. ARCHITECTURE  — the full architecture document
      2. PROJECT_PATTERN — the final_output summary
      3. DEBUG_SOLUTION  — if any debugging occurred, save error context +
                           diagnosis so future projects can recognise similar errors
    """
    store = _get_store()
    project_id = state["project_id"]
    now = datetime.now(timezone.utc).isoformat()
    entries: list[MemoryEntry] = []

    # 1. Architecture memory
    if state.get("architecture"):
        entries.append(MemoryEntry(
            id=new_id(),
            type=MemoryType.ARCHITECTURE,
            content=state["architecture"],
            metadata={
                "project_id": project_id,
                "idea": state["idea"],
                "timestamp": now,
            },
        ))

    # 2. Project pattern memory
    if state.get("final_output"):
        entries.append(MemoryEntry(
            id=new_id(),
            type=MemoryType.PROJECT_PATTERN,
            content=state["final_output"],
            metadata={
                "project_id": project_id,
                "idea": state["idea"],
                "timestamp": now,
            },
        ))

    # 3. Debug solution memory (only if debugging actually happened)
    last_diagnosis = state.get("last_debug_diagnosis", "")
    if state.get("debug_attempts", 0) > 0 and last_diagnosis:
        content = (
            f"Error encountered:\n{state.get('_last_error_context', 'unknown error')}\n\n"
            f"Diagnosis and fix:\n{last_diagnosis}"
        )
        entries.append(MemoryEntry(
            id=new_id(),
            type=MemoryType.DEBUG_SOLUTION,
            content=content,
            metadata={
                "project_id": project_id,
                "idea": state["idea"],
                "debug_attempts": str(state["debug_attempts"]),
                "timestamp": now,
            },
        ))

    store.add_many(entries)
    logger.info(
        "[MemoryAgent] Saved %d memory entry/entries for project '%s'.",
        len(entries),
        project_id,
    )

    return {**state, "status": "done"}
