"""
schema.py — Memory entry types for the AI CTO System.

Every piece of knowledge stored in the vector database is a MemoryEntry.
The MemoryType controls which collection it lives in and how it is retrieved.
"""

from enum import Enum
from typing import TypedDict


class MemoryType(str, Enum):
    """
    Categories of memory the system can store and retrieve.

    ARCHITECTURE    — Architecture decisions from past projects.
                      Retrieved by PlanningAgent to inform new designs.

    DEBUG_SOLUTION  — Error → diagnosis → fix mappings.
                      Retrieved by DebugAgent to recognise known errors.

    PROJECT_PATTERN — High-level summaries of completed projects.
                      Retrieved by PlanningAgent to avoid reinventing wheels.
    """
    ARCHITECTURE = "architecture"
    DEBUG_SOLUTION = "debug_solution"
    PROJECT_PATTERN = "project_pattern"


class MemoryEntry(TypedDict):
    """
    A single unit of stored knowledge.

    Fields:
        id          — Unique identifier (UUID).
        type        — MemoryType category.
        content     — The text stored and embedded (the thing being remembered).
        metadata    — Arbitrary key/value context stored alongside the embedding.
                      Examples: project_id, idea, error_snippet, timestamp.
    """
    id: str
    type: MemoryType
    content: str
    metadata: dict[str, str]
