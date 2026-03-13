"""
planning.py — Planning Agent

Responsibilities:
  - Receive the raw user idea from ProjectState
  - Call Claude to produce a Markdown architecture document
  - Parse a structured task list from the response
  - Write both back into ProjectState

The Planning Agent runs exactly ONCE per project (at the start).
"""

import json
import re
import logging
from anthropic import Anthropic
from ai_cto.state import ProjectState, TaskItem
from ai_cto.mock_llm import is_mock_mode, mock_planning_node

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software architect and AI CTO.
Given a software idea, you must:
1. Write a concise architecture document (Markdown, max 400 words).
2. Break the work into 3-6 discrete, sequential implementation tasks.

Return your response as valid JSON with this exact shape:
{
  "architecture": "<markdown string>",
  "tasks": [
    {"id": 1, "title": "...", "description": "..."},
    ...
  ]
}

Rules:
- Tasks must be small enough that one can be implemented in a single file or module.
- Do not include testing or deployment tasks in v1.
- Architecture must cover: components, data flow, tech stack choices.
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def planning_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Planning Agent.

    Transforms state:
        state.idea  →  state.architecture + state.tasks + state.status
    """
    logger.info("[PlanningAgent] Starting architecture planning for: %s", state["idea"])

    if is_mock_mode():
        return mock_planning_node(state)

    client = Anthropic()

    # Prepend any retrieved memory context to enrich the planning prompt
    memory_context = state.get("memory_context", "")
    user_content = f"Software idea: {state['idea']}"
    if memory_context:
        user_content = f"{memory_context}\n\n---\n\n{user_content}"
        logger.info("[PlanningAgent] Injecting %d chars of memory context.", len(memory_context))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ]
    )

    raw = response.content[0].text
    logger.debug("[PlanningAgent] Raw response:\n%s", raw)

    # Parse JSON — strip markdown fences if present
    json_str = _extract_json(raw)
    parsed = json.loads(json_str)

    architecture: str = parsed["architecture"]
    tasks: list[TaskItem] = [
        TaskItem(
            id=t["id"],
            title=t["title"],
            description=t["description"],
            status="pending",
        )
        for t in parsed["tasks"]
    ]

    logger.info("[PlanningAgent] Produced %d tasks.", len(tasks))

    return {
        **state,
        "architecture": architecture,
        "tasks": tasks,
        "current_task_index": 0,
        "status": "coding",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown code fences and return the raw JSON string."""
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    # Fallback: find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text
