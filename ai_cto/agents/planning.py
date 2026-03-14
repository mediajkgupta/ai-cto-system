"""
planning.py — Planning Agent

Responsibilities:
  - Receive the raw user idea from ProjectState
  - Call Claude to produce a Markdown architecture document
  - Write the architecture back into ProjectState

The Planning Agent runs exactly ONCE per project (at the start).
Task decomposition is handled by the TaskManagerAgent which runs next.
"""

import json
import re
import logging
from anthropic import Anthropic
from ai_cto.state import ProjectState
from ai_cto.mock_llm import is_mock_mode, mock_planning_node

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software architect and AI CTO.
Given a software idea, write a concise architecture document in Markdown (max 400 words).

Return your response as valid JSON with this exact shape:
{
  "architecture": "<markdown string>"
}

Rules:
- Architecture must cover: components, data flow, tech stack choices.
- Be concrete and specific — avoid generic boilerplate.
- Do not include a task list; that will be handled separately.
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def planning_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Planning Agent.

    Transforms state:
        state.idea  →  state.architecture + state.status
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

    json_str = _extract_json(raw)
    parsed = json.loads(json_str)

    architecture: str = parsed["architecture"]

    logger.info("[PlanningAgent] Architecture ready (%d chars).", len(architecture))

    return {
        **state,
        "architecture": architecture,
        "status": "task_managing",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown code fences and return the raw JSON string."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text
