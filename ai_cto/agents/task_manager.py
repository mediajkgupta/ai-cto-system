"""
task_manager.py — Task Manager Agent (V2)

Responsibilities:
  - Receive the architecture document from PlanningAgent
  - Call Claude to decompose the project into a structured, ordered task list
  - Each task carries: id, title, description, priority, depends_on, outputs,
    retries, max_retries, created_at, updated_at
  - Set active_task_id to the first unblocked task (highest priority with no
    unmet dependencies)
  - Add a "started" event to task_history for the first active task
  - Persist the initial task board to workspace/<project_id>/task_board.json
  - Write tasks, active_task_id, and task_history back into ProjectState

Design decisions:
  - Tasks are dependency-aware: a task won't start until all tasks listed in
    its depends_on have status == "done"
  - Priority (1 = highest) orders tasks within the same dependency tier
  - The agent runs exactly ONCE per project, immediately after PlanningAgent
  - In mock mode it returns MOCK_TASKS without any API call
"""

import json
import re
import logging
from anthropic import Anthropic
from ai_cto.state import (
    ProjectState, make_task_item, get_next_runnable_task,
    task_history_event,
)
from ai_cto.tools.task_board import persist_task_board
from ai_cto.mock_llm import is_mock_mode, mock_task_manager_node

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software project manager and AI CTO.
Given a software project architecture document, decompose the work into a
structured, ordered task list for an autonomous coding agent.

Return your response as valid JSON with this exact shape:
{
  "tasks": [
    {
      "id": 1,
      "title": "short action title",
      "description": "detailed description of what to implement",
      "priority": 1,
      "depends_on": [],
      "outputs": ["main.py", "utils/helper.py"]
    },
    ...
  ]
}

Rules:
- Produce 2–6 tasks. Each task must be implementable in one coding session.
- priority is an integer where 1 = highest. Use 1–5.
- depends_on lists IDs of tasks that MUST complete before this task starts.
- outputs lists the relative file paths this task will produce.
- Tasks must collectively cover the full implementation (no gaps).
- Do not include testing, deployment, or documentation tasks.
- The first task (no depends_on) should set up the project foundation.
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def task_manager_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Task Manager Agent.

    Transforms state:
        state.architecture  →  state.tasks + state.active_task_id
                               + state.task_history + task_board.json on disk
    """
    logger.info(
        "[TaskManagerAgent] Building task list for project '%s'.",
        state["project_id"],
    )

    if is_mock_mode():
        return mock_task_manager_node(state)

    client = Anthropic()

    user_message = (
        f"Project idea: {state['idea']}\n\n"
        f"Architecture:\n{state['architecture']}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text
    logger.debug("[TaskManagerAgent] Raw response:\n%s", raw)

    json_str = _extract_json(raw)
    parsed = json.loads(json_str)

    tasks = [
        make_task_item(
            id=t["id"],
            title=t["title"],
            description=t["description"],
            priority=t.get("priority", 5),
            depends_on=t.get("depends_on", []),
            outputs=t.get("outputs", []),
        )
        for t in parsed["tasks"]
    ]

    first_task = get_next_runnable_task(tasks)
    active_id = first_task["id"] if first_task else None
    first_idx = next(
        (i for i, t in enumerate(tasks) if t["id"] == active_id), 0
    ) if active_id is not None else 0

    # Record that the first task is starting
    history = list(state.get("task_history", []))
    if active_id is not None:
        history.append(task_history_event(active_id, "started"))

    logger.info(
        "[TaskManagerAgent] Created %d task(s). First active task id=%s.",
        len(tasks),
        active_id,
    )

    new_state = {
        **state,
        "tasks": tasks,
        "active_task_id": active_id,
        "current_task_index": first_idx,
        "task_history": history,
        "status": "coding",
    }

    # Persist initial task board to disk
    persist_task_board(new_state)

    return new_state


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
