"""
coding.py — Coding Agent

Responsibilities:
  - Receive the current task from ProjectState
  - Call Claude with the task description + architecture context
  - Parse the response into a {filepath: content} dict
  - Write the dict into state.generated_files
  - Write files to disk via file_writer

The Coding Agent runs once per task (and again after each debug patch).
"""

import json
import re
import logging
from anthropic import Anthropic
from ai_cto.state import ProjectState
from ai_cto.tools.file_writer import write_files
from ai_cto.mock_llm import is_mock_mode, mock_coding_node

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software engineer.
Given a task description and project architecture, write the implementation code.

Return your response as valid JSON with this exact shape:
{
  "files": {
    "relative/path/to/file.py": "<full file contents>",
    ...
  },
  "entry_point": "relative/path/to/main_file.py"
}

Rules:
- File paths are relative to the project root.
- Each file must be complete and runnable — no placeholder comments like '# TODO'.
- Prefer simple, readable code over clever abstractions.
- Include only the files needed for this specific task.
- entry_point is the file the ExecutionAgent should run (e.g. 'main.py').
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def coding_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Coding Agent.

    Transforms state:
        state.tasks[current_task_index]  →  state.generated_files + disk files
    """
    task = state["tasks"][state["current_task_index"]]
    logger.info("[CodingAgent] Implementing task %d: %s", task["id"], task["title"])

    if is_mock_mode():
        return mock_coding_node(state)

    client = Anthropic()

    user_message = (
        f"Architecture:\n{state['architecture']}\n\n"
        f"Task to implement:\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
    )

    # If we're retrying after a debug patch, include the previous error
    if state.get("error_context"):
        user_message += (
            f"\nPrevious attempt failed with this error — fix it:\n"
            f"{state['error_context']}\n\n"
            f"Previously generated files for reference:\n"
            f"{json.dumps(state['generated_files'], indent=2)}"
        )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text
    logger.debug("[CodingAgent] Raw response:\n%s", raw)

    json_str = _extract_json(raw)
    parsed = json.loads(json_str)

    generated_files: dict[str, str] = parsed["files"]
    entry_point: str = parsed.get("entry_point", "main.py")

    # Materialise files to workspace/<project_id>/
    write_files(
        project_id=state["project_id"],
        files=generated_files,
    )

    logger.info(
        "[CodingAgent] Wrote %d file(s). Entry point: %s",
        len(generated_files),
        entry_point,
    )

    # Mark task as in_progress
    updated_tasks = list(state["tasks"])
    updated_tasks[state["current_task_index"]] = {
        **task,
        "status": "in_progress",
    }

    return {
        **state,
        "generated_files": generated_files,
        "tasks": updated_tasks,
        "status": "executing",
        # Store entry_point in generated_files metadata via a sentinel key
        "_entry_point": entry_point,
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
