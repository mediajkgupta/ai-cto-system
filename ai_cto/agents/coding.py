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
from ai_cto.state import ProjectState, get_active_task, update_task_in_list
from ai_cto.tools.file_writer import write_files
from ai_cto.mock_llm import is_mock_mode, mock_coding_node

logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 2  # extra attempts after the first on JSON parse failure

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

    Retries the LLM call up to MAX_PARSE_RETRIES times on JSON parse failure.
    On exhaustion returns status='verifying_failed' so the debug loop can
    record the failure and advance to the next retry.
    """
    task = get_active_task(state) or state["tasks"][state["current_task_index"]]
    logger.info("[CodingAgent] Implementing task %d: %s", task["id"], task["title"])

    if is_mock_mode():
        return mock_coding_node(state)

    client = Anthropic()
    user_message = _build_user_message(state, task)

    parsed = None
    last_error = None
    for attempt in range(1, MAX_PARSE_RETRIES + 2):  # 1 .. MAX_PARSE_RETRIES+1
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text
            logger.debug("[CodingAgent] Raw response (attempt %d):\n%s", attempt, raw)

            json_str = _extract_json(raw)
            parsed = json.loads(json_str)
            if "files" not in parsed or not isinstance(parsed["files"], dict):
                raise ValueError("Response JSON missing 'files' dict.")
            break

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[CodingAgent] Attempt %d/%d failed to parse (%s). %s.",
                attempt,
                MAX_PARSE_RETRIES + 1,
                exc,
                "Retrying" if attempt <= MAX_PARSE_RETRIES else "Giving up",
            )

    if parsed is None:
        error_msg = (
            f"CodingAgent failed to produce valid JSON after "
            f"{MAX_PARSE_RETRIES + 1} attempt(s). Last error: {last_error}"
        )
        logger.error("[CodingAgent] %s", error_msg)
        return {
            **state,
            "error_context": error_msg,
            "status": "verifying_failed",
        }

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

    # Mark active task as in_progress
    updated_tasks = update_task_in_list(state["tasks"], task["id"], status="in_progress")

    return {
        **state,
        "generated_files": generated_files,
        "tasks": updated_tasks,
        "status": "executing",
        # Store entry_point in generated_files metadata via a sentinel key
        "_entry_point": entry_point,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_message(state: ProjectState, task: dict) -> str:
    """
    Build the user-turn message for the LLM.

    Three cases:
      1. Fresh start: architecture + task.
      2. Retry after execution failure (error_context set): include previous
         error and previously generated files for reference.
      3. Retry after a successful debug patch (debug_attempts > 0, no
         error_context): inject patched file contents so the LLM preserves
         all prior fixes and only makes targeted changes.
    """
    msg = (
        f"Architecture:\n{state['architecture']}\n\n"
        f"Task to implement:\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
    )

    if state.get("error_context"):
        # Case 2: retrying after a known failure
        msg += (
            f"\nPrevious attempt failed with this error — fix it:\n"
            f"{state['error_context']}\n\n"
            f"Previously generated files for reference:\n"
            f"{json.dumps(state['generated_files'], indent=2)}"
        )
    elif state.get("debug_attempts", 0) > 0 and state.get("generated_files"):
        # Case 3: debug agent produced a patch; preserve its fixes
        patched_contents = "\n".join(
            f"### {path}\n```python\n{content}\n```"
            for path, content in state["generated_files"].items()
        )
        msg += (
            f"\nPreviously patched files (preserve these fixes, "
            f"only modify what is strictly necessary):\n"
            f"{patched_contents}"
        )

    return msg


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
