"""
debug.py — Debug Agent (V2)

Responsibilities:
  - Receive error context (stderr + exit code or verification errors) from state
  - Receive the failing generated_files
  - Call Claude to diagnose the error and produce a patched file set
  - Track retries PER TASK (TaskItem.retries) rather than per project
  - If task.max_retries exceeded: mark that task failed, persist task board,
    and halt the project with status="failed"
  - Otherwise: write the patch, update task.retries, persist board, route back
    to CodingAgent for re-execution

V2 changes from V1:
  - Per-task retry counter (TaskItem.retries) is the primary gate
  - state.debug_attempts still incremented for backward compat with MemoryAgent
  - Exhausted path calls mark_task_failed + persist_task_board
  - Successful patch path calls persist_task_board
"""

import json
import re
import logging
from ai_cto.providers import get_provider
from ai_cto.state import (
    ProjectState, get_active_task, update_task_in_list, now_iso,
    task_history_event,
)
from ai_cto.tools.file_writer import write_files
from ai_cto.tools.task_board import mark_task_failed, persist_task_board
from ai_cto.mock_llm import is_mock_mode, mock_debug_node
from ai_cto.agents.memory import retrieve_debug_solutions

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software debugger.
You will be given:
1. The error output from running a program.
2. The current source files that caused the error.
3. Optionally: relevant past debug solutions from similar errors.

Your job:
- Diagnose the root cause of the error (use past solutions as hints if relevant).
- Return patched versions of only the files that need to change.

Return your response as valid JSON with this exact shape:
{
  "diagnosis": "<one paragraph explaining the root cause>",
  "files": {
    "relative/path/to/fixed_file.py": "<full corrected file contents>",
    ...
  }
}

Rules:
- Only include files you actually changed.
- File contents must be complete — do not use '...' or placeholder comments.
- Fix the root cause, not just the symptom.
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def debug_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Debug Agent.

    Transforms state:
        state.error_context + state.generated_files  →  patched generated_files

    V2 retry contract:
        - reads  task.retries      (per-task; incremented here)
        - reads  task.max_retries  (per-task cap; defaults to state.max_debug_attempts)
        - writes task.retries, task.updated_at  on each attempt
        - on exhaustion: marks task failed, persists board, returns status="failed"
        - also increments state.debug_attempts for MemoryAgent backward compat
    """
    active = get_active_task(state)

    # ── Compute retry counters ─────────────────────────────────────────────────
    if active is not None:
        task_retries = active.get("retries", 0) + 1
        task_max_retries = active.get("max_retries", state["max_debug_attempts"])
    else:
        # Fallback: no active task tracked — use project-level counter
        task_retries = state["debug_attempts"] + 1
        task_max_retries = state["max_debug_attempts"]

    logger.info(
        "[DebugAgent] Debug attempt %d/%d for task id=%s in project '%s'",
        task_retries,
        task_max_retries,
        active["id"] if active else "?",
        state["project_id"],
    )

    # ── Guard: exhaustion check (runs before mock mode) ────────────────────────
    if task_retries > task_max_retries:
        logger.error(
            "[DebugAgent] Task id=%s exceeded max retries (%d). Marking failed.",
            active["id"] if active else "?",
            task_max_retries,
        )
        error_msg = state.get("error_context", "")

        if active is not None:
            failed_updates = mark_task_failed(state, active["id"], error_msg)
            # Add a "failed" history entry
            history = failed_updates.get("task_history", state.get("task_history", []))
            failed_state = {
                **state,
                **failed_updates,
                "task_history": history,
                "debug_attempts": task_retries,
                "project_status": "failed",
                "status": "failed",
                "final_output": (
                    f"Project failed: task '{active['title']}' exhausted "
                    f"{task_max_retries} debug attempt(s).\n"
                    f"Last error:\n{error_msg}"
                ),
            }
        else:
            failed_state = {
                **state,
                "debug_attempts": task_retries,
                "project_status": "failed",
                "status": "failed",
                "final_output": (
                    f"Project failed after {task_max_retries} debug attempt(s).\n"
                    f"Last error:\n{error_msg}"
                ),
            }

        persist_task_board(failed_state)
        return failed_state

    # ── Mock mode ──────────────────────────────────────────────────────────────
    if is_mock_mode():
        result = mock_debug_node(state, task_retries)
        # Update per-task retries in the result so it's reflected in task_board.json
        if active is not None:
            updated_tasks = update_task_in_list(
                result["tasks"], active["id"],
                retries=task_retries,
                updated_at=now_iso(),
            )
            history = list(result.get("task_history", [])) + [
                task_history_event(active["id"], "retry",
                                   detail=f"attempt {task_retries}")
            ]
            result = {
                **result,
                "tasks": updated_tasks,
                "task_history": history,
                "debug_attempts": task_retries,
            }
            persist_task_board(result)
        return result

    # ── Real LLM call ──────────────────────────────────────────────────────────
    provider = get_provider()

    # Fetch debug solutions from memory store relevant to this specific error.
    # This is a direct store query (not the state.memory_context which only
    # contains architecture + project patterns from the start of the run).
    debug_memory = retrieve_debug_solutions(
        state["error_context"],
        exclude_mock=not is_mock_mode(),
    )
    user_message = ""
    if debug_memory:
        user_message += f"{debug_memory}\n\n---\n\n"

    user_message += (
        f"Error output:\n{state['error_context']}\n\n"
        f"Current source files:\n"
        f"{json.dumps(state['generated_files'], indent=2)}"
    )

    raw = provider.complete(
        system=SYSTEM_PROMPT,
        user=user_message,
        max_tokens=4096,
    )
    logger.debug("[DebugAgent] Raw response:\n%s", raw)

    json_str = _extract_json(raw)
    parsed = json.loads(json_str)

    diagnosis: str = parsed.get("diagnosis", "")
    patched_files: dict[str, str] = parsed.get("files", {})

    logger.info(
        "[DebugAgent] Diagnosis: %s | Patching %d file(s).",
        diagnosis[:120],
        len(patched_files),
    )

    # Merge patches + write to disk
    merged_files = {**state["generated_files"], **patched_files}
    write_files(project_id=state["project_id"], files=patched_files)

    # Update per-task retries on the task object
    tasks = state["tasks"]
    if active is not None:
        tasks = update_task_in_list(
            tasks, active["id"],
            retries=task_retries,
            updated_at=now_iso(),
        )

    # Record retry event in history
    history = list(state.get("task_history", [])) + [
        task_history_event(
            active["id"] if active else -1,
            "retry",
            detail=f"attempt {task_retries}: {diagnosis[:100]}",
        )
    ]

    new_state = {
        **state,
        "generated_files": merged_files,
        "tasks": tasks,
        "task_history": history,
        "debug_attempts": task_retries,          # backward compat for MemoryAgent
        "last_debug_diagnosis": diagnosis,        # saved to memory on completion
        "_last_error_context": state["error_context"],  # the error that was fixed
        "error_context": "",      # cleared — will be repopulated by ExecutionAgent
        "status": "coding",       # route back to CodingAgent for re-execution
    }

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
