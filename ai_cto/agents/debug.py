"""
debug.py — Debug Agent

Responsibilities:
  - Receive the error context (stderr + exit code) from ProjectState
  - Receive the failing generated_files
  - Call Claude to diagnose the error and produce a patched file set
  - Increment debug_attempts counter
  - If max_debug_attempts exceeded, mark status as 'failed'
  - Otherwise, route back to the Coding Agent for re-execution

The Debug Agent does NOT rewrite from scratch — it patches specific files.
"""

import json
import re
import logging
from anthropic import Anthropic
from ai_cto.state import ProjectState
from ai_cto.tools.file_writer import write_files
from ai_cto.mock_llm import is_mock_mode, mock_debug_node

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
        state.error_context + state.generated_files  →  patched state.generated_files
    """
    attempts = state["debug_attempts"] + 1
    max_attempts = state["max_debug_attempts"]

    logger.info(
        "[DebugAgent] Debug attempt %d/%d for project '%s'",
        attempts,
        max_attempts,
        state["project_id"],
    )

    # Guard: always checked first, even in mock mode
    if attempts > max_attempts:
        logger.error("[DebugAgent] Max debug attempts reached. Marking as failed.")
        return {
            **state,
            "debug_attempts": attempts,
            "status": "failed",
            "final_output": (
                f"Project failed after {max_attempts} debug attempts.\n"
                f"Last error:\n{state['error_context']}"
            ),
        }

    if is_mock_mode():
        return mock_debug_node(state, attempts)

    client = Anthropic()

    # Retrieve past debug solutions for similar errors (injected by MemoryAgent)
    memory_context = state.get("memory_context", "")

    user_message = ""
    if memory_context:
        user_message += f"Past debug solutions for reference:\n{memory_context}\n\n---\n\n"

    user_message += (
        f"Error output:\n{state['error_context']}\n\n"
        f"Current source files:\n"
        f"{json.dumps(state['generated_files'], indent=2)}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text
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

    # Merge patches into the existing file set
    merged_files = {**state["generated_files"], **patched_files}

    # Write patched files to disk
    write_files(project_id=state["project_id"], files=patched_files)

    return {
        **state,
        "generated_files": merged_files,
        "debug_attempts": attempts,
        "last_debug_diagnosis": diagnosis,           # saved to memory on completion
        "_last_error_context": state["error_context"],  # the error that was fixed
        "error_context": "",      # cleared — will be repopulated by ExecutionAgent
        "status": "coding",       # route back to CodingAgent for re-execution
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
