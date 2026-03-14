"""
execution.py — Execution Agent

Responsibilities:
  - Take the files written to workspace/<project_id>/
  - Run the entry point inside a Docker container (or subprocess fallback)
  - Capture stdout, stderr, and exit code
  - Write ExecutionResult into state

Safety design:
  - Docker mode: --network none, read-only filesystem, memory capped
  - Subprocess fallback: used only in local dev when Docker is unavailable
  - Hard timeout of 30 seconds prevents runaway processes

Timeout handling:
  - A single timeout prefixes the error_context with [TIMEOUT] and routes
    to the debug loop (may be a slow start, not a hard bug).
  - Two consecutive timeouts on the same task skip the debug loop and mark
    the task failed immediately (the code is stuck, not incorrect).
"""

import logging
import ai_cto.tools.file_writer as _fw
from ai_cto.state import ProjectState, ExecutionResult
from ai_cto.tools.sandbox import run_in_docker, run_in_subprocess, docker_available

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 30        # seconds
MAX_CONSECUTIVE_TIMEOUTS = 2  # fail the task after this many consecutive timeouts

_TIMEOUT_MARKER = "Execution timed out"


# ── Agent node ─────────────────────────────────────────────────────────────────

def execution_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Execution Agent.

    Transforms state:
        state.generated_files (on disk)  →  state.execution_result
    """
    project_id = state["project_id"]
    entry_point = state.get("_entry_point", "main.py")

    logger.info(
        "[ExecutionAgent] Running project '%s', entry: %s",
        project_id,
        entry_point,
    )

    workspace_path = _workspace_path(project_id)

    if docker_available():
        logger.info("[ExecutionAgent] Using Docker sandbox.")
        result = run_in_docker(
            workspace_path=workspace_path,
            entry_point=entry_point,
            timeout=EXECUTION_TIMEOUT,
        )
    else:
        logger.warning(
            "[ExecutionAgent] Docker not available — using subprocess fallback. "
            "Do NOT use this in production."
        )
        result = run_in_subprocess(
            workspace_path=workspace_path,
            entry_point=entry_point,
            timeout=EXECUTION_TIMEOUT,
        )

    logger.info(
        "[ExecutionAgent] Exit code: %d | stdout: %d chars | stderr: %d chars",
        result["exit_code"],
        len(result["stdout"]),
        len(result["stderr"]),
    )

    # ── Success ───────────────────────────────────────────────────────────────
    if result["exit_code"] == 0:
        return {
            **state,
            "execution_result": result,
            "error_context": "",
            "execution_timeout_count": 0,
            "status": "done",
        }

    # ── Failure: check for timeout ────────────────────────────────────────────
    is_timeout = _TIMEOUT_MARKER in result.get("stderr", "")
    timeout_count = state.get("execution_timeout_count", 0)

    if is_timeout:
        timeout_count += 1
        logger.warning(
            "[ExecutionAgent] Execution timed out (consecutive count=%d/%d).",
            timeout_count,
            MAX_CONSECUTIVE_TIMEOUTS,
        )

        if timeout_count >= MAX_CONSECUTIVE_TIMEOUTS:
            error_context = (
                f"[TIMEOUT] Task timed out {timeout_count} time(s) consecutively "
                f"(>{EXECUTION_TIMEOUT}s). The code appears stuck; marking task failed.\n\n"
                + _build_error_context(result)
            )
            logger.error(
                "[ExecutionAgent] Consecutive timeout limit reached — failing task."
            )
            return {
                **state,
                "execution_result": result,
                "error_context": error_context,
                "execution_timeout_count": timeout_count,
                "status": "failed",
            }

        # Single timeout — route through debug loop with advisory prefix
        error_context = (
            f"[TIMEOUT] Execution exceeded {EXECUTION_TIMEOUT}s. "
            f"This may be an infinite loop or a blocking call. "
            f"Check for missing exit conditions or blocking I/O.\n\n"
            + _build_error_context(result)
        )
        return {
            **state,
            "execution_result": result,
            "error_context": error_context,
            "execution_timeout_count": timeout_count,
            "status": "debugging",
        }

    # ── Non-timeout failure ───────────────────────────────────────────────────
    return {
        **state,
        "execution_result": result,
        "error_context": _build_error_context(result),
        "execution_timeout_count": 0,   # reset; this was a code bug, not a timeout
        "status": "debugging",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _workspace_path(project_id: str) -> str:
    """
    Return the absolute path to the project workspace directory.

    Reads WORKSPACE_ROOT from file_writer at call time (not import time)
    so that test fixtures can redirect it via monkeypatch.
    """
    return str(_fw.WORKSPACE_ROOT / project_id)


def _build_error_context(result: ExecutionResult) -> str:
    """Combine stdout + stderr into a single error string for the DebugAgent."""
    parts = []
    if result["stdout"].strip():
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"].strip():
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"EXIT CODE: {result['exit_code']}")
    return "\n\n".join(parts)
