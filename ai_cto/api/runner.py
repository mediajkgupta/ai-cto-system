"""
runner.py — Background pipeline runner for the AI CTO API.

Launches the LangGraph pipeline in a daemon thread so that POST /projects
returns immediately while the pipeline runs asynchronously.

Design:
  - One thread per project_id (concurrent projects are independent)
  - _RUNS: in-memory run-state registry (RUNNING/COMPLETE/FAILED/CANCELLED)
  - _CANCEL_FLAGS: per-project threading.Event; set to request cancellation
  - Disk artefacts (task_board.json, run_summary.json) are the durable record;
    _RUNS is only used to prevent duplicate starts and to service cancel requests.
    Status reads always prefer disk over _RUNS.
  - Safe to call start_project_run() concurrently for different project IDs.

Cancellation:
  - cancel_project_run(project_id) sets the project's Event flag.
  - _run_pipeline checks the flag between each graph node event and exits early
    if it is set, writing a partial run_summary.json before terminating.
  - Only RUNNING projects can be cancelled; COMPLETE/FAILED returns False.
"""

import logging
import threading
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RunState(str, Enum):
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"
    CANCELLED = "cancelled"


# ── In-process registries ──────────────────────────────────────────────────────

_RUNS: dict[str, RunState] = {}
_CANCEL_FLAGS: dict[str, threading.Event] = {}
_RUNS_LOCK = threading.Lock()


def get_run_state(project_id: str) -> Optional[RunState]:
    """Return the current in-process run state for a project, or None."""
    return _RUNS.get(project_id)


def is_running(project_id: str) -> bool:
    return _RUNS.get(project_id) == RunState.RUNNING


# ── Pipeline launcher ──────────────────────────────────────────────────────────

def start_project_run(idea: str, project_id: str) -> bool:
    """
    Launch the full pipeline for a new project in a background thread.

    Returns True if the run was started, False if a run is already active
    for this project_id (caller should return 409).
    """
    with _RUNS_LOCK:
        if _RUNS.get(project_id) == RunState.RUNNING:
            return False
        cancel_flag = threading.Event()
        _RUNS[project_id] = RunState.RUNNING
        _CANCEL_FLAGS[project_id] = cancel_flag

    thread = threading.Thread(
        target=_run_pipeline,
        args=(idea, project_id, cancel_flag),
        daemon=True,
        name=f"ai-cto-{project_id}",
    )
    thread.start()
    logger.info("[Runner] Started pipeline thread for project '%s'.", project_id)
    return True


def cancel_project_run(project_id: str) -> bool:
    """
    Request cancellation of an actively running project.

    Sets the cancellation Event for the pipeline thread.  The thread will stop
    between the next graph node event and mark itself CANCELLED.

    Returns True if the project was running and the flag was set.
    Returns False if the project is not currently running (already done, failed,
    cancelled, or unknown to this process).
    """
    with _RUNS_LOCK:
        if _RUNS.get(project_id) != RunState.RUNNING:
            return False
        flag = _CANCEL_FLAGS.get(project_id)
        if flag is None:
            return False
        flag.set()
        logger.info("[Runner] Cancellation requested for project '%s'.", project_id)
    return True


# ── Internal ───────────────────────────────────────────────────────────────────

def _run_pipeline(idea: str, project_id: str, cancel_flag: threading.Event) -> None:
    """Thread target. Runs the full pipeline, checking cancel_flag between nodes."""
    from ai_cto.state import initial_state
    from ai_cto.graph import build_graph
    from ai_cto.tools.run_logger import RunLogger

    try:
        state = initial_state(idea=idea, project_id=project_id)
        graph = build_graph()
        run_log = RunLogger(project_id)

        node_state = state
        for event in graph.stream(state):
            # Check for cancellation before processing this node's output
            if cancel_flag.is_set():
                logger.info(
                    "[Runner] Cancellation flag set for '%s' — stopping after node '%s'.",
                    project_id,
                    list(event.keys())[0],
                )
                node_name = list(event.keys())[0]
                node_state = event[node_name]
                run_log.record_event(node_name, node_state)
                break

            node_name = list(event.keys())[0]
            node_state = event[node_name]
            run_log.record_event(node_name, node_state)
            logger.debug(
                "[Runner:%s] node=%s status=%s",
                project_id, node_name, node_state.get("status"),
            )

        run_log.save(node_state)

        with _RUNS_LOCK:
            if cancel_flag.is_set():
                _RUNS[project_id] = RunState.CANCELLED
                logger.info("[Runner] Project '%s' cancelled.", project_id)
            else:
                final_status = node_state.get("status", "unknown")
                _RUNS[project_id] = (
                    RunState.COMPLETE if final_status == "done" else RunState.FAILED
                )
                logger.info(
                    "[Runner] Pipeline complete for '%s': final_status=%s",
                    project_id, final_status,
                )

    except Exception as exc:  # noqa: BLE001
        logger.error("[Runner] Pipeline crashed for '%s': %s", project_id, exc, exc_info=True)
        with _RUNS_LOCK:
            _RUNS[project_id] = RunState.FAILED
