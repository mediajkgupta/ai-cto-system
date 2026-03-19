"""
run_logger.py — Per-run audit log for the AI CTO pipeline.

Records one entry per graph node invocation, then writes a structured
run_summary.json to workspace/<project_id>/ when the run completes.

Usage (in graph stream loop):
    logger = RunLogger(project_id)
    for event in graph.stream(state):
        for node_name, node_state in event.items():
            logger.record_event(node_name, node_state)
    logger.save(final_state)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency at module load time
_WORKSPACE_ROOT: Optional[Path] = None


def _workspace_root() -> Path:
    global _WORKSPACE_ROOT
    if _WORKSPACE_ROOT is None:
        from ai_cto.tools.file_writer import WORKSPACE_ROOT
        _WORKSPACE_ROOT = WORKSPACE_ROOT
    return _WORKSPACE_ROOT


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """
    Accumulates agent events for a single pipeline run and persists them
    to workspace/<project_id>/run_summary.json on completion.

    Thread safety: not thread-safe; designed for single-threaded graph execution.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._started_at = _now_iso()
        self._events: list[dict] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def record_event(self, node_name: str, node_state: dict) -> None:
        """
        Append one event to the run log.

        Called once per node invocation in the graph stream loop.
        Captures: node name, routing status, active task id, timestamp.
        """
        entry = {
            "node": node_name,
            "timestamp": _now_iso(),
            "status": node_state.get("status", ""),
            "active_task_id": node_state.get("active_task_id"),
        }
        # Attach error context snippet (first 200 chars) if present
        error_context = node_state.get("error_context", "")
        if error_context:
            entry["error_snippet"] = error_context[:200]

        self._events.append(entry)
        logger.debug("[RunLogger] Event: node=%s status=%s", node_name, entry["status"])

    def save(self, final_state: dict) -> Optional[Path]:
        """
        Write run_summary.json to workspace/<project_id>/.

        Returns the path of the written file, or None if the write failed.

        Summary contains:
            project_id, idea, started_at, finished_at, duration_seconds,
            final_status, tasks_total, tasks_completed, tasks_failed,
            files_generated, events
        """
        try:
            finished_at = _now_iso()
            duration = _compute_duration(self._started_at, finished_at)

            tasks = final_state.get("tasks", [])
            generated_files = final_state.get("generated_files", {})

            from ai_cto.tools.status import get_status_report
            status_report = get_status_report(final_state)

            summary = {
                "project_id": self.project_id,
                "idea": final_state.get("idea", ""),
                "started_at": self._started_at,
                "finished_at": finished_at,
                "duration_seconds": duration,
                "status": status_report,
                "files_generated": sorted(generated_files.keys()),
                "files_generated_count": len(generated_files),
                "events": self._events,
            }

            out_path = _workspace_root() / self.project_id / "run_summary.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            logger.info(
                "[RunLogger] Run summary written to %s (%d events, %ds).",
                out_path,
                len(self._events),
                duration or 0,
            )
            return out_path

        except Exception as exc:  # noqa: BLE001
            logger.error("[RunLogger] Failed to write run summary: %s", exc)
            return None


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _compute_duration(started_at: str, finished_at: str) -> Optional[int]:
    """Return elapsed time in whole seconds, or None if parsing fails."""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
        return int((end - start).total_seconds())
    except Exception:  # noqa: BLE001
        return None
