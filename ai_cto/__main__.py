"""
__main__.py — CLI entry point for the AI CTO System.

Usage:
    python -m ai_cto "Build a REST API for a todo app" --id todo-api
    python -m ai_cto "Build a weather dashboard" --id weather-dash --verbose
"""

import argparse
import logging
import re
import sys
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-cto",
        description="AI CTO System — turn an idea into working software.",
    )
    parser.add_argument(
        "idea",
        nargs="?",
        help="Describe the software you want to build. Not required when using --resume.",
    )
    parser.add_argument(
        "--id",
        dest="project_id",
        default=None,
        help="Project slug used for the workspace directory (auto-generated if omitted).",
    )
    parser.add_argument(
        "--resume",
        dest="resume_project_id",
        default=None,
        metavar="PROJECT_ID",
        help=(
            "Resume an interrupted project by reading its task_board.json. "
            "The idea argument is not required when resuming."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Validate: need either idea or --resume
    if not args.resume_project_id and not args.idea:
        parser.error("the following arguments are required: idea (or use --resume PROJECT_ID)")

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.resume_project_id:
        _run_resume(args.resume_project_id)
    else:
        project_id = args.project_id or _slugify(args.idea)
        _run_new(args.idea, project_id)


def _run_new(idea: str, project_id: str) -> None:
    from ai_cto.state import initial_state
    from ai_cto.graph import build_graph
    from ai_cto.tools.run_logger import RunLogger

    print(f"\n{'='*60}")
    print(f"  AI CTO System — Project: {project_id}")
    print(f"  Idea: {idea}")
    print(f"{'='*60}\n")

    state = initial_state(idea=idea, project_id=project_id)
    graph = build_graph()
    run_log = RunLogger(project_id)

    # Stream events so the user sees progress in real time
    node_state = state
    for event in graph.stream(state):
        node_name = list(event.keys())[0]
        node_state = event[node_name]
        status = node_state.get("status", "?")
        print(f"[{node_name.upper()}] status={status}")
        run_log.record_event(node_name, node_state)

    run_log.save(node_state)

    # Print final result
    final = node_state  # last event state
    print(f"\n{'='*60}")
    if final.get("status") == "done":
        print("SUCCESS")
        print(final.get("final_output", ""))
    else:
        print("FAILED")
        print(final.get("final_output", "No output available."))
    print(f"{'='*60}\n")

    sys.exit(0 if final.get("status") == "done" else 1)


def _run_resume(resume_project_id: str) -> None:
    from pathlib import Path
    import ai_cto.tools.file_writer as _fw
    from ai_cto.tools.task_board import resume_state_from_board
    from ai_cto.graph import build_resume_graph

    board_path = _fw.WORKSPACE_ROOT / resume_project_id / "task_board.json"

    print(f"\n{'='*60}")
    print(f"  AI CTO System — RESUME: {resume_project_id}")
    print(f"  Loading: {board_path}")
    print(f"{'='*60}\n")

    try:
        state = resume_state_from_board(board_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    completed_before = len(state.get("completed_task_ids", []))
    total = len(state.get("tasks", []))
    print(f"  Resuming from task {state['active_task_id']} "
          f"({completed_before}/{total} tasks already done)\n")

    graph = build_resume_graph()

    from ai_cto.tools.run_logger import RunLogger
    run_log = RunLogger(resume_project_id)

    node_state = state
    for event in graph.stream(state):
        node_name = list(event.keys())[0]
        node_state = event[node_name]
        status = node_state.get("status", "?")
        active = node_state.get("active_task_id")
        print(f"[{node_name.upper()}] status={status}  active_task={active}")
        run_log.record_event(node_name, node_state)

    run_log.save(node_state)
    final = node_state
    print(f"\n{'='*60}")
    if final.get("status") == "done":
        print("SUCCESS")
        print(final.get("final_output", ""))
    else:
        print("FAILED / BLOCKED")
        print(final.get("final_output", "No output available."))
    print(f"{'='*60}\n")

    sys.exit(0 if final.get("status") == "done" else 1)


def _slugify(text: str) -> str:
    """Convert a string into a safe directory slug."""
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:40] or "project"


if __name__ == "__main__":
    main()
