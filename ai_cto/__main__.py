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
        help="Describe the software you want to build.",
    )
    parser.add_argument(
        "--id",
        dest="project_id",
        default=None,
        help="Project slug used for the workspace directory (auto-generated if omitted).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Derive project_id from idea if not provided
    project_id = args.project_id or _slugify(args.idea)

    # Lazy import to avoid loading heavy deps before env is configured
    from ai_cto.state import initial_state
    from ai_cto.graph import build_graph

    print(f"\n{'='*60}")
    print(f"  AI CTO System — Project: {project_id}")
    print(f"  Idea: {args.idea}")
    print(f"{'='*60}\n")

    state = initial_state(idea=args.idea, project_id=project_id)
    graph = build_graph()

    # Stream events so the user sees progress in real time
    for event in graph.stream(state):
        node_name = list(event.keys())[0]
        node_state = event[node_name]
        status = node_state.get("status", "?")
        print(f"[{node_name.upper()}] status={status}")

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


def _slugify(text: str) -> str:
    """Convert a string into a safe directory slug."""
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:40] or "project"


if __name__ == "__main__":
    main()
