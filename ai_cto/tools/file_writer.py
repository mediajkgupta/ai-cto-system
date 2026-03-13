"""
file_writer.py — Utility: materialise generated files to disk.

The CodingAgent and DebugAgent produce a {filepath: content} dict.
This module writes those files under workspace/<project_id>/.

All paths are treated as relative to the project root and are
sanitised to prevent directory traversal attacks.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Root of all generated project workspaces
WORKSPACE_ROOT = Path(__file__).parent.parent.parent / "workspace"


def write_files(project_id: str, files: dict[str, str]) -> Path:
    """
    Write a dict of {relative_path: content} under workspace/<project_id>/.

    Returns the project directory path.

    Raises:
        ValueError: if a file path attempts directory traversal.
    """
    project_dir = WORKSPACE_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files.items():
        safe_path = _safe_join(project_dir, rel_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        logger.debug("[FileWriter] Wrote: %s", safe_path)

    logger.info("[FileWriter] Wrote %d file(s) to %s", len(files), project_dir)
    return project_dir


def _safe_join(base: Path, rel: str) -> Path:
    """
    Safely join base + rel, raising ValueError on directory traversal.

    Example:
        _safe_join(Path("/workspace/my-app"), "src/main.py")
        → Path("/workspace/my-app/src/main.py")  ✓

        _safe_join(Path("/workspace/my-app"), "../../etc/passwd")
        → raises ValueError  ✗
    """
    target = (base / rel).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise ValueError(
            f"Directory traversal attempt blocked: '{rel}' escapes project root."
        )
    return target
