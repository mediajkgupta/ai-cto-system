"""
sandbox.py — Safe code execution utilities.

Two execution modes:
  1. Docker (preferred) — isolated container, no network, memory capped
  2. Subprocess (fallback) — local Python process, dev-only

Docker command used:
    docker run --rm
        --network none          (no outbound network)
        --memory 256m           (memory cap)
        --cpus 0.5              (CPU cap)
        --read-only             (immutable filesystem)
        --tmpfs /tmp            (allow temp writes)
        -v <workspace>:/app:ro  (mount project read-only)
        -w /app
        python:3.11-slim
        python <entry_point>
"""

import os
import shutil
import subprocess
import logging
from ai_cto.state import ExecutionResult

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def docker_available() -> bool:
    """Return True if the Docker CLI is present and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_in_docker(
    workspace_path: str,
    entry_point: str,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Run `python <entry_point>` inside an isolated Docker container.

    The workspace directory is mounted read-only at /app.
    """
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/tmp",
        "-v", f"{workspace_path}:/app:ro",
        "-w", "/app",
        "python:3.11-slim",
        "python", entry_point,
    ]

    logger.debug("[Sandbox] Docker command: %s", " ".join(cmd))

    return _run(cmd, timeout)


def run_in_subprocess(
    workspace_path: str,
    entry_point: str,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Run `python <entry_point>` as a local subprocess.

    WARNING: No isolation — use only for local development.
    """
    entry_abs = os.path.join(workspace_path, entry_point)
    cmd = ["python", entry_abs]

    logger.debug("[Sandbox] Subprocess command: %s", " ".join(cmd))

    return _run(cmd, timeout, cwd=workspace_path)


# ── Internal ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int, cwd: str | None = None) -> ExecutionResult:
    """Execute a command and return an ExecutionResult."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[Sandbox] Execution timed out after %ds.", timeout)
        return ExecutionResult(
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            exit_code=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[Sandbox] Unexpected error: %s", exc)
        return ExecutionResult(
            stdout="",
            stderr=str(exc),
            exit_code=1,
        )
