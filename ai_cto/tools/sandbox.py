"""
sandbox.py — Safe code execution utilities.

Two execution modes:
  1. Docker (preferred) — isolated container, no network, memory capped
  2. Subprocess (fallback) — local process, dev-only

Runtime config is detected automatically from the generated file set:
  - Python projects  → python:3.11-slim  / python <entry>
  - Node.js projects → node:18-alpine    / node <entry>

Docker command (always uses a writable /project tmpfs so installs work):
    docker run --rm
        --network none          (no outbound network)
        --memory <limit>        (256 MB Python / 512 MB Node)
        --cpus 0.5              (CPU cap)
        --read-only             (immutable base filesystem)
        --tmpfs /project:rw,exec,size=100m  (writable work dir)
        --tmpfs /tmp            (allow temp writes)
        -v <workspace>:/app:ro  (mount project read-only)
        <image>
        sh -c "cp -r /app/. /project && [install] && <command>"
"""

import os
import shutil
import subprocess
import logging
from typing import TypedDict, Optional

from ai_cto.state import ExecutionResult

logger = logging.getLogger(__name__)


# ── Runtime configuration ───────────────────────────────────────────────────────

class RuntimeConfig(TypedDict):
    """Per-language execution recipe used by both Docker and subprocess runners."""
    image: str               # Docker image tag
    memory: str              # Docker --memory value  (e.g. "256m")
    command: list            # Run command, relative to project dir  e.g. ["python", "main.py"]
    install_cmd: Optional[list]  # Optional pre-run install command, or None


def get_runtime_config(files: dict, entry_point: str) -> RuntimeConfig:
    """
    Detect the language stack from generated files and return a RuntimeConfig.

    Detection rules (same as verification stack detection):
      - Any .js/.ts file or package.json present → Node.js
      - Otherwise → Python

    Node.js:
      image   = node:18-alpine
      install = npm install --production  (only if package.json present)
      command = node <entry_point>

    Python:
      image   = python:3.11-slim
      install = pip install -q -r requirements.txt  (only if requirements.txt present)
      command = python <entry_point>
    """
    # Lazy import to avoid circular dependency (verification.checks imports sandbox indirectly)
    from ai_cto.verification.checks import detect_stack

    stack = detect_stack(files)

    if stack == "node":
        has_package_json = "package.json" in files
        return RuntimeConfig(
            image="node:18-alpine",
            memory="512m",   # npm needs more headroom
            command=["node", entry_point],
            install_cmd=["npm", "install", "--production"] if has_package_json else None,
        )

    # Python or unknown
    has_requirements = "requirements.txt" in files
    return RuntimeConfig(
        image="python:3.11-slim",
        memory="256m",
        command=["python", entry_point],
        install_cmd=["pip", "install", "-q", "-r", "requirements.txt"] if has_requirements else None,
    )


# ── Public API ──────────────────────────────────────────────────────────────────

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
    files: Optional[dict] = None,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Run the project in an isolated Docker container.

    The workspace is mounted read-only at /app. The container copies it to
    /project (a writable tmpfs), runs any required installs, then executes
    the entry point. This ensures installs (npm, pip) can write to the project
    directory while the base filesystem remains immutable.

    Args:
        workspace_path: Absolute path to the project workspace on the host.
        entry_point:    Relative path of the entry file (e.g. "main.py", "server.js").
        files:          Generated file dict used for stack detection. If None,
                        detected from entry_point extension.
        timeout:        Hard execution timeout in seconds.
    """
    config = get_runtime_config(files or {}, entry_point)

    # Build the shell command that runs inside the container
    shell_parts = ["cp -r /app/. /project"]
    if config["install_cmd"]:
        shell_parts.append(" ".join(config["install_cmd"]))
    shell_parts.append(" ".join(config["command"]))
    shell_cmd = " && ".join(shell_parts)

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", config["memory"],
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/project:rw,exec,size=100m",
        "--tmpfs", "/tmp",
        "-v", f"{workspace_path}:/app:ro",
        "-w", "/project",
        config["image"],
        "sh", "-c", shell_cmd,
    ]

    logger.debug(
        "[Sandbox] Docker command (%s): %s",
        config["image"],
        " ".join(cmd),
    )
    return _run(cmd, timeout)


def run_in_subprocess(
    workspace_path: str,
    entry_point: str,
    files: Optional[dict] = None,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Run the project as a local subprocess (dev fallback — no isolation).

    Detects the language stack and invokes the correct interpreter.
    Runs from workspace_path so relative imports work.

    WARNING: No sandboxing — use only for local development.
    """
    config = get_runtime_config(files or {}, entry_point)
    cmd = config["command"]   # e.g. ["python", "main.py"] or ["node", "server.js"]

    logger.debug(
        "[Sandbox] Subprocess command (%s): %s",
        config["image"],
        " ".join(cmd),
    )
    return _run(cmd, timeout, cwd=workspace_path)


# ── Internal ────────────────────────────────────────────────────────────────────

def _run(cmd: list, timeout: int, cwd: str | None = None) -> ExecutionResult:
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
