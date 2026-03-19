"""
verification.py — Verification Agent

Runs between CodingAgent and ExecutionAgent.

Responsibilities:
  1. Run all static checks (syntax, security, deps, architecture, runtime risks)
  2. Generate pytest tests via LLM (non-blocking — failure is a warning only)
  3. Write generated tests to workspace/<project_id>/tests/ on disk
  4. If ERROR-level issues found:
       - Populate state.error_context with a structured report
       - Set status="verifying_failed" → graph routes to DebugAgent
  5. If only warnings (or clean):
       - Attach VerificationResult + generated tests to state
       - Set status="executing" → graph routes to ExecutionAgent

Routing contract:
    status="verifying_failed"  →  DebugAgent  (errors that must be fixed before running)
    status="executing"         →  ExecutionAgent
"""

import logging
from ai_cto.state import ProjectState
from ai_cto.verification.checks import (
    run_all_checks, VerificationResult, detect_stack, detect_node_entry,
)
from ai_cto.verification.test_generator import generate_tests
from ai_cto.tools.file_writer import write_files

logger = logging.getLogger(__name__)


def verification_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Verification Agent.

    Transforms state:
        state.generated_files  →  state.verification_result + state.error_context (on failure)
    """
    entry_point = state.get("_entry_point", "main.py")
    files = state["generated_files"]

    # ── Auto-detect stack and correct entry point ──────────────────────────────
    # Prevents 'main.py' being used as a universal success path for non-Python
    # projects.  When Node.js files are detected, override entry_point to the
    # real JS entry file — or keep the explicit value if it names a non-empty JS
    # file itself.
    stack = detect_stack(files)
    if stack == "node":
        current_is_empty = not files.get(entry_point, "").strip()
        current_is_python_default = entry_point == "main.py"
        if current_is_empty or current_is_python_default:
            js_entry = detect_node_entry(files)
            if js_entry:
                logger.info(
                    "[VerificationAgent] Node.js stack detected — overriding entry "
                    "point '%s' → '%s'.",
                    entry_point, js_entry,
                )
                entry_point = js_entry
            else:
                logger.warning(
                    "[VerificationAgent] Node.js stack detected but no non-empty "
                    "JS entry file found — proceeding with '%s'.",
                    entry_point,
                )

    logger.info(
        "[VerificationAgent] Checking %d file(s), entry point: %s (stack: %s)",
        len(files),
        entry_point,
        stack,
    )

    # ── Step 1: static checks (no LLM) ────────────────────────────────────────
    result = run_all_checks(files=files, entry_point=entry_point)

    # ── Step 2: generate tests (LLM, non-blocking) ────────────────────────────
    generated_tests = _try_generate_tests(state)
    result = VerificationResult(**{**result, "generated_tests": generated_tests})

    # Write generated test files to disk alongside the project
    if generated_tests:
        write_files(project_id=state["project_id"], files=generated_tests)
        logger.info(
            "[VerificationAgent] Wrote %d test file(s) to workspace.", len(generated_tests)
        )

    # ── Step 3: decide routing ─────────────────────────────────────────────────
    if not result["passed"]:
        error_context = _format_verification_errors(result)
        logger.warning(
            "[VerificationAgent] FAILED — %d error(s), %d warning(s). Routing to DebugAgent.",
            result["error_count"],
            result["warning_count"],
        )
        return {
            **state,
            "verification_result": result,
            "error_context": error_context,
            "status": "verifying_failed",   # controller maps this → debug
        }

    logger.info(
        "[VerificationAgent] PASSED — %d warning(s). Routing to ExecutionAgent.",
        result["warning_count"],
    )
    return {
        **state,
        "verification_result": result,
        "error_context": "",
        "status": "executing",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _try_generate_tests(state: ProjectState) -> dict[str, str]:
    """
    Attempt test generation. Returns {} on any failure — never blocks the pipeline.

    Uses _current_task_files when available (set by CodingAgent to just the files
    written in this coding pass) so the TestGenerator writes tests scoped to what
    the current task actually produces — not the full accumulated project state.
    This prevents scaffold tasks from triggering complete CRUD test suites before
    the implementation exists.
    """
    # Prefer current-task scope; fall back to full accumulated files if not set
    task_files = state.get("_current_task_files") or state["generated_files"]
    try:
        return generate_tests(
            files=task_files,
            architecture=state.get("architecture", ""),
            project_id=state["project_id"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[VerificationAgent] Test generation failed (non-fatal): %s", exc)
        return {}


def _format_verification_errors(result: VerificationResult) -> str:
    """
    Build an error_context string from ERROR-severity issues.
    This is fed directly into the DebugAgent prompt.
    """
    errors = [i for i in result["issues"] if i["severity"] == "error"]
    lines = [
        "VERIFICATION FAILED — the following errors must be fixed before execution:\n"
    ]
    for issue in errors:
        loc = issue["file"]
        if issue["line"]:
            loc += f":{issue['line']}"
        lines.append(f"  [{issue['check'].upper()}] {loc}")
        lines.append(f"  {issue['message']}\n")

    # Append warnings as informational context for the DebugAgent
    warnings = [i for i in result["issues"] if i["severity"] == "warning"]
    if warnings:
        lines.append(f"\nAdditionally, {len(warnings)} warning(s) were found:")
        for w in warnings:
            loc = w["file"]
            if w["line"]:
                loc += f":{w['line']}"
            lines.append(f"  [{w['check']}] {loc} — {w['message']}")

    return "\n".join(lines)
