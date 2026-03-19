"""
task_manager.py — Task Manager Agent (V2, LLM-backed)

Two modes, same node:
  1. Initial planning  (state.tasks == []):
       architecture → full structured task list
  2. Refinement        (state.tasks != []):
       existing partial task list + history → refined / extended list
       (called e.g. when resuming or if the pipeline needs more tasks)

Both modes:
  - Pass full state context to the LLM so it knows what is already done
  - Protect completed / failed / blocked tasks — they are NEVER overwritten
  - Validate LLM output (duplicate IDs, missing dep references)
  - Retry the LLM call up to MAX_PLAN_RETRIES times on parse / validation failure
  - Persist task_board.json after the final task list is settled
  - Set active_task_id via get_next_runnable_task() (priority + dep scheduling)
  - Record a "started" history event for the first active task

Mock mode: returns MOCK_TASKS / MOCK_TASKS_EXTENDED without any API call.
"""

import json
import re
import logging
from ai_cto.providers import get_provider
from ai_cto.state import (
    ProjectState, make_task_item, get_next_runnable_task,
    task_history_event,
)
from ai_cto.tools.task_board import persist_task_board
from ai_cto.mock_llm import is_mock_mode, mock_task_manager_node

# ── Post-processing filter: keyword sets ────────────────────────────────────

_SETUP_WORDS = frozenset({
    "setup", "set", "install", "installation", "environment", "env",
    "initialize", "init", "bootstrap", "scaffold", "configure",
    "configuration", "prerequisite", "dependencies", "venv", "virtualenv",
})
_TEST_WORDS = frozenset({
    "test", "tests", "testing", "spec", "specs", "jest", "pytest",
    "unittest", "mocha", "coverage",
})
_DEPLOY_WORDS = frozenset({
    "deploy", "deployment", "docker", "dockerize", "dockerfile",
    "containerize", "container", "kubernetes", "k8s", "pipeline",
    "devops", "ansible", "terraform", "nginx", "release", "makefile",
})

_TEST_OUTPUT_PATTERNS = (
    ".test.js", ".spec.js", ".test.ts", ".spec.ts",
    ".test.py", "_test.py", "test_", "/test/", "/tests/",
)
_DEPLOY_OUTPUT_PATTERNS = (
    "dockerfile", "docker-compose", ".github/",
    "jenkinsfile", "makefile", ".sh",
)

logger = logging.getLogger(__name__)

MAX_PLAN_RETRIES = 2  # extra attempts after the first on parse / validation failure

# ── System prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software project manager and AI CTO.
Decompose the given software project into a structured, ordered task list for
an autonomous coding agent.

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "tasks": [
    {
      "id": 1,
      "title": "short action title (max 60 chars)",
      "description": "detailed description of what to implement (2-4 sentences)",
      "priority": 1,
      "depends_on": [],
      "outputs": ["main.py"]
    }
  ]
}

HARD RULES — violating any of these makes the output invalid:

WHAT TO INCLUDE:
- 2-5 tasks only. Each task writes source code files.
- Every task MUST list at least one concrete output file (e.g. "app.py", "models.py").
- Each task must be implementable by writing or editing source files alone.
- Cover the full implementation with no gaps.

WHAT TO EXCLUDE — do NOT include any task that:
- Sets up a development environment, installs tools, or configures infrastructure.
- Writes tests, test suites, or test fixtures (no pytest, unittest, jest, etc.).
- Creates Dockerfiles, docker-compose files, CI/CD pipelines, or deployment configs.
- Writes documentation, README files, or comments only.
- Is vague — if you cannot name a concrete output file for the task, omit it.
- Duplicates work already covered by another task in the list.

FIELD RULES:
- id: sequential integers starting at 1 (or max(completed_ids)+1 if resuming).
- priority: integer 1-5, where 1 = highest priority (runs first).
- depends_on: list of task IDs that must complete before this one starts.
- outputs: list of relative file paths this task produces. Must be non-empty.
- Do NOT include tasks already listed as completed.
"""


# ── Agent node ──────────────────────────────────────────────────────────────────

def task_manager_node(state: ProjectState) -> ProjectState:
    """
    LangGraph node: Task Manager Agent.

    Transforms state:
        state.architecture  ->  state.tasks + state.active_task_id
                                + state.task_history + task_board.json on disk

    Context-aware: if state already has tasks (refinement / resume scenario),
    the LLM receives current task context and completed tasks are never touched.
    """
    logger.info(
        "[TaskManagerAgent] Building task list for project '%s' "
        "(%d existing tasks, %d completed).",
        state["project_id"],
        len(state.get("tasks", [])),
        len(state.get("completed_task_ids", [])),
    )

    if is_mock_mode():
        return mock_task_manager_node(state)

    provider = get_provider()
    user_message = _build_user_message(state)

    llm_tasks = None
    last_error = None
    for attempt in range(1, MAX_PLAN_RETRIES + 2):  # 1 .. MAX_PLAN_RETRIES+1
        try:
            raw = provider.complete(
                system=SYSTEM_PROMPT,
                user=user_message,
                max_tokens=2048,
            )
            logger.debug("[TaskManagerAgent] Raw response (attempt %d):\n%s", attempt, raw)

            json_str = _extract_json(raw)
            parsed = json.loads(json_str)
            if "tasks" not in parsed or not isinstance(parsed["tasks"], list):
                raise ValueError("Response JSON missing 'tasks' list.")
            llm_tasks = parsed["tasks"]
            break

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[TaskManagerAgent] Attempt %d/%d failed to parse (%s). %s.",
                attempt,
                MAX_PLAN_RETRIES + 1,
                exc,
                "Retrying" if attempt <= MAX_PLAN_RETRIES else "Giving up",
            )

    if llm_tasks is None:
        raise RuntimeError(
            f"TaskManagerAgent failed to produce valid JSON after "
            f"{MAX_PLAN_RETRIES + 1} attempt(s). Last error: {last_error}"
        )

    # ── Merge: protect completed / failed / blocked tasks ──────────────────────
    protected_ids = set(
        state.get("completed_task_ids", [])
        + state.get("failed_task_ids", [])
        + state.get("blocked_task_ids", [])
    )
    tasks = _merge_tasks(
        existing_tasks=state.get("tasks", []),
        llm_tasks=llm_tasks,
        protected_ids=protected_ids,
    )

    # ── Filter: deterministically remove non-coding tasks ──────────────────────
    raw_count = len(tasks)
    tasks, removed_entries = _filter_tasks(tasks, protected_ids)
    removed_ids = {e["task"]["id"] for e in removed_entries}
    if removed_entries:
        logger.info(
            "[TaskManagerAgent] Filter removed %d task(s) (kept %d/%d):",
            len(removed_entries), len(tasks), raw_count,
        )
        for entry in removed_entries:
            logger.info(
                "  REMOVED task id=%d '%s' — %s",
                entry["task"]["id"], entry["task"].get("title", "?"), entry["reason"],
            )
    tasks = _repair_dependencies(tasks, removed_ids)

    # ── Validate (warns; does not raise) ───────────────────────────────────────
    _validate_tasks(tasks, protected_ids)

    # ── Schedule first runnable task ───────────────────────────────────────────
    first_task = get_next_runnable_task(tasks)
    active_id = first_task["id"] if first_task else None
    first_idx = (
        next((i for i, t in enumerate(tasks) if t["id"] == active_id), 0)
        if active_id is not None
        else 0
    )

    history = list(state.get("task_history", []))
    if active_id is not None:
        history.append(task_history_event(active_id, "started"))

    logger.info(
        "[TaskManagerAgent] %d total task(s) (%d protected + %d from LLM, "
        "%d filtered out). First active task id=%s.",
        len(tasks),
        len(protected_ids),
        len(tasks) - len(protected_ids),
        len(removed_entries),
        active_id,
    )

    new_state = {
        **state,
        "tasks": tasks,
        "active_task_id": active_id,
        "current_task_index": first_idx,
        "task_history": history,
        "status": "coding",
    }

    persist_task_board(new_state)
    return new_state


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _build_user_message(state: ProjectState) -> str:
    """
    Build the user-turn message sent to the LLM.

    Sections (all optional except idea + architecture):
      1. Memory context (past architectures — injected by MemoryAgent)
      2. Project idea + architecture document
      3. Current task state (only when existing tasks are present)
         - Completed tasks with IDs (LLM must avoid these IDs)
         - Failed tasks with last error snippet
         - Remaining pending tasks (LLM may keep or refine)
    """
    lines = []

    # Note: memory_context (past architectures) is intentionally NOT injected here.
    # It is already used by PlanningAgent to inform the architecture document.
    # Including it again in the task manager prompt would double the input length
    # (~+500 tokens) and push CPU prefill time past the socket timeout on local
    # inference. Task decomposition quality does not depend on raw past architectures —
    # it depends on the current architecture, which is always present.

    lines.append(f"Project idea: {state['idea']}")
    lines.append("")
    lines.append(f"Architecture:\n{state['architecture']}")

    existing_tasks = state.get("tasks", [])
    if not existing_tasks:
        return "\n".join(lines)

    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))
    blocked_ids = set(state.get("blocked_task_ids", []))

    lines.append("\n\n--- Current task state ---")

    done_tasks = [t for t in existing_tasks if t["id"] in completed_ids]
    if done_tasks:
        taken = sorted(completed_ids)
        lines.append(
            f"\nCompleted tasks (IDs {taken} are taken — do NOT include these):"
        )
        for t in done_tasks:
            lines.append(f"  [{t['id']}] {t['title']}")
        max_id = max(t["id"] for t in existing_tasks)
        lines.append(f"\nStart any new task IDs from {max_id + 1} to avoid conflicts.")

    failed_tasks = [t for t in existing_tasks if t["id"] in failed_ids]
    if failed_tasks:
        lines.append(
            "\nFailed tasks (exhausted retries — redesign or replace them):"
        )
        for t in failed_tasks:
            err = t.get("error_context", "")
            lines.append(f"  [{t['id']}] {t['title']}")
            if err:
                lines.append(f"       last error: {err[:120]}")

    pending_tasks = [
        t for t in existing_tasks
        if t["id"] not in completed_ids
        and t["id"] not in failed_ids
        and t["id"] not in blocked_ids
    ]
    if pending_tasks:
        lines.append(
            "\nExisting pending tasks (include these or refine them if needed):"
        )
        for t in pending_tasks:
            lines.append(
                f"  [{t['id']}] {t['title']}"
                f"  priority={t.get('priority', 5)}"
                f"  depends_on={t.get('depends_on', [])}"
            )

    return "\n".join(lines)


def _merge_tasks(
    existing_tasks: list,
    llm_tasks: list,
    protected_ids: set,
) -> list:
    """
    Merge LLM-generated tasks with the existing task list.

    Rules:
    1. Tasks whose IDs are in protected_ids are preserved verbatim from
       existing_tasks regardless of what the LLM returned.
    2. LLM tasks whose IDs match protected_ids are dropped (with a warning).
    3. All other LLM tasks are converted to fresh TaskItems and appended.
    4. Final order: protected tasks (original order) then new LLM tasks.

    Returns a new list — does not mutate either input.
    """
    skipped = [lt["id"] for lt in llm_tasks if lt["id"] in protected_ids]
    if skipped:
        logger.warning(
            "[TaskManagerAgent] LLM returned %d protected task(s) (ids=%s) — "
            "keeping existing versions to preserve completed work.",
            len(skipped),
            skipped,
        )

    new_llm_tasks = [lt for lt in llm_tasks if lt["id"] not in protected_ids]

    result = [t for t in existing_tasks if t["id"] in protected_ids]

    for lt in new_llm_tasks:
        result.append(
            make_task_item(
                id=lt["id"],
                title=lt["title"],
                description=lt["description"],
                priority=lt.get("priority", 5),
                depends_on=lt.get("depends_on", []),
                outputs=lt.get("outputs", []),
            )
        )

    return result


def _validate_tasks(tasks: list, protected_ids: set) -> None:
    """
    Validate the merged task list. Logs warnings; never raises.

    Checks:
    - No duplicate IDs
    - All depends_on references resolve to a task in the list
    - Non-protected tasks have non-empty title, description, and priority
    """
    all_ids = {t["id"] for t in tasks}
    seen: set = set()

    for t in tasks:
        tid = t["id"]

        if tid in seen:
            logger.warning(
                "[TaskManagerAgent] Duplicate task id=%d — second occurrence ignored "
                "by scheduler but may cause unexpected behaviour.",
                tid,
            )
        seen.add(tid)

        for dep in t.get("depends_on", []):
            if dep not in all_ids:
                logger.warning(
                    "[TaskManagerAgent] Task id=%d depends_on id=%d which does not "
                    "exist in the task list. This task will never become runnable.",
                    tid,
                    dep,
                )

        if tid not in protected_ids:
            for field in ("title", "description", "priority"):
                if not t.get(field):
                    logger.warning(
                        "[TaskManagerAgent] Task id=%d has missing or empty '%s'.",
                        tid,
                        field,
                    )


def _classify_task(task: dict) -> str | None:
    """
    Return a rejection reason string if the task should be filtered out,
    or None if the task is acceptable.

    Rejection criteria (applied in order):
    1. No output files listed.
    2. Output files are test files (by path pattern).
    3. All output files are deploy/config files (by path pattern).
    4. Title contains setup/environment keywords.
    5. Title contains testing keywords.
    6. Title contains deployment/devops keywords.
    """
    title = task.get("title", "").lower()
    outputs = [o.lower() for o in task.get("outputs", [])]
    title_words = set(re.findall(r"\w+", title))

    if not outputs:
        return "no output files listed"

    if any(any(p in o for p in _TEST_OUTPUT_PATTERNS) for o in outputs):
        return f"outputs include test files: {task.get('outputs')}"

    if outputs and all(any(p in o for p in _DEPLOY_OUTPUT_PATTERNS) for o in outputs):
        return f"outputs are deploy/config files: {task.get('outputs')}"

    matched_setup = title_words & _SETUP_WORDS
    if matched_setup:
        return f"setup/environment task (title matched: {sorted(matched_setup)})"

    matched_test = title_words & _TEST_WORDS
    if matched_test:
        return f"testing task (title matched: {sorted(matched_test)})"

    matched_deploy = title_words & _DEPLOY_WORDS
    if matched_deploy:
        return f"deployment/devops task (title matched: {sorted(matched_deploy)})"

    return None


def _remove_output_duplicates(tasks: list, protected_ids: set) -> tuple[list, list]:
    """
    Remove tasks whose output files are a subset of outputs already covered
    by an earlier task in the list (deduplication by output overlap).
    Protected tasks are never removed.

    Returns (kept, removed_entries).
    """
    seen_outputs: set = set()
    kept: list = []
    removed: list = []

    for task in tasks:
        if task["id"] in protected_ids:
            seen_outputs.update(o.lower() for o in task.get("outputs", []))
            kept.append(task)
            continue

        task_outputs = {o.lower() for o in task.get("outputs", [])}
        overlap = task_outputs & seen_outputs
        if task_outputs and overlap == task_outputs:
            removed.append({
                "task": task,
                "reason": f"duplicate outputs already covered: {sorted(overlap)}",
            })
        else:
            seen_outputs.update(task_outputs)
            kept.append(task)

    return kept, removed


def _filter_tasks(tasks: list, protected_ids: set) -> tuple[list, list]:
    """
    Remove non-coding tasks from the list.

    Rules (deterministic — no LLM):
    - Protected tasks (completed/failed/blocked) are always kept.
    - Non-protected tasks are tested against _classify_task():
        * setup/install/environment tasks → removed
        * test/testing tasks → removed
        * deploy/docker/devops tasks → removed
        * tasks with no output files → removed
    - Output deduplication: tasks whose outputs are already fully covered
      by an earlier task are removed.

    Returns (kept_tasks, removed_entries) where each removed entry is:
        {"task": <task_dict>, "reason": <str>}
    """
    kept: list = []
    removed: list = []

    for task in tasks:
        if task["id"] in protected_ids:
            kept.append(task)
            continue
        reason = _classify_task(task)
        if reason:
            removed.append({"task": task, "reason": reason})
        else:
            kept.append(task)

    # Second pass: dedup by output overlap
    kept, dup_removed = _remove_output_duplicates(kept, protected_ids)
    removed.extend(dup_removed)

    return kept, removed


def _repair_dependencies(tasks: list, removed_ids: set) -> list:
    """
    Remove references to deleted task IDs from every task's depends_on list.
    Returns a new list; does not mutate input.
    """
    result = []
    for task in tasks:
        deps = task.get("depends_on", [])
        cleaned = [d for d in deps if d not in removed_ids]
        if cleaned != deps:
            logger.warning(
                "[TaskManagerAgent] Task id=%d: removed %d broken dep(s) %s "
                "(those tasks were filtered out).",
                task["id"],
                len(deps) - len(cleaned),
                sorted(set(deps) - set(cleaned)),
            )
            task = {**task, "depends_on": cleaned}
        result.append(task)
    return result


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
