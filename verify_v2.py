"""
verify_v2.py -- Live verification driver for Task Manager V2.

Usage:
    python verify_v2.py
"""

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Force mock mode before any ai_cto imports
os.environ["AI_CTO_MOCK_LLM"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)

import logging
logging.basicConfig(level=logging.WARNING)


def section(title):
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def sub(label):
    print(f"\n  --- {label}")

def show_json(data, indent=4):
    raw = json.dumps(data, indent=2, default=str)
    for line in raw.splitlines():
        print(" " * indent + line)


# ===========================================================================
# SCENARIO 2 + 3 : 4-task MOCK_LLM pipeline run
# ===========================================================================

section("SCENARIO 2+3 : 4-Task MOCK_LLM Pipeline Run")

os.environ["AI_CTO_MOCK_EXTENDED"] = "true"

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    import ai_cto.tools.file_writer as _fw
    _fw.WORKSPACE_ROOT = tmp

    from ai_cto.state import initial_state
    from ai_cto.graph import build_graph

    project_id = "crm-demo"
    state = initial_state(
        idea="Build a SaaS CRM with contacts and deals",
        project_id=project_id,
    )
    graph = build_graph()

    sub("Pipeline event stream (node -> status -> done_ids -> project_status):")
    final = None
    task_seen = set()

    for event in graph.stream(state):
        node = list(event.keys())[0]
        s = event[node]
        active = s.get("active_task_id")
        tasks = s.get("tasks", [])
        done_ids = s.get("completed_task_ids", [])
        proj = s.get("project_status", "?")

        if active and active not in task_seen:
            task_seen.add(active)
            t = next((t for t in tasks if t["id"] == active), None)
            if t:
                print(f"\n    >> TASK {active}/{len(tasks)}"
                      f"  [{t['status'].upper()}]  \"{t['title']}\"")
                print(f"       depends_on={t['depends_on']}  priority={t['priority']}"
                      f"  retries={t['retries']}/{t['max_retries']}")

        print(f"       [{node:<20}]  status={s.get('status','?'):<18}"
              f"  done={done_ids}  project={proj}")
        final = s

    sub("Final state summary:")
    print(f"    project_status   : {final['project_status']}")
    print(f"    routing status   : {final['status']}")
    print(f"    completed_ids    : {final['completed_task_ids']}")
    print(f"    failed_ids       : {final['failed_task_ids']}")
    print(f"    blocked_ids      : {final['blocked_task_ids']}")
    print(f"    task_history len : {len(final['task_history'])} events")
    print(f"    debug retries    : {sum(t.get('retries',0) for t in final['tasks'])}")
    print()
    print("    final_output:")
    for line in final["final_output"].splitlines():
        print(f"      {line}")

    sub("SCENARIO 2 -- task_board.json (full):")
    board_path = tmp / project_id / "task_board.json"
    show_json(json.loads(board_path.read_text()))

os.environ.pop("AI_CTO_MOCK_EXTENDED", None)


# ===========================================================================
# SCENARIO 4 : forced per-task failure (retries exhausted)
# ===========================================================================

section("SCENARIO 4 : Forced Per-Task Failure (per-task retries exhausted)")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    _fw.WORKSPACE_ROOT = tmp

    from ai_cto.state import (
        initial_state, make_task_item, get_active_task, update_task_in_list,
    )
    from ai_cto.agents.debug import debug_node

    t1 = make_task_item(
        id=1, title="Build auth module",
        description="JWT-based auth", max_retries=2,
    )
    t1 = {**t1, "status": "in_progress"}

    state = {
        **initial_state("Build a SaaS CRM", "fail-demo"),
        "tasks": [t1],
        "active_task_id": 1,
        "current_task_index": 0,
        "generated_files": {"main.py": "print('stub')"},
        "error_context": "NameError: name 'jwt' is not defined on line 12",
    }

    sub("Initial task:")
    print(f"    id={t1['id']}  title=\"{t1['title']}\"")
    print(f"    retries={t1['retries']}  max_retries={t1['max_retries']}  status={t1['status']}")
    print()
    print("    Note: T2 and T3 do NOT exist -- only task 1 is in the list.")

    for attempt in range(1, 6):
        print(f"\n    -- debug_node call #{attempt} "
              f"(task.retries before call = {state['tasks'][0].get('retries',0)}) --")

        result = debug_node(state)

        active = get_active_task(result) or next(
            (t for t in result["tasks"] if t["id"] == 1), None
        )
        print(f"    task.retries after  : {active['retries'] if active else '?'}")
        print(f"    task.status         : {active['status'] if active else '?'}")
        print(f"    routing status      : {result['status']}")
        print(f"    project_status      : {result['project_status']}")

        if result["status"] == "failed":
            print(f"\n    *** FAILURE TRIGGERED on attempt #{attempt}")
            print(f"    1 in failed_task_ids : {1 in result['failed_task_ids']}")
            print(f"    task error_context   : {active['error_context'][:60] if active else '?'}...")
            print()
            print("    final_output:")
            for line in result["final_output"].splitlines():
                print(f"      {line}")

            sub("task_board.json at failure:")
            show_json(json.loads((tmp / "fail-demo" / "task_board.json").read_text()))
            break

        # Simulate the retry loop: feed result back with error still set
        state = {
            **result,
            "error_context": "NameError: name 'jwt' is not defined on line 12",
        }


# ===========================================================================
# SCENARIO 5 : blocked dependency demo
# ===========================================================================

section("SCENARIO 5 : Blocked Dependency Demo")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    _fw.WORKSPACE_ROOT = tmp

    from ai_cto.graph import advance_task_node
    from ai_cto.state import (
        initial_state, make_task_item,
        get_next_runnable_task, project_is_blocked,
    )

    # T1 (active), T2 depends on T3, T3 depends on T4 -- T4 does NOT exist
    t1 = {**make_task_item(id=1, title="Scaffold structure",  description="", priority=1), "status": "in_progress"}
    t2 = {**make_task_item(id=2, title="Implement auth",       description="", priority=2, depends_on=[3]), "status": "pending"}
    t3 = {**make_task_item(id=3, title="Set up database",      description="", priority=3, depends_on=[4]), "status": "pending"}
    # T4 referenced but absent -- permanent deadlock for T2 and T3

    state = {
        **initial_state("Build a SaaS CRM", "blocked-demo"),
        "tasks": [t1, t2, t3],
        "active_task_id": 1,
        "current_task_index": 0,
        "generated_files": {},
    }

    sub("Task board BEFORE advance_task_node:")
    for t in state["tasks"]:
        print(f"    [{t['id']}] {t['title']:<30}  status={t['status']:<12}  depends_on={t.get('depends_on')}")

    runnable_before = get_next_runnable_task(state["tasks"])
    print(f"\n    get_next_runnable_task() = task id {runnable_before['id'] if runnable_before else None}"
          f"  (T1 is in_progress, not pending -- returns None)")

    sub("Calling advance_task_node (marks T1 done, then finds no runnable tasks):")
    result = advance_task_node(state)

    tasks_after = result["tasks"]
    runnable_after = get_next_runnable_task(tasks_after)
    blocked = project_is_blocked(tasks_after)

    print(f"    T1 status after advance      : {next(t for t in tasks_after if t['id']==1)['status']}")
    print(f"    T2 status                    : {next(t for t in tasks_after if t['id']==2)['status']}")
    print(f"    T3 status                    : {next(t for t in tasks_after if t['id']==3)['status']}")
    print(f"    get_next_runnable_task()     : {runnable_after}  (None = no eligible task)")
    print(f"    project_is_blocked()         : {blocked}")
    print(f"    result['status']             : {result['status']}")
    print(f"    result['project_status']     : {result['project_status']}")
    print(f"    completed_task_ids           : {result['completed_task_ids']}")

    sub("final_output:")
    for line in result["final_output"].splitlines():
        print(f"    {line}")

    sub("task_board.json at blocked state:")
    show_json(json.loads((tmp / "blocked-demo" / "task_board.json").read_text()))


print(f"\n{'=' * 68}")
print("  All verification scenarios completed successfully.")
print(f"{'=' * 68}\n")
