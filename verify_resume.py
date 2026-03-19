#!/usr/bin/env python
"""
verify_resume.py — End-to-end resume proof in MOCK_LLM mode.

Phase 1: Start a 5-task project. Run the pipeline until exactly 3 tasks
         complete, then stop (simulating a process crash mid-run).
         Print the saved task_board.json at that point.

Phase 2: Call --resume logic. Prove the system:
         - Loads the existing task_board.json (not a fresh state)
         - Skips already-completed tasks 1, 2, 3
         - Resumes from task 4 (next runnable)
         - Completes tasks 4 and 5
         - Shows the final completed state

Run: python verify_resume.py
"""

import os
import sys
import json
import logging
import shutil
import tempfile
from pathlib import Path

# ── Activate mock mode BEFORE importing any ai_cto modules ────────────────────
os.environ["AI_CTO_MOCK_LLM"] = "true"
os.environ["AI_CTO_MOCK_FIVE"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)

# Windows-safe stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Silence verbose chromadb / langchain noise
for noisy in ("chromadb", "langchain", "httpx", "httpcore", "anthropic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SEP  = "=" * 64
SEP2 = "-" * 64

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subsection(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


# ── Setup isolated workspace ───────────────────────────────────────────────────

import ai_cto.tools.file_writer as _fw
_WORKSPACE = Path(tempfile.mkdtemp(prefix="ai_cto_resume_demo_"))
_fw.WORKSPACE_ROOT = _WORKSPACE
PROJECT_ID = "saas-crm-demo"

print(f"\nWorkspace: {_WORKSPACE}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Run until 3 tasks complete, then stop
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 1  Starting 5-task project (will stop after task 3 completes)")

from ai_cto.state import initial_state
from ai_cto.graph import build_graph

IDEA = "Build a SaaS CRM with contacts, deals, and pipeline management"
state = initial_state(idea=IDEA, project_id=PROJECT_ID)
graph = build_graph()

interrupted_after = None
final_phase1_state = None
node_count = 0

print(f"\nIdea: {IDEA}")
print(f"Project ID: {PROJECT_ID}\n")

for event in graph.stream(state):
    node_name = list(event.keys())[0]
    node_state = event[node_name]
    node_count += 1

    completed = node_state.get("completed_task_ids", [])
    active = node_state.get("active_task_id")
    status = node_state.get("status", "?")

    print(f"  [{node_name.upper():20s}] status={status:18s} "
          f"completed={completed}  active_task={active}")

    # Stop as soon as 3 tasks are done (after advance_task persists the board)
    if node_name == "advance_task" and len(completed) == 3:
        interrupted_after = node_state
        print(f"\n  *** SIMULATED CRASH: stopping after {len(completed)} tasks complete ***")
        break

# Show task plan from the first task_manager event so we can verify all 5 tasks exist
board_path = _WORKSPACE / PROJECT_ID / "task_board.json"

# ── 1a. Show the 5-task plan ────────────────────────────────────────────────
subsection("1a. Task plan produced by TaskManagerAgent")
if board_path.exists():
    board = json.loads(board_path.read_text(encoding="utf-8"))
    print(f"\nproject_id   : {board['project_id']}")
    print(f"project_goal : {board['project_goal'][:70]}...")
    print(f"\nAll 5 tasks:")
    for t in board["tasks"]:
        marker = "[DONE]" if t["status"] == "done" else "[pending]"
        dep = f" depends_on={t['depends_on']}" if t['depends_on'] else ""
        print(f"  {marker}  T{t['id']} (priority={t['priority']}{dep}): {t['title']}")

# ── 1b. Show task_board.json after interruption ─────────────────────────────
subsection("1b. task_board.json state at point of simulated crash")
if board_path.exists():
    board = json.loads(board_path.read_text(encoding="utf-8"))
    print(f"\nproject_status    : {board['project_status']}")
    print(f"active_task_id    : {board['active_task_id']}  (next task that would run)")
    print(f"completed_task_ids: {board['completed_task_ids']}")
    print(f"failed_task_ids   : {board['failed_task_ids']}")
    print(f"architecture saved: {'yes (' + str(len(board.get('architecture', ''))) + ' chars)' if board.get('architecture') else 'NO -- bug'}")
    print(f"\nPer-task statuses:")
    for t in board["tasks"]:
        print(f"  T{t['id']} [{t['status']:10s}] retries={t['retries']}  '{t['title']}'")
    print(f"\nLast 4 history events:")
    for ev in board["task_history"][-4:]:
        print(f"  {ev['timestamp'][:19]}  T{ev['task_id']} {ev['event']:8s}  {ev.get('detail', '')[:40]}")
    print(f"\npersisted_at: {board['persisted_at']}")
else:
    print("ERROR: board not found — Phase 1 did not write a board")
    sys.exit(1)

# ── 1c. Show generated workspace files ──────────────────────────────────────
subsection("1c. Workspace files on disk after crash")
project_dir = _WORKSPACE / PROJECT_ID
for p in sorted(project_dir.rglob("*")):
    if p.is_file() and p.name != "task_board.json" and "__pycache__" not in str(p):
        rel = p.relative_to(project_dir)
        size = p.stat().st_size
        print(f"  {str(rel):30s}  {size:5d} bytes")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Resume from task_board.json
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 2  Resuming from task_board.json")

from ai_cto.tools.task_board import resume_state_from_board, load_task_board
from ai_cto.graph import build_resume_graph

print(f"\nLoading: {board_path}")

# ── 2a. Prove it loads from disk, not fresh state ────────────────────────────
board_before_resume = load_task_board(board_path)
subsection("2a. Board loaded from disk (not a fresh state)")
print(f"\nproject_id         : {board_before_resume['project_id']}")
print(f"completed_task_ids : {board_before_resume['completed_task_ids']}  <- tasks already done, will be skipped")
print(f"active_task_id     : {board_before_resume['active_task_id']}        <- resume entry point")
print(f"architecture chars : {len(board_before_resume.get('architecture', ''))}  (persisted in board, restored on resume)")

# ── 2b. resume_state_from_board reconstructs the state ──────────────────────
resumed_state = resume_state_from_board(board_path)

subsection("2b. State reconstructed by resume_state_from_board()")
print(f"\nresumed_state['status']           : {resumed_state['status']}")
print(f"resumed_state['active_task_id']   : {resumed_state['active_task_id']}  <- picks up at T4, skips T1-T3")
print(f"resumed_state['completed_task_ids']: {resumed_state['completed_task_ids']}  <- preserved from board")
print(f"resumed_state['architecture'][:60]: {resumed_state['architecture'][:60]}...")
print(f"resumed_state['generated_files']  : {list(resumed_state['generated_files'].keys())}  <- scanned from disk")
print(f"\nPer-task statuses in resumed state:")
for t in resumed_state["tasks"]:
    print(f"  T{t['id']} [{t['status']:10s}] retries={t['retries']}  '{t['title']}'")
print(f"\nLast history event: {resumed_state['task_history'][-1]}")

# Verify: resume started at T4, not T1
assert resumed_state["active_task_id"] == 4, \
    f"Expected active_task_id=4 but got {resumed_state['active_task_id']}"
assert resumed_state["completed_task_ids"] == [1, 2, 3], \
    f"Expected completed=[1,2,3] but got {resumed_state['completed_task_ids']}"
assert resumed_state["architecture"], "Architecture must be non-empty after resume"
print("\n  [OK] active_task_id=4  (skipped T1-T3)")
print("  [OK] completed_task_ids=[1,2,3] preserved")
print("  [OK] architecture restored from board")

# ── 2c. Run the resume graph ─────────────────────────────────────────────────
subsection("2c. Running build_resume_graph() to complete tasks 4 and 5")
print()

resume_graph = build_resume_graph()
final_state = None
tasks_completed_in_resume = []

for event in resume_graph.stream(resumed_state):
    node_name = list(event.keys())[0]
    node_state = event[node_name]

    completed = node_state.get("completed_task_ids", [])
    active = node_state.get("active_task_id")
    status = node_state.get("status", "?")

    print(f"  [{node_name.upper():20s}] status={status:18s} "
          f"completed={completed}  active_task={active}")

    # Track which tasks were completed during this resume run
    if node_name == "advance_task":
        new = set(completed) - {1, 2, 3}
        for tid in sorted(new):
            if tid not in tasks_completed_in_resume:
                tasks_completed_in_resume.append(tid)

    final_state = node_state


# ── 2d. Final completed state ────────────────────────────────────────────────
subsection("2d. Final completed state")
if final_state:
    print(f"\nstatus          : {final_state.get('status')}")
    print(f"project_status  : {final_state.get('project_status')}")
    print(f"active_task_id  : {final_state.get('active_task_id')}  (None = no active task)")
    print(f"completed_task_ids: {final_state.get('completed_task_ids')}")
    print(f"tasks_completed_in_this_resume_run: {tasks_completed_in_resume}")
    print(f"\ngenerated_files : {list(final_state.get('generated_files', {}).keys())}")
    print(f"\nfinal_output:\n")
    print(final_state.get("final_output", "(none)"))

# ── 2e. Final task_board.json ────────────────────────────────────────────────
subsection("2e. Final task_board.json on disk")
if board_path.exists():
    board = json.loads(board_path.read_text(encoding="utf-8"))
    print(f"\nproject_status    : {board['project_status']}")
    print(f"active_task_id    : {board['active_task_id']}")
    print(f"completed_task_ids: {board['completed_task_ids']}")
    print(f"\nPer-task final statuses:")
    for t in board["tasks"]:
        print(f"  T{t['id']} [{t['status']:10s}] retries={t['retries']}  '{t['title']}'")

# ── Assertions ───────────────────────────────────────────────────────────────
section("ASSERTIONS")

errors = []

if final_state and final_state.get("status") == "done":
    print("  [PASS] Final status = 'done'")
else:
    errors.append(f"Final status expected 'done', got {final_state.get('status') if final_state else 'None'}")

if final_state and final_state.get("project_status") == "complete":
    print("  [PASS] project_status = 'complete'")
else:
    errors.append(f"project_status expected 'complete'")

if final_state and set(final_state.get("completed_task_ids", [])) == {1, 2, 3, 4, 5}:
    print("  [PASS] All 5 tasks completed")
else:
    errors.append(f"Expected completed={[1,2,3,4,5]}, got {final_state.get('completed_task_ids') if final_state else None}")

if final_state and final_state.get("active_task_id") is None:
    print("  [PASS] active_task_id = None (cleared on completion)")
else:
    errors.append(f"active_task_id should be None, got {final_state.get('active_task_id') if final_state else None}")

if tasks_completed_in_resume == [4, 5]:
    print("  [PASS] Resume completed only tasks 4 and 5 (skipped 1, 2, 3)")
else:
    errors.append(f"Resume should have completed [4,5], got {tasks_completed_in_resume}")

board_final = json.loads(board_path.read_text(encoding="utf-8")) if board_path.exists() else {}
if board_final.get("project_status") == "complete":
    print("  [PASS] task_board.json project_status = 'complete'")
else:
    errors.append("task_board.json not marked complete")

if board_final.get("active_task_id") is None:
    print("  [PASS] task_board.json active_task_id = null")
else:
    errors.append(f"task_board.json active_task_id should be null, got {board_final.get('active_task_id')}")

if errors:
    print(f"\n  FAILURES ({len(errors)}):")
    for e in errors:
        print(f"    [FAIL] {e}")
    sys.exit(1)
else:
    print(f"\n  All assertions passed.")

# ── Cleanup ──────────────────────────────────────────────────────────────────
shutil.rmtree(_WORKSPACE, ignore_errors=True)
print(f"\n[Workspace cleaned up]\n")
