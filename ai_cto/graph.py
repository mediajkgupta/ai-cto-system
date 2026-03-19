"""
graph.py — LangGraph StateGraph wiring for the AI CTO System (V2).

Graph topology:

    START
      └─► memory_retrieve        ← query vector DB for relevant past experience
              └─► planning        ← architecture document (memory-informed)
                      └─► task_manager  ← structured task list + initial board persist
                              └─► coding
                                      └─► verification
                                              ├─► [errors]   → debug
                                              │                   ├─► [retry]      → coding
                                              │                   └─► [exhausted]  → END (failed)
                                              └─► [clean]    → execution
                                                                  ├─► [success] → advance_task
                                                                  │                   ├─► [more tasks]  → coding
                                                                  │                   ├─► [all done]    → memory_save → END
                                                                  │                   └─► [dep blocked] → END (blocked)
                                                                  └─► [failure] → debug

V2 changes:
  - advance_task_node uses per-task helpers from task_board.py
  - Blocked-project detection: if pending tasks exist but none can run,
    the project halts with status="blocked"
  - task_board.json persisted after every task advance
  - task_history updated on task done/started events
  - route_after_advance gains a new "end" branch for blocked projects
"""

import logging
from langgraph.graph import StateGraph, END

from ai_cto.state import (
    ProjectState,
    get_active_task,
    get_next_runnable_task,
    task_history_event,
)
from ai_cto.tools.task_board import (
    mark_task_done,
    project_is_complete,
    project_is_blocked,
    persist_task_board,
)
from ai_cto.agents.planning import planning_node
from ai_cto.agents.task_manager import task_manager_node
from ai_cto.agents.coding import coding_node
from ai_cto.agents.verification import verification_node
from ai_cto.agents.execution import execution_node
from ai_cto.agents.debug import debug_node
from ai_cto.agents.memory import memory_retrieve_node, memory_save_node

logger = logging.getLogger(__name__)


# ── Controller edge functions (routing logic) ──────────────────────────────────

def route_after_verification(state: ProjectState) -> str:
    """
    Returns:
        "debug"     → ERROR-level verification issues found
        "execution" → clean or warnings only; safe to run
    """
    if state["status"] == "verifying_failed":
        logger.info("[Controller] Verification failed → routing to DebugAgent.")
        return "debug"
    logger.info("[Controller] Verification passed → routing to ExecutionAgent.")
    return "execution"


def route_after_execution(state: ProjectState) -> str:
    """
    Returns:
        "debug"        → execution failed, attempts remaining
        "advance_task" → execution succeeded
        "end"          → consecutive timeout limit reached; task marked failed
    """
    status = state["status"]
    if status == "failed":
        logger.error("[Controller] Execution hit consecutive timeout limit → ending.")
        return "end"
    if status == "debugging":
        logger.info("[Controller] Execution failed → routing to DebugAgent.")
        return "debug"
    logger.info("[Controller] Execution succeeded → advancing task.")
    return "advance_task"


def route_after_debug(state: ProjectState) -> str:
    """
    Returns:
        "coding" → debug produced a patch, retry execution
        "end"    → max retries exceeded, halt with failure
    """
    if state["status"] == "failed":
        logger.error("[Controller] Debug exhausted. Ending with failure.")
        return "end"
    logger.info("[Controller] Debug patch ready → routing back to CodingAgent.")
    return "coding"


def route_after_advance(state: ProjectState) -> str:
    """
    Returns:
        "memory_save" → all tasks done, persist learnings then END
        "end"         → project is blocked (dep deadlock) or failed
        "coding"      → more tasks remain
    """
    status = state["status"]
    if status == "saving_memory":
        return "memory_save"
    if status in ("blocked", "failed"):
        return "end"
    return "coding"


# ── Internal node: advance to next task (V2) ──────────────────────────────────

def advance_task_node(state: ProjectState) -> ProjectState:
    """
    Mark the active task as done, then select the next unblocked task.

    V2 behaviour:
      1. Uses mark_task_done() to update tasks list, completed_task_ids,
         and task_history atomically.
      2. Persists task_board.json after marking done.
      3. Calls project_is_complete() — if True, transitions to saving_memory.
      4. Calls project_is_blocked() — if True, builds a blocked summary and
         halts with status="blocked".
      5. Otherwise, picks the next runnable task via get_next_runnable_task()
         using priority + dependency scheduling, records a "started" history
         event, resets debug_attempts, and routes back to coding.
    """
    active = get_active_task(state)
    if active is None:
        # Fallback: use current_task_index (guards missing active_task_id)
        idx = state["current_task_index"]
        active = state["tasks"][idx]

    # ── 1. Mark active task done ───────────────────────────────────────────────
    done_updates = mark_task_done(state, active["id"])
    state = {**state, **done_updates}
    tasks = state["tasks"]

    logger.info(
        "[Controller] Task id=%d '%s' completed. %d/%d done.",
        active["id"],
        active["title"],
        len(state["completed_task_ids"]),
        len(tasks),
    )

    # ── Build the final return state, then persist once ───────────────────────

    # ── 2. All tasks complete → save memory ───────────────────────────────────
    if project_is_complete(tasks):
        logger.info("[Controller] All tasks complete. Saving to memory.")
        summary = _build_success_summary(state, tasks)
        result = {
            **state,
            "project_status": "complete",
            "active_task_id": None,   # no active task in terminal state
            "status": "saving_memory",
            "final_output": summary,
        }
        persist_task_board(result)
        return result

    # ── 3. Dependency deadlock → halt with blocked status ─────────────────────
    if project_is_blocked(tasks):
        pending = [t for t in tasks if t["status"] == "pending"]
        logger.warning(
            "[Controller] Project blocked — %d pending task(s) with unmet dependencies.",
            len(pending),
        )
        summary = _build_blocked_summary(state, tasks)
        result = {
            **state,
            "project_status": "blocked",
            "status": "blocked",
            "final_output": summary,
        }
        persist_task_board(result)
        return result

    # ── 4. Advance to next runnable task ──────────────────────────────────────
    next_task = get_next_runnable_task(tasks)
    next_idx = next(
        (i for i, t in enumerate(tasks) if t["id"] == next_task["id"]), 0
    )

    history = list(state.get("task_history", [])) + [
        task_history_event(next_task["id"], "started")
    ]

    logger.info(
        "[Controller] Next task: id=%d '%s' (priority=%d).",
        next_task["id"],
        next_task["title"],
        next_task.get("priority", 5),
    )

    result = {
        **state,
        "active_task_id": next_task["id"],
        "current_task_index": next_idx,
        "task_history": history,
        "debug_attempts": 0,             # reset project-level counter for MemoryAgent
        "execution_timeout_count": 0,    # reset per-task timeout counter
        "error_context": "",
        "status": "coding",
    }
    persist_task_board(result)
    return result


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_resume_graph() -> StateGraph:
    """
    Build a graph for resuming an interrupted project from a task_board.json.

    Identical to build_graph() except the entry point is 'coding' — skipping
    memory_retrieve, planning, and task_manager (which are already done and
    whose results are loaded from the board).

    The state must have been prepared by resume_state_from_board() before
    passing to this graph.
    """
    graph = StateGraph(ProjectState)

    graph.add_node("coding", coding_node)
    graph.add_node("verification", verification_node)
    graph.add_node("execution", execution_node)
    graph.add_node("debug", debug_node)
    graph.add_node("advance_task", advance_task_node)
    graph.add_node("memory_save", memory_save_node)

    graph.set_entry_point("coding")
    graph.add_edge("coding", "verification")

    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"debug": "debug", "execution": "execution"},
    )
    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {"debug": "debug", "advance_task": "advance_task", "end": END},
    )
    graph.add_conditional_edges(
        "debug",
        route_after_debug,
        {"coding": "coding", "end": END},
    )
    graph.add_conditional_edges(
        "advance_task",
        route_after_advance,
        {"coding": "coding", "memory_save": "memory_save", "end": END},
    )
    graph.add_edge("memory_save", END)

    return graph.compile()


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready to invoke with an initial state dict.
    """
    graph = StateGraph(ProjectState)

    # Register nodes
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("planning", planning_node)
    graph.add_node("task_manager", task_manager_node)
    graph.add_node("coding", coding_node)
    graph.add_node("verification", verification_node)
    graph.add_node("execution", execution_node)
    graph.add_node("debug", debug_node)
    graph.add_node("advance_task", advance_task_node)
    graph.add_node("memory_save", memory_save_node)

    # Linear edges
    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "planning")
    graph.add_edge("planning", "task_manager")
    graph.add_edge("task_manager", "coding")
    graph.add_edge("coding", "verification")

    # Conditional: after verification
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"debug": "debug", "execution": "execution"},
    )

    # Conditional: after execution
    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {"debug": "debug", "advance_task": "advance_task", "end": END},
    )

    # Conditional: after debug
    graph.add_conditional_edges(
        "debug",
        route_after_debug,
        {"coding": "coding", "end": END},
    )

    # Conditional: after task advance (V2 adds "end" for blocked projects)
    graph.add_conditional_edges(
        "advance_task",
        route_after_advance,
        {
            "coding": "coding",
            "memory_save": "memory_save",
            "end": END,           # blocked or failed project halts here
        },
    )

    # Memory save always ends the graph
    graph.add_edge("memory_save", END)

    return graph.compile()


# ── Summary builders ───────────────────────────────────────────────────────────

def _build_success_summary(state: ProjectState, tasks: list) -> str:
    completed = [t for t in tasks if t["status"] == "done"]
    total_retries = sum(t.get("retries", 0) for t in tasks)
    lines = [
        f"Project '{state['project_id']}' completed successfully.",
        f"Idea: {state['idea']}",
        f"Tasks completed: {len(completed)}/{len(tasks)}",
        f"Total debug retries: {total_retries}",
        "",
        "Files generated:",
        *[f"  - {f}" for f in state["generated_files"].keys()],
    ]
    return "\n".join(lines)


def _build_blocked_summary(state: ProjectState, tasks: list) -> str:
    pending = [t for t in tasks if t["status"] == "pending"]
    done = [t for t in tasks if t["status"] == "done"]
    lines = [
        f"Project '{state['project_id']}' is BLOCKED.",
        f"Idea: {state['idea']}",
        f"Tasks completed: {len(done)}/{len(tasks)}",
        f"Blocked tasks ({len(pending)} pending with unresolvable dependencies):",
        *[f"  - [{t['id']}] {t['title']}  depends_on={t.get('depends_on', [])}"
          for t in pending],
        "",
        "This usually means a required dependency task failed before completing.",
    ]
    return "\n".join(lines)
