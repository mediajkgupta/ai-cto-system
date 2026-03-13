"""
graph.py — LangGraph StateGraph wiring for the AI CTO System.

Graph topology (V1 + Memory + Verification):

    START
      └─► memory_retrieve        ← query vector DB for relevant past experience
              └─► planning        ← architecture + task list (memory-informed)
                      └─► coding
                              └─► verification   ← static checks + test generation
                                      ├─► [errors]   → debug
                                      │                   ├─► [retry]      → coding
                                      │                   └─► [exhausted]  → END (failed)
                                      └─► [clean]    → execution
                                                          ├─► [success] → advance_task
                                                          │                   ├─► [more tasks] → coding
                                                          │                   └─► [all done]   → memory_save → END
                                                          └─► [failure] → debug

Controller logic lives entirely in the conditional edge functions below.
No LLM calls occur in routing — agents only transform state.
"""

import logging
from langgraph.graph import StateGraph, END

from ai_cto.state import ProjectState
from ai_cto.agents.planning import planning_node
from ai_cto.agents.coding import coding_node
from ai_cto.agents.verification import verification_node
from ai_cto.agents.execution import execution_node
from ai_cto.agents.debug import debug_node
from ai_cto.agents.memory import memory_retrieve_node, memory_save_node

logger = logging.getLogger(__name__)


# ── Controller edge functions (routing logic) ──────────────────────────────────

def route_after_verification(state: ProjectState) -> str:
    """
    Decide what happens after the VerificationAgent runs.

    Returns:
        "debug"     → ERROR-level issues found; DebugAgent must fix them
        "execution" → clean (or warnings only); safe to run
    """
    if state["status"] == "verifying_failed":
        logger.info("[Controller] Verification failed → routing to DebugAgent.")
        return "debug"
    logger.info("[Controller] Verification passed → routing to ExecutionAgent.")
    return "execution"


def route_after_execution(state: ProjectState) -> str:
    """
    Decide what happens after the ExecutionAgent runs.

    Returns:
        "debug"          → execution failed, attempts remaining
        "advance_task"   → execution succeeded
    """
    if state["status"] == "debugging":
        logger.info("[Controller] Execution failed → routing to DebugAgent.")
        return "debug"

    logger.info("[Controller] Execution succeeded → advancing task.")
    return "advance_task"


def route_after_debug(state: ProjectState) -> str:
    """
    Decide what happens after the DebugAgent runs.

    Returns:
        "coding"   → debug produced a patch, retry execution
        "end"      → max retries exceeded, halt with failure
    """
    if state["status"] == "failed":
        logger.error("[Controller] Debug exhausted. Ending with failure.")
        return "end"

    logger.info("[Controller] Debug patch ready → routing back to CodingAgent.")
    return "coding"


# ── Internal node: advance to next task ───────────────────────────────────────

def advance_task_node(state: ProjectState) -> ProjectState:
    """
    Mark the current task as done and decide whether to continue or finish.

    Runs after a successful execution.
    """
    tasks = list(state["tasks"])
    idx = state["current_task_index"]

    tasks[idx] = {**tasks[idx], "status": "done"}
    logger.info("[Controller] Task %d/%d completed.", idx + 1, len(tasks))

    next_idx = idx + 1

    if next_idx < len(tasks):
        logger.info("[Controller] Moving to task %d.", next_idx + 1)
        return {
            **state,
            "tasks": tasks,
            "current_task_index": next_idx,
            "debug_attempts": 0,
            "error_context": "",
            "status": "coding",
        }
    else:
        logger.info("[Controller] All tasks complete. Saving to memory.")
        summary = _build_summary(state, tasks)
        return {
            **state,
            "tasks": tasks,
            "status": "saving_memory",   # signal memory_save_node to run
            "final_output": summary,
        }


def route_after_advance(state: ProjectState) -> str:
    """Route after advance_task_node."""
    if state["status"] == "saving_memory":
        return "memory_save"
    return "coding"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready to invoke with an initial state dict.
    """
    graph = StateGraph(ProjectState)

    # Register nodes
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("planning", planning_node)
    graph.add_node("coding", coding_node)
    graph.add_node("verification", verification_node)
    graph.add_node("execution", execution_node)
    graph.add_node("debug", debug_node)
    graph.add_node("advance_task", advance_task_node)
    graph.add_node("memory_save", memory_save_node)

    # Entry point: always retrieve memories first
    graph.set_entry_point("memory_retrieve")

    # Linear edges
    graph.add_edge("memory_retrieve", "planning")
    graph.add_edge("planning", "coding")
    graph.add_edge("coding", "verification")

    # Conditional: after verification
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "debug": "debug",
            "execution": "execution",
        },
    )

    # Conditional: after execution
    graph.add_conditional_edges(
        "execution",
        route_after_execution,
        {
            "debug": "debug",
            "advance_task": "advance_task",
        },
    )

    # Conditional: after debug
    graph.add_conditional_edges(
        "debug",
        route_after_debug,
        {
            "coding": "coding",
            "end": END,
        },
    )

    # Conditional: after task advance — more work or save + finish
    graph.add_conditional_edges(
        "advance_task",
        route_after_advance,
        {
            "coding": "coding",
            "memory_save": "memory_save",
        },
    )

    # Memory save always ends the graph
    graph.add_edge("memory_save", END)

    return graph.compile()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_summary(state: ProjectState, tasks: list) -> str:
    completed = [t for t in tasks if t["status"] == "done"]
    lines = [
        f"Project '{state['project_id']}' completed successfully.",
        f"Idea: {state['idea']}",
        f"Tasks completed: {len(completed)}/{len(tasks)}",
        "",
        "Files generated:",
        *[f"  - {f}" for f in state["generated_files"].keys()],
    ]
    return "\n".join(lines)
