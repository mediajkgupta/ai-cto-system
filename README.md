# AI CTO System

An autonomous multi-agent pipeline that converts a natural language software idea into
running, verified code — no human intervention required after the initial prompt.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 3. Run a project
python -m ai_cto "Build a REST API for a todo app" --id todo-api

# 4. Run without API credits (mock mode — great for testing)
AI_CTO_MOCK_LLM=true python -m ai_cto "Build a REST API" --id todo-mock
```

Generated files land in `workspace/<project_id>/`.
Task progress is written to `workspace/<project_id>/task_board.json` after every state change.

---

## Architecture

```
START
  └─► memory_retrieve     ← query past architectures / debug solutions from ChromaDB
          └─► planning     ← produce architecture document (Claude)
                  └─► task_manager   ← decompose into ordered tasks (Claude)
                          └─► coding          ← implement active task (Claude)
                                  └─► verification   ← 5 static checks + test gen
                                          ├─► [errors]  → debug → coding (retry)
                                          └─► [clean]   → execution
                                                            ├─► [success] → advance_task
                                                            │     ├─► [more tasks]   → coding
                                                            │     ├─► [all done]     → memory_save → END
                                                            │     └─► [dep blocked]  → END (blocked)
                                                            └─► [failure] → debug
```

All agents are pure functions (`state_in → state_out`).
Routing is handled entirely by conditional edge functions — no LLM calls in routing.

---

## Task Manager V2

### How it works

The **Task Manager Agent** runs once per project, immediately after the Planning Agent.
It receives the architecture document and decomposes the project into a structured,
ordered task list.

Each task is assigned:
- A unique integer **id**
- A human-readable **title** and **description**
- A **priority** (1 = highest; lower number runs first)
- A **depends_on** list of task IDs that must complete before this task can start
- An **outputs** list of expected file paths
- Per-task **retries** and **max_retries** counters
- **created_at** and **updated_at** ISO-8601 UTC timestamps

After task creation, the initial `task_board.json` is written to disk.

### Task scheduling

The system always selects the next task using this rule:

1. Task must have `status == "pending"`
2. All IDs in `depends_on` must have `status == "done"`
3. Among eligible candidates, the task with the **lowest priority number** wins
4. Ties are broken by insertion order

This is implemented in `state.get_next_runnable_task()`.

### Task statuses

| Status | Meaning |
|--------|---------|
| `pending` | Waiting to be started |
| `in_progress` | Currently being coded / verified / executed |
| `done` | Completed successfully |
| `failed` | Exhausted per-task retries; project halts |
| `blocked` | Dependencies can never be satisfied (dep deadlock) |

### Per-task retries

Retries are tracked **per task** in `TaskItem.retries`, not at the project level.

- Each time the Debug Agent is invoked for a task, `task.retries` increments
- When `task.retries > task.max_retries`, that task is marked `failed`
- The project halts with `status="failed"` and a clear summary
- `state.debug_attempts` is also incremented for backward compatibility with the
  Memory Agent (which uses it to decide whether to save a debug solution)

### Blocked project detection

After each task completes, the system checks:

1. `project_is_complete(tasks)` — all tasks done → save memory and exit
2. `project_is_blocked(tasks)` — pending tasks exist but none can run →
   halt with `status="blocked"` and a summary explaining which tasks are stuck

A project becomes blocked when a required dependency task has `failed` or `blocked`
status, making it impossible for dependent tasks to ever start.

---

## task_board.json

Every state change that advances or fails a task writes:

```
workspace/<project_id>/task_board.json
```

**Schema:**

```json
{
  "project_id": "todo-api",
  "project_goal": "Build a REST API for a todo app",
  "project_status": "active | complete | failed | blocked",
  "active_task_id": 2,
  "tasks": [
    {
      "id": 1,
      "title": "Scaffold project structure",
      "description": "...",
      "status": "done",
      "priority": 1,
      "depends_on": [],
      "outputs": ["main.py", "requirements.txt"],
      "error_context": "",
      "retries": 0,
      "max_retries": 3,
      "created_at": "2026-01-01T00:00:00+00:00",
      "updated_at": "2026-01-01T00:01:00+00:00"
    }
  ],
  "completed_task_ids": [1],
  "failed_task_ids": [],
  "blocked_task_ids": [],
  "task_history": [
    {"task_id": 1, "event": "started", "detail": "", "timestamp": "..."},
    {"task_id": 1, "event": "done",    "detail": "", "timestamp": "..."},
    {"task_id": 2, "event": "started", "detail": "", "timestamp": "..."}
  ],
  "persisted_at": "2026-01-01T00:01:01+00:00"
}
```

**task_history events:**

| Event | When |
|-------|------|
| `started` | Task becomes the active task |
| `done` | Task completed successfully |
| `failed` | Task exhausted retries |
| `blocked` | Task marked blocked (dep deadlock) |
| `retry` | Debug Agent attempted a fix |

---

## MOCK_LLM Mode

Activate mock mode to run the full pipeline without any API calls or credits:

```bash
# Via environment variable
AI_CTO_MOCK_LLM=true python -m ai_cto "Build an app" --id my-app

# Or in .env
AI_CTO_MOCK_LLM=true
```

Mock mode activates automatically when `ANTHROPIC_API_KEY` is absent.

In mock mode:
- PlanningAgent returns a pre-defined Todo API architecture
- TaskManagerAgent returns a deterministic single-task decomposition
- CodingAgent writes a known-good `main.py` (stdlib only, exits 0)
- DebugAgent applies a known-good patch regardless of the error
- VerificationAgent runs all 5 real static checks (no mock)
- ExecutionAgent runs the real subprocess (no mock)
- MemoryAgent runs normally against a real local ChromaDB

The mock code passes all 5 verification checks and exits 0. The full
pipeline completes successfully, including `task_board.json` on disk.

---

## Project State

Key fields in `ProjectState`:

| Field | Type | Description |
|-------|------|-------------|
| `idea` | str | Original user goal |
| `project_id` | str | Workspace directory slug |
| `architecture` | str | Markdown architecture doc |
| `tasks` | list[TaskItem] | Full task list |
| `active_task_id` | int\|None | Currently executing task |
| `completed_task_ids` | list[int] | Done task IDs |
| `failed_task_ids` | list[int] | Failed task IDs |
| `blocked_task_ids` | list[int] | Blocked task IDs |
| `task_history` | list[dict] | Chronological event log |
| `project_status` | str | `active\|complete\|failed\|blocked` |
| `status` | str | Graph routing signal |
| `final_output` | str | Human-readable summary |

---

## Verification

Before any code runs, the Verification Agent checks:

1. **SyntaxChecker** — Python AST parse
2. **SecurityChecker** — unsafe patterns (exec, subprocess with shell=True, etc.)
3. **DependencyChecker** — imports vs requirements.txt
4. **ArchitectureChecker** — entry point exists, structure matches architecture
5. **RuntimeRiskChecker** — infinite loops, missing error handling, etc.

Plus LLM-generated pytest tests written to `workspace/<project_id>/tests/`.

---

## Memory

Past projects are stored in a local ChromaDB vector database:

- **ARCHITECTURE** — architecture documents (retrieved by PlanningAgent)
- **PROJECT_PATTERN** — project summaries (retrieved by PlanningAgent)
- **DEBUG_SOLUTION** — error→fix pairs (retrieved by DebugAgent)

On success, the system persists all three types for future projects to learn from.

---

## Running Tests

```bash
# All tests (mock mode, no API needed)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_task_manager_v2.py -v
python -m pytest tests/test_task_board.py -v
python -m pytest tests/test_mock_pipeline.py -v
```

**Test coverage by file:**

| File | What it tests |
|------|---------------|
| `test_state.py` | `ProjectState` and `TaskItem` defaults |
| `test_task_manager.py` | Task helpers, mock task manager, per-task debug failure |
| `test_task_manager_v2.py` | V2 orchestration: deps, retries, history, board, mock pipeline |
| `test_task_board.py` | All task_board.py helpers + JSON persistence |
| `test_graph_routing.py` | Graph routing edge functions + advance_task_node |
| `test_mock_pipeline.py` | Full end-to-end pipeline in mock mode |
| `test_memory_store.py` | ChromaDB store operations |
| `test_memory_agent.py` | Memory retrieve/save nodes |
| `test_verification_checks.py` | All 5 static checkers |
| `test_verification_agent.py` | Verification agent node |
| `test_file_writer.py` | Path traversal protection |

---

## Project Structure

```
ai_cto/
  agents/
    planning.py       ← architecture document (Claude)
    task_manager.py   ← task decomposition (Claude) + board persist
    coding.py         ← file generation (Claude)
    verification.py   ← 5 static checks + test gen
    execution.py      ← Docker/subprocess sandbox
    debug.py          ← error diagnosis + patch (Claude, per-task retries)
    memory.py         ← ChromaDB retrieve/save
  memory/
    schema.py         ← MemoryEntry, MemoryType
    store.py          ← ChromaMemoryStore
  tools/
    file_writer.py    ← safe file materialisation
    sandbox.py        ← Docker + subprocess runners
    task_board.py     ← task state transitions + JSON persistence
  verification/
    checks.py         ← 5 static checkers
    test_generator.py ← LLM-based pytest generation
  state.py            ← ProjectState + TaskItem TypedDicts + helpers
  graph.py            ← LangGraph StateGraph wiring
  mock_llm.py         ← pre-defined responses for mock mode
  __main__.py         ← CLI entry point
workspace/            ← generated project files (git-ignored)
tests/                ← pytest test suite
```
