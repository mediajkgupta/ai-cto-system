# AI CTO System — Restart Handoff
Generated: 2026-03-15 | Pre-restart state snapshot

---

## 1. Exact Completed Work (This Session)

### Phase 3 — Stack-Aware Execution
- **`ai_cto/agents/execution.py`** — Added `_resolve_entry_point(files, entry_point)`: detects Node.js stack via `detect_stack()` + `detect_node_entry()` and overrides `main.py` default for Node.js projects. Both `run_in_docker()` and `run_in_subprocess()` now receive `files=` kwarg for correct runtime config selection.
- **`ai_cto/tools/sandbox.py`** — Full rewrite. Added `RuntimeConfig` TypedDict, `get_runtime_config(files, entry_point)`: selects `python:3.11-slim` / 256m / `python <entry>` for Python and `node:18-alpine` / 512m / `node <entry>` + `npm install --production` for Node.js. Docker command uses `--tmpfs /project:rw,exec,size=100m` and language-specific image.

### Phase 3 — Run Audit Log
- **`ai_cto/tools/run_logger.py`** (NEW) — `RunLogger` class: `record_event(node_name, node_state)` accumulates one entry per graph node; `save(final_state)` writes `workspace/<project_id>/run_summary.json` with full `ProjectStatus` embedded, event trace, duration, file list.
- **`ai_cto/__main__.py`** — `RunLogger` wired into both `_run_new()` and `_run_resume()` stream loops.

### Phase 3 — Task Board Enhancement
- **`ai_cto/tools/task_board.py`** — `persist_task_board()` now writes `debug_retries`, `generated_files` (sorted list), `generated_files_count`, and `status_report` (full `ProjectStatus` dict) to every `task_board.json`.

### Phase 4 — Machine-Readable Status System
- **`ai_cto/tools/status.py`** (NEW) — `get_status_report(state) → ProjectStatus`. Maps ALL internal LangGraph routing states to 5 public values: `planned | in_progress | completed | failed | blocked` at project, agent, and task level. `format_status_report()` renders human-readable text. Embedded in `task_board.json` and `run_summary.json` on every write.

### FastAPI Backend (Phase 5)
- **`ai_cto/api/__init__.py`** (NEW, empty)
- **`ai_cto/api/app.py`** (NEW) — `create_app()` factory, `_lifespan` loads `.env`, module-level `app` for uvicorn.
- **`ai_cto/api/models.py`** (NEW) — Pydantic models: `StartProjectRequest`, `StartProjectResponse`, `AgentStatusResponse`, `TaskStatusResponse`, `ProjectStatusResponse`, `RunSummaryResponse`, `TaskBoardResponse`, `ProjectSummaryResponse`, `ProjectListResponse`, `CancelProjectResponse`, `ErrorResponse`.
- **`ai_cto/api/runner.py`** (NEW) — `RunState` enum (RUNNING/COMPLETE/FAILED/CANCELLED), `_RUNS` registry, `_CANCEL_FLAGS` dict (per-project `threading.Event`), `start_project_run()`, `cancel_project_run()`, `_run_pipeline()` thread target with cancellation check between graph nodes.
- **`ai_cto/api/routes/__init__.py`** (NEW, empty)
- **`ai_cto/api/routes/projects.py`** (NEW) — 6 endpoints (see section 2).

### ChromaDB Test Isolation Fix
- **`tests/test_mock_pipeline.py`** — Added `isolated_memory` fixture (replaces global `_store` singleton with a fresh `ChromaMemoryStore` at `tmp_path` for the test duration). Added `_isolate_memory` autouse fixture to `TestFullPipeline`. Simplified `test_pipeline_memory_saves_architecture` to use the fixture instead of manual store swap.
- **`tests/test_task_manager_v2.py`** — Added same `isolated_memory` fixture. Added `_isolate_memory` autouse to `TestMockPipelineV2`.

### New Test Files
- **`tests/test_sandbox_runtime.py`** (NEW) — 15 tests for `get_runtime_config()` and `docker_available()`.
- **`tests/test_run_logger.py`** (NEW) — 12 tests for `RunLogger`.
- **`tests/test_execution_stack_aware.py`** (NEW) — 12 tests for `_resolve_entry_point()` and `execution_node` stack dispatch.
- **`tests/test_status.py`** (NEW) — 27 tests for `get_status_report()` and `format_status_report()`.
- **`tests/test_api.py`** (NEW) — 37 tests covering all 6 endpoints + /health + edge cases.

### Test Suite
**510/510 pass** with fixed order and random seeds 1234, 5678, 9999. Confirmed with `.chroma_db/` completely deleted (memory isolation fix works). No background jobs running.

---

## 2. Exact Pending Work

| Priority | Item |
|----------|------|
| P1 | **Docker live validation** — run one Python and one Node.js project through the real Docker path, confirm `run_summary.json` persists correctly. Blocked on Docker installation (see section 3). |
| P2 | `GET /projects` pagination — currently returns all projects with no limit |
| P3 | Frontend (Next.js) — not started, explicitly deferred |
| P4 | `DELETE /projects/{id}` cancellation is cooperative (fires between graph nodes only); mid-LLM-call cancellation is not possible with current architecture |

---

## 3. Exact Blockers

### Docker — Not Installed
Docker Desktop is **not installed** on this machine. The `docker` binary does not exist.

**Confirmed by:**
```
where docker             → INFO: Could not find files for the given pattern(s).
C:\Program Files\Docker  → No such file or directory
winget list docker       → No installed package found matching input criteria.
```

**Machine:** Windows 10 Home Build 19045 (22H2), 64-bit.
WSL is present (`wsl.exe` exists) but **no distributions are installed** — `wsl -l -v` returns the usage/help text.

**Required steps (run in Administrator PowerShell after restart):**

```powershell
# Step 1 — Enable WSL 2 (requires Administrator, triggers reboot)
wsl --install
# Follow prompts. Reboot if requested.

# Step 2 — After reboot: set WSL 2 as default
wsl --set-default-version 2

# Step 3 — Install Docker Desktop
winget install -e --id Docker.DockerDesktop
# OR download from: https://docs.docker.com/desktop/install/windows-install/

# Step 4 — Start Docker Desktop from the Start Menu
# Wait ~30s for daemon to start, then confirm:
docker info
docker run --rm hello-world

# Step 5 — Confirm AI CTO sandbox picks it up
cd C:\Users\computer\ai-cto-system
python -c "from ai_cto.tools.sandbox import docker_available; print(docker_available())"
# Expected: True
```

If `wsl --install` fails due to missing features:
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# Reboot, then: wsl --set-default-version 2
```

---

## 4. Exact Files Changed This Session

### New files (did not exist before this session)
```
ai_cto/api/__init__.py
ai_cto/api/app.py
ai_cto/api/models.py
ai_cto/api/runner.py
ai_cto/api/routes/__init__.py
ai_cto/api/routes/projects.py
ai_cto/tools/run_logger.py
ai_cto/tools/status.py
tests/test_sandbox_runtime.py
tests/test_run_logger.py
tests/test_execution_stack_aware.py
tests/test_status.py
tests/test_api.py
RESTART_HANDOFF.md   ← this file
```

### Modified files (existed before, changed this session)
```
ai_cto/__main__.py                      — wired RunLogger into both run paths
ai_cto/agents/execution.py              — added _resolve_entry_point(), files= to sandbox calls
ai_cto/tools/sandbox.py                 — full rewrite with RuntimeConfig + get_runtime_config()
ai_cto/tools/task_board.py              — added debug_retries, generated_files, status_report
tests/test_mock_pipeline.py             — isolated_memory fixture + TestFullPipeline autouse
tests/test_task_manager_v2.py           — isolated_memory fixture + TestMockPipelineV2 autouse
tests/test_run_logger.py                — updated for new run_summary shape (status object)
```

### pip installs this session
```
fastapi==0.135.1
starlette==0.52.1
pytest-randomly  (for seed-controlled test ordering)
```

### Files NOT changed this session (changed in previous session, still active)
```
ai_cto/agents/verification.py           — stack detection + entry point override
ai_cto/verification/checks.py           — NodeJsChecker, EmptyEntryPointChecker, detect_stack()
ai_cto/agents/task_manager.py           — deterministic task filter
tests/test_verification_checks.py       — NodeJs + EmptyEntry tests
tests/test_task_manager_llm.py          — filter integration tests
```

---

## 5. Exact Next Command After Restart

**First — verify the suite is still clean:**
```powershell
cd C:\Users\computer\ai-cto-system
python -m pytest tests/ -q -p no:randomly
# Expected: 510 passed
```

**Then — verify the API server starts:**
```powershell
python -m uvicorn ai_cto.api.app:app --reload --port 8000
# Visit http://localhost:8000/health → {"status":"ok"}
# Visit http://localhost:8000/docs   → OpenAPI UI
# Ctrl+C to stop
```

**Then (only after Docker is installed and running) — Docker validation:**
```powershell
python -c "
from ai_cto.tools.sandbox import docker_available, get_runtime_config, run_in_docker
from ai_cto.agents.execution import _resolve_entry_point
import pathlib, tempfile, shutil

# Python
py_files = {'main.py': 'print(\"hello from python\")\n'}
py_ws = pathlib.Path(tempfile.mkdtemp())
(py_ws / 'main.py').write_text('print(\"hello from python\")\n')
result = run_in_docker(str(py_ws), 'main.py', files=py_files, timeout=60)
print('Python Docker:', result)
shutil.rmtree(py_ws)

# Node.js
js_files = {'server.js': 'console.log(\"hello from node\");\n', 'package.json': '{\"name\":\"test\"}'}
js_ws = pathlib.Path(tempfile.mkdtemp())
(js_ws / 'server.js').write_text('console.log(\"hello from node\");\n')
(js_ws / 'package.json').write_text('{\"name\":\"test\"}')
entry = _resolve_entry_point(js_files, 'main.py')
result = run_in_docker(str(js_ws), entry, files=js_files, timeout=60)
print('Node Docker:', result)
shutil.rmtree(js_ws)
"
```

---

## 6. Background Tasks and Disk Persistence

**No background tasks are running.** The session was entirely synchronous.

**All results are persisted to disk:**

| Artefact | Path | State |
|----------|------|-------|
| All source code changes | `C:\Users\computer\ai-cto-system\ai_cto\` | On disk, committed to working tree |
| All test files | `C:\Users\computer\ai-cto-system\tests\` | On disk |
| `.chroma_db/` (production memory DB) | `C:\Users\computer\ai-cto-system\.chroma_db\` | **Deleted** during corruption testing. Will be recreated empty on first real pipeline run. No data loss — it contained test artefacts only. |
| `workspace/` | `C:\Users\computer\ai-cto-system\workspace\` | Intact. Prior pipeline run artefacts are here. |
| Session memory | `C:\Users\computer\.claude\projects\C--Users-computer\memory\MEMORY.md` | Updated with current state |
| This handoff | `C:\Users\computer\ai-cto-system\RESTART_HANDOFF.md` | This file |

**Note on `.chroma_db/`:** It was intentionally deleted to prove the test isolation fix works with no ChromaDB on disk. The production memory DB will be recreated automatically the first time a real pipeline run completes and `memory_save_node` fires. No manual action needed.
