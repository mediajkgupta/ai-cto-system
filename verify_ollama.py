#!/usr/bin/env python
"""
verify_ollama.py — End-to-end smoke test for Ollama provider integration.

What it does:
  1. Confirms Ollama is reachable at OLLAMA_BASE_URL (default: http://localhost:11434)
  2. Prints which model will be used
  3. Runs the PlanningAgent for a simple project idea via the provider abstraction
  4. Prints the generated architecture document
  5. Optionally runs the full pipeline (pass --full to enable)

Usage:
    # Ollama must already be running: ollama serve
    AI_CTO_LLM_PROVIDER=ollama python verify_ollama.py
    AI_CTO_LLM_PROVIDER=ollama OLLAMA_MODEL=codellama:7b python verify_ollama.py
    AI_CTO_LLM_PROVIDER=ollama python verify_ollama.py --full
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import shutil
from pathlib import Path

# ── Require explicit Ollama provider — never run as mock ─────────────────────
if os.environ.get("AI_CTO_LLM_PROVIDER", "").strip().lower() != "ollama":
    print(
        "ERROR: AI_CTO_LLM_PROVIDER must be set to 'ollama' before running this script.\n"
        "       Example:\n"
        "           AI_CTO_LLM_PROVIDER=ollama python verify_ollama.py\n"
        "       See docs/ollama.md for setup instructions."
    )
    sys.exit(1)

# Windows-safe stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("chromadb", "langchain", "httpx", "httpcore", "anthropic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SEP  = "=" * 64
SEP2 = "-" * 64

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ── Parse args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Verify Ollama provider integration.")
parser.add_argument(
    "--full",
    action="store_true",
    help="Run the full 5-node pipeline (not just PlanningAgent).",
)
args = parser.parse_args()


# ── Step 1: Connectivity check ────────────────────────────────────────────────

section("STEP 1  Checking Ollama connectivity")

from ai_cto.providers.ollama import OllamaProvider
ollama = OllamaProvider()

base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
model    = os.environ.get("OLLAMA_MODEL",    "deepseek-coder:latest")

print(f"\n  Base URL : {base_url}")
print(f"  Model    : {model}")
print()

if not ollama.is_reachable():
    fail(f"Cannot reach Ollama at {base_url}")
    print("\n  Is the server running?  Try:  ollama serve")
    sys.exit(1)

ok(f"Ollama reachable at {base_url}")
ok(f"Model: {model}")


# ── Step 2: Single provider.complete() call ───────────────────────────────────

section("STEP 2  Calling PlanningAgent via provider abstraction")

IDEA = "Build a simple REST API for a task manager with CRUD endpoints"

PLANNING_SYSTEM = """You are a senior software architect and AI CTO.
Given a software idea, write a concise architecture document in Markdown (max 400 words).

Return your response as valid JSON with this exact shape:
{
  "architecture": "<markdown string>"
}

Rules:
- Architecture must cover: components, data flow, tech stack choices.
- Be concrete and specific — avoid generic boilerplate.
- Do not include a task list; that will be handled separately.
"""

print(f"\n  Idea: {IDEA}")
print(f"\n  Sending request to Ollama (this may take 30–120 s on CPU)...\n")

from ai_cto.providers import get_provider
provider = get_provider()

try:
    raw = provider.complete(system=PLANNING_SYSTEM, user=f"Software idea: {IDEA}", max_tokens=2048)
except RuntimeError as exc:
    fail(f"Provider call failed: {exc}")
    sys.exit(1)

# Parse the JSON response
import re
def _extract_json(text):
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1)
    start = text.find("{")
    end   = text.rfind("}") + 1
    return text[start:end] if start != -1 and end > start else text

try:
    parsed = json.loads(_extract_json(raw))
    architecture = parsed["architecture"]
except (json.JSONDecodeError, KeyError) as exc:
    fail(f"Could not parse JSON response: {exc}")
    print(f"\n  Raw response:\n{raw[:500]}")
    sys.exit(1)

ok(f"Architecture generated ({len(architecture)} chars)")
print(f"\n{SEP2}\n  Architecture\n{SEP2}")
print(architecture[:1200])
if len(architecture) > 1200:
    print(f"\n  ... ({len(architecture) - 1200} chars truncated)")


# ── Step 3 (optional): Full pipeline ─────────────────────────────────────────

if not args.full:
    section("RESULT")
    print("\n  Basic Ollama integration working.")
    print("  Run with --full to exercise the complete pipeline.\n")
    sys.exit(0)

section("STEP 3  Full pipeline run (--full mode)")

import ai_cto.tools.file_writer as _fw
_WORKSPACE = Path(tempfile.mkdtemp(prefix="ai_cto_ollama_demo_"))
_fw.WORKSPACE_ROOT = _WORKSPACE
PROJECT_ID = "ollama-task-api-demo"

print(f"\n  Workspace: {_WORKSPACE}")
print(f"  Project  : {PROJECT_ID}")
print(f"  Idea     : {IDEA}\n")

from ai_cto.state import initial_state
from ai_cto.graph import build_graph

state = initial_state(idea=IDEA, project_id=PROJECT_ID)
graph = build_graph()

final_state = None
node_log    = []

try:
    for event in graph.stream(state):
        node_name  = list(event.keys())[0]
        node_state = event[node_name]
        status     = node_state.get("status", "?")
        completed  = node_state.get("completed_task_ids", [])
        active     = node_state.get("active_task_id")
        print(f"  [{node_name.upper():20s}] status={status:20s} "
              f"completed={completed}  active_task={active}")
        node_log.append(node_name)
        final_state = node_state
except KeyboardInterrupt:
    print("\n  (interrupted by user)")

print(f"\n{SEP2}\n  Pipeline summary\n{SEP2}")
if final_state:
    print(f"\n  status           : {final_state.get('status')}")
    print(f"  project_status   : {final_state.get('project_status')}")
    print(f"  completed_tasks  : {final_state.get('completed_task_ids')}")
    print(f"  generated_files  : {list(final_state.get('generated_files', {}).keys())}")
    print(f"\n  final_output:\n")
    print(final_state.get("final_output", "(none)"))
else:
    print("  (no final state — pipeline may have been interrupted)")

# Cleanup
shutil.rmtree(_WORKSPACE, ignore_errors=True)
print(f"\n  [Workspace cleaned up]\n")
