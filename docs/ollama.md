# Running AI CTO System with Ollama (no paid API required)

Ollama lets you run powerful open-source code models locally.
The AI CTO System supports Ollama as a first-class LLM provider through the
`AI_CTO_LLM_PROVIDER=ollama` environment variable.

---

## 1. Install Ollama

### macOS / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download and run the installer from <https://ollama.com/download/windows>.

---

## 2. Pull a coding model

`deepseek-coder:latest` is the default and performs well on coding tasks:

```bash
ollama pull deepseek-coder:latest
```

Other good alternatives:

| Model | Size | Notes |
|---|---|---|
| `deepseek-coder:6.7b` | ~4 GB | Fast, good coding performance |
| `deepseek-coder:latest` | ~7 GB | Best deepseek-coder quality |
| `codellama:7b` | ~4 GB | Meta's coding-focused model |
| `codellama:13b` | ~8 GB | Stronger, needs more RAM |
| `mistral:latest` | ~4 GB | Good general reasoning |

---

## 3. Start the Ollama server

```bash
ollama serve
```

The server listens on `http://localhost:11434` by default.
You can verify it is running:

```bash
curl http://localhost:11434/api/tags
```

---

## 4. Configure the AI CTO System

Set the following in your shell or `.env` file:

```bash
AI_CTO_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434   # default — change for remote server
OLLAMA_MODEL=deepseek-coder:latest       # default — change to any pulled model
```

**Important:** Do **not** set `AI_CTO_MOCK_LLM=true` — that would bypass Ollama
and use the built-in mock responses instead.

---

## 5. Run a project

```bash
python -m ai_cto "Build a REST API for a task manager with CRUD endpoints"
```

Or use the resume flow:

```bash
python verify_ollama.py   # end-to-end sanity check (see below)
```

---

## 6. End-to-end sanity check

`verify_ollama.py` in the project root:

- Checks that Ollama is reachable
- Runs one real pipeline request (PlanningAgent only, no code execution)
- Prints the generated architecture so you can confirm the model is working

```bash
python verify_ollama.py
```

Expected output (abbreviated):

```
[OK] Ollama reachable at http://localhost:11434
[OK] Model: deepseek-coder:latest

--- Architecture ---
## Task Manager REST API — Architecture
...
```

---

## 7. Performance tips

- **Use a GPU if available.** Ollama automatically uses CUDA / Metal when detected.
  CPU inference on a 7 B model takes 30–120 s per request — set a long timeout
  or use a smaller model (6.7 B / 3 B).
- **Increase context window** for long codebases by passing `OLLAMA_NUM_CTX=8192`
  in your environment (not yet exposed in `options` by default).
- **Remote Ollama** works too — set `OLLAMA_BASE_URL=http://your-server:11434` and
  make sure the port is accessible.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `OllamaProvider: cannot reach http://localhost:11434` | Run `ollama serve` first |
| Slow / timeouts | Use a smaller model; the pipeline timeout is 300 s per request |
| Garbled JSON from model | Try `codellama:13b` or `deepseek-coder:latest`; smaller models sometimes produce malformed JSON that the retry loop catches |
| `model not found` error | Run `ollama pull <model>` then retry |
| Still hitting mock mode | Confirm `AI_CTO_LLM_PROVIDER=ollama` is set **and** `AI_CTO_MOCK_LLM` is unset or not `true` |
