# mfca — My First Coding Agent

An autonomous, multi-turn AI coding assistant built for the Advanced Media Technologies (AMT) Master's module. The agent iteratively reasons, inspects directories, reads and writes source code, executes files inside a secure, network-isolated Docker sandbox, and fetches web content, all through a swappable LLM backend (local Ollama model or OpenAI's API).

## Table of Contents
- [Features](#features)
- [Architecture & Isolation](#architecture--isolation)
- [Technologies](#technologies)
- [Installation & Prerequisites](#installation--prerequisites)
- [Execution / Usage](#execution--usage)
- [Benchmarking & Evaluation](#benchmarking--evaluation)
- [Change Log](#change-log)
- [Contributors & License](#contributors--license)

## Features

**Swappable LLM backend:**

The same agent loop, tool set, and memory manager run unchanged against either a local Ollama model or the OpenAI API. Switching backends is a configuration change (`.env`), not a code change.

**Multi-Turn Conversational Loop:**

Retains conversation history across turns, so follow-up requests ("now run it," "fix that bug") work without re-explaining context.

**Autonomous Tool-Calling Ecosystem:**

- `ReadDirectoryTool` — lists the contents of a directory.
- `ReadCodeTool` — reads the contents of a file.
- `WriteCodeTool` — writes or modifies a file, restricted to the sandbox workspace.
- `ExecuteCodeTool` — runs a Python file inside the isolated sandbox container.
- `WebFetchTool` — makes an HTTP GET/POST request, with SSRF protections (blocks loopback, private, and cloud-metadata address ranges).

**Context and token budget management:**

A memory manager keeps long conversations inside the model's context window by (a) only sending the tool schemas relevant to the current prompt, and (b) compressing older turns into a summary, written by an actual model call when possible, once the conversation approaches the model's real context limit (derived from the model itself, not a fixed constant).

## Architecture & Isolation

To safely process LLM-generated code without putting the host machine at risk, the system runs as two Docker containers defined in `docker-compose.yml`:

- **`agent_controller`** — holds the conversation, talks to the LLM provider, and dispatches tool calls.
- **`sandbox_worker`** — an otherwise idle container that only exists to execute code on request, reached via `docker exec` (not a network call), with **no network route to the internet or any other Docker network**. `agent_controller` sits on a separate bridge network with outbound access for talking to the model.

Every executed command also runs under a hard 5-second timeout. See the accompanying report for the full design rationale, including the trade-offs of mounting the Docker socket into `agent_controller`.

## Technologies

- **Language:** Python 3.11
- **LLM backends:** Ollama (local, e.g. `gemma4:e2b`) or OpenAI API (e.g. `gpt-4o` / `gpt-4o-mini`)
- **Virtualization:** Docker & Docker Compose
- **Benchmarking:** [MASEval](https://github.com/maseval/MASEval), plus [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and [smolagents](https://github.com/huggingface/smolagents) as comparison baselines

## Installation & Prerequisites

Ensure Docker Desktop is running. If you want to use a local model, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull gemma4:e2b
```

1. Clone the repository and step into it:

```bash
git clone https://github.com/DavidRestle/mfca---my-first-coding-agent.git
cd mfca---my-first-coding-agent
```

2. Install Python dependencies (needed even if you only run the interactive agent, not just the benchmark harness):

```bash
pip install -r requirements.txt
```

3. Configure your environment:

```bash
cp .env.example .env
```

Then edit `.env`: set `LLM_PROVIDER` to `ollama` or `openai`, and set `LLM_MODEL` accordingly. If using OpenAI, fill in `OPENAI_API_KEY`. If using Ollama from inside Docker, `OLLAMA_HOST` is already set correctly in `docker-compose.yml` (`http://host.docker.internal:11434`), pointing back at the host machine.

## Execution / Usage

Build and start the whole system (both containers) and attach to the agent's interactive console in one command:

```bash
docker compose run --rm agent_controller
```

Example multi-turn session:

```
You: List the files inside the sandbox workspace directory.
You: Write a script named print.py that prints a custom logging message, and save it using your tools.
You: Now execute that script and show me the output.
```

### Environment Teardown

```bash
docker compose down
```

## Benchmarking & Evaluation

`benchmarks/maseval_bench/` wraps this agent (`mfca`) alongside two existing open-source frameworks, `mini-swe-agent` and `smolagents`, under [MASEval](https://github.com/maseval/MASEval)'s shared Task/Environment/Evaluator interfaces, so all three are scored on identical tasks with identical logic. This is the harness behind the Evaluation section of the accompanying report.

To rerun the local-model comparison (requires Ollama running with the model pulled):

```bash
python -m benchmarks.maseval_bench.run_all
```

To rerun the cloud-model comparison (requires `OPENAI_API_KEY` set in `.env`):

```bash
python -m benchmarks.maseval_bench.run_all_openai
```

Each run writes results to `results/benchmark_results.json` / `results/benchmark_results_openai.json`, and saves a full transcript per (scenario, agent) pair to `conversation_memory/benchmark_transcripts/`. **The results and transcripts already in this repo are the exact ones the report's numbers and transcript-level claims are based on.** A fresh rerun is expected to produce different numbers, both LLM backends are non-deterministic, and the report's own limitations section notes that every reported number is a single run, not an average, so some of what looks like a framework difference can just be run-to-run variance.

## Change Log

- **Evaluation phase** — Added the MASEval-based benchmark harness comparing mfca against mini-swe-agent and smolagents, an independent token/tool-call counting proxy for Ollama, and a local-vs-cloud model comparison.
- **Sandboxing hardening** — Network-isolated sandbox worker, Docker-socket-based command dispatch, path normalization against directory traversal, WebFetchTool SSRF protections.
- **Memory management** — Context-window-aware history compression (model-written summaries with a deterministic fallback) and selective tool-schema exposure.
- **Provider abstraction** — Swappable `LLMInterface` behind `OpenAIProvider`/`OllamaProvider`, selected via configuration.
- **v1.2** — Multi-turn conversation memory.
- **v1.0** — `ExecuteCodeTool` and container isolation.
- **v0.3** — `WriteCodeTool`.
- **v0.2** — `ReadDirectoryTool`, `ReadCodeTool`.
- **v0.1** — Initial prototype.

## Contributors

- **Author:** David Restle
- **Academic Context:** Master's Student — Module: Advanced Media Technologies (AMT), Summer Semester 2026

## License

This project is open-source and licensed under the MIT License.
