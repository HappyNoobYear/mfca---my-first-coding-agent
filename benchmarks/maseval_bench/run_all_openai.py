"""Same suite as run_all.py, run against OpenAI instead of a local Ollama
model, for the cloud-model comparison point in the report.

No independent counting proxy exists for OpenAI the way OllamaProxy
intercepts Ollama's wire traffic, so tokens and tool-call counts here are
each framework's own reported usage rather than a proxy-verified ground
truth. Tokens come from the provider's real API response (r["usage"]),
not framework guesswork, so this is still trustworthy, just not
cross-checked the same way the Ollama run is.

Usage: python -m benchmarks.maseval_bench.run_all_openai
"""

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["LLM_PROVIDER"] = "openai"
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")

from benchmarks.maseval_bench.coding_benchmark import (
    MiniClaudeCodeCodingBenchmark,
    MiniSweAgentCodingBenchmark,
    build_simple_file_read_task,
    build_multi_turn_task,
    count_tool_calls_from_trace,
)
from benchmarks.maseval_bench.smolagents_benchmark import SmolAgentsCodingBenchmark
from benchmarks.maseval_bench.humaneval_tasks import (
    build_has_close_elements_task,
    build_mean_absolute_deviation_task,
    extract_smolagents_tool_calls,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_PATH = _PROJECT_ROOT / "results" / "benchmark_results_openai.json"
_TRANSCRIPTS_DIR = _PROJECT_ROOT / "conversation_memory" / "benchmark_transcripts"

_AGENTS = {
    # "mfca" is the internal agent key, see the matching comment in run_all.py.
    "mfca": (MiniClaudeCodeCodingBenchmark, "mfca"),
    "mini-swe-agent": (MiniSweAgentCodingBenchmark, "mini-swe-agent"),
    "smolagents": (SmolAgentsCodingBenchmark, "smolagents"),
}


def _save_transcript(scenario: str, agent_name: str, task_query: str, messages: list) -> None:
    _TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = f"{scenario}__{agent_name}__openai"
    data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "scenario": scenario,
        "agent": agent_name,
        "provider": "openai",
        "task_query": task_query,
        "messages": messages,
    }
    with open(_TRANSCRIPTS_DIR / f"{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _run_one(bench_cls, agent_name, task, scenario, **bench_kwargs):
    benchmark = bench_cls(progress_bar=False, **bench_kwargs)
    reports = benchmark.run(tasks=[task], agent_data={})
    r = reports[0]
    self_tokens = 0
    if r["usage"]:
        self_tokens = r["usage"].get("agents", {}).get(agent_name, {}).get("total_tokens", 0)
    agent_traces = r["traces"].get("agents", {}).get(agent_name, {})
    messages = agent_traces.get("messages", [])
    _save_transcript(scenario, agent_name, task.query, messages)
    smolagents_calls = extract_smolagents_tool_calls(messages)
    tool_calls = count_tool_calls_from_trace(messages, agent_traces, smolagents_calls)
    result = {
        "status": r["status"],
        "error": r["error"]["error_message"][:300] if r["error"] else None,
        "eval": r["eval"],
        "self_reported_tokens": self_tokens,
        "tool_calls": tool_calls,
    }
    print(
        f"  {agent_name:<16} status={result['status']:<22} "
        f"tokens={result['self_reported_tokens']:<6} tool_calls={result['tool_calls']:<3} "
        f"eval={result['eval']}"
    )
    return result


def main():
    all_results = {}

    print("=== simple_file_read ===")
    scenario = "simple_file_read"
    task = build_simple_file_read_task()
    all_results[scenario] = {
        label: _run_one(bench_cls, agent_name, task, scenario)
        for label, (bench_cls, agent_name) in _AGENTS.items()
    }

    print("\n=== multi_turn_5 ===")
    scenario = "multi_turn_5"
    task = build_multi_turn_task()
    all_results[scenario] = {
        label: _run_one(bench_cls, agent_name, task, scenario, max_invocations=5)
        for label, (bench_cls, agent_name) in _AGENTS.items()
    }

    print("\n=== humaneval_has_close_elements ===")
    scenario = "humaneval_has_close_elements"
    task = build_has_close_elements_task()
    all_results[scenario] = {
        label: _run_one(bench_cls, agent_name, task, scenario)
        for label, (bench_cls, agent_name) in _AGENTS.items()
    }

    print("\n=== humaneval_mean_absolute_deviation ===")
    scenario = "humaneval_mean_absolute_deviation"
    task = build_mean_absolute_deviation_task()
    all_results[scenario] = {
        label: _run_one(bench_cls, agent_name, task, scenario)
        for label, (bench_cls, agent_name) in _AGENTS.items()
    }

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
