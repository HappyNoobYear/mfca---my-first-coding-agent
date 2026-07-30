"""Runs all MASEval-wrapped coding-agent benchmarks against a live Ollama
instance and exports results.

Usage: python -m benchmarks.maseval_bench.run_all
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Force the Ollama backend for this run without touching .env -- the
# interactive CLI's own default provider stays whatever the user configured.
os.environ["LLM_PROVIDER"] = "ollama"
os.environ.setdefault("LLM_MODEL", "gemma4:e2b")

from benchmarks.ollama_proxy import OllamaProxy
from benchmarks.maseval_bench.coding_benchmark import (
    MfcaCodingBenchmark,
    MiniSweAgentCodingBenchmark,
    build_simple_file_read_task,
    build_multi_turn_task,
)
from benchmarks.maseval_bench.smolagents_benchmark import SmolAgentsCodingBenchmark
from benchmarks.maseval_bench.humaneval_tasks import (
    build_has_close_elements_task,
    build_mean_absolute_deviation_task,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_PATH = _PROJECT_ROOT / "results" / "benchmark_results.json"
# One saved transcript per (scenario, agent) pair, in the same
# {timestamp, session_id, messages} shape ExternalMemoryBackend already uses
# for interactive CLI sessions -- so a benchmark run leaves behind the same
# kind of readable record a real user session does, instead of nothing.
_TRANSCRIPTS_DIR = _PROJECT_ROOT / "conversation_memory" / "benchmark_transcripts"

_AGENTS = {
    "mfca": (MfcaCodingBenchmark, "mfca"),
    "mini-swe-agent": (MiniSweAgentCodingBenchmark, "mini-swe-agent"),
    "smolagents": (SmolAgentsCodingBenchmark, "smolagents"),
}


def _save_transcript(scenario: str, agent_name: str, task_query: str, messages: list) -> None:
    _TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = f"{scenario}__{agent_name}"
    data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "scenario": scenario,
        "agent": agent_name,
        "task_query": task_query,
        "messages": messages,
    }
    with open(_TRANSCRIPTS_DIR / f"{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _run_one(bench_cls, agent_name, task, proxy, scenario, **bench_kwargs):
    proxy.reset()
    benchmark = bench_cls(progress_bar=False, **bench_kwargs)
    reports = benchmark.run(tasks=[task], agent_data={})
    r = reports[0]
    stats = proxy.get_stats()
    self_tokens = 0
    if r["usage"]:
        self_tokens = r["usage"].get("agents", {}).get(agent_name, {}).get("total_tokens", 0)
    messages = r["traces"].get("agents", {}).get(agent_name, {}).get("messages", [])
    _save_transcript(scenario, agent_name, task.query, messages)
    result = {
        "status": r["status"],
        "error": r["error"]["error_message"][:300] if r["error"] else None,
        "eval": r["eval"],
        "self_reported_tokens": self_tokens,
        "proxy_tokens": stats["total_tokens"],
        "proxy_tool_calls": stats["tool_calls"],
    }
    print(
        f"  {agent_name:<16} status={result['status']:<22} "
        f"proxy_tokens={result['proxy_tokens']:<6} proxy_tool_calls={result['proxy_tool_calls']:<3} "
        f"eval={result['eval']}"
    )
    return result


def main():
    proxy = OllamaProxy(proxy_port=11435)
    proxy.start()
    os.environ["OLLAMA_HOST"] = proxy.url

    all_results = {}
    try:
        print("=== simple_file_read ===")
        scenario = "simple_file_read"
        task = build_simple_file_read_task()
        all_results[scenario] = {
            label: _run_one(bench_cls, agent_name, task, proxy, scenario)
            for label, (bench_cls, agent_name) in _AGENTS.items()
        }

        print("\n=== multi_turn_5 ===")
        scenario = "multi_turn_5"
        task = build_multi_turn_task()
        all_results[scenario] = {
            label: _run_one(bench_cls, agent_name, task, proxy, scenario, max_invocations=5)
            for label, (bench_cls, agent_name) in _AGENTS.items()
        }

        print("\n=== humaneval_has_close_elements ===")
        scenario = "humaneval_has_close_elements"
        task = build_has_close_elements_task()
        all_results[scenario] = {
            label: _run_one(bench_cls, agent_name, task, proxy, scenario)
            for label, (bench_cls, agent_name) in _AGENTS.items()
        }

        print("\n=== humaneval_mean_absolute_deviation ===")
        scenario = "humaneval_mean_absolute_deviation"
        task = build_mean_absolute_deviation_task()
        all_results[scenario] = {
            label: _run_one(bench_cls, agent_name, task, proxy, scenario)
            for label, (bench_cls, agent_name) in _AGENTS.items()
        }
    finally:
        proxy.stop()

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
