"""Spike: benchmarking Mini Claude Code through MASEval instead of the hand-rolled test_runner.py harness.

MASEval (github.com/maseval/MASEval, ACL 2026) is a framework-agnostic
multi-agent evaluation library. This spike wraps Mini Claude Code as a MASEval AgentAdapter
and runs it through a minimal coding-agent Task/Environment/Evaluator, since
none of MASEval's shipped benchmarks (MACS, CONVERSE, MultiAgentBench, COLBENCH,
GAIA-2, tau2-bench, MMLU) test coding tool-use -- that part is still ours to design.

OllamaProxy (benchmarks/ollama_proxy.py) is unaffected by this: it works at the
network layer, independent of whatever orchestrates the calls above it.
"""
