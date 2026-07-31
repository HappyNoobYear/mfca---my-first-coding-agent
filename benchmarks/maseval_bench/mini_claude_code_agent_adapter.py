"""MASEval AgentAdapter for Mini Claude Code.

MASEval's adapter contract is deliberately minimal ("run the agent and
retrieve its messages" -- see maseval's own AGENTS.md). Mini Claude Code
already exposes almost exactly this shape: Agent.agent_loop(query) runs one
turn, and Agent.messages is already a plain list of OpenAI-format
role/content(/tool_calls) dicts, which is exactly what MessageHistory
expects. Only asyncio needs bridging, since MASEval's AgentAdapter contract
is synchronous.
"""

import asyncio
from typing import Any

from maseval.core.agent import AgentAdapter
from maseval.core.exceptions import EnvironmentError as MASEvalEnvironmentError
from maseval.core.history import MessageHistory
from maseval.core.usage import TokenUsage, Usage

from src.Agent.Agent import Agent

# The literal fallback string Agent.agent_loop() returns when the LLM provider
# returns None (Agent.py:128) -- a graceful, non-raising degradation that's
# correct for the interactive CLI (the user just sees an error and retries),
# but wrong for benchmarking: without this check, a failed API call silently
# reports as a successful "answer", since a non-empty error string is truthy.
_PROVIDER_FAILURE_MARKER = "Did not receive a valid response from the Ollama API"


class MiniClaudeCodeAgentAdapter(AgentAdapter):
    """Wraps Mini Claude Code's Agent class for MASEval.

    name defaults to the internal agent key "mfca" rather than the display
    name "Mini Claude Code" -- this string is threaded through the benchmark
    harness as a dict key and becomes part of saved transcript filenames and
    results/benchmark_results.json's structure, so changing it here would
    orphan every already-saved transcript and result file without a full
    benchmark rerun.
    """

    def __init__(self, agent_instance: Agent, name: str = "mfca", **kwargs: Any):
        super().__init__(agent_instance, name, **kwargs)
        # Mini Claude Code resets Agent._last_tokens_used to 0 at the start of
        # every agent_loop() call, but MASEval only calls _gather_usage() once
        # per task -- after all invocations (all turns) complete. Without this
        # running total, a multi-turn task would silently report only the
        # last turn's tokens, losing every earlier turn's usage.
        self._cumulative_tokens = 0

    def _run_agent(self, query: str) -> Any:
        """Run one Mini Claude Code turn and capture its message history for tracing."""
        answer = asyncio.run(self.agent.agent_loop(query))
        self._cumulative_tokens += self.agent._last_tokens_used
        # Mini Claude Code's message dicts are already OpenAI-format
        # (role/content/tool_calls/tool_call_id) -- the exact shape
        # MessageHistory expects, no conversion needed.
        self.messages = MessageHistory(list(self.agent.messages))

        if _PROVIDER_FAILURE_MARKER in (answer or ""):
            # Not the agent's fault -- the LLM API call itself failed. Raising
            # here (rather than letting the evaluator score this as a real
            # answer) makes MASEval classify the task as ENVIRONMENT_ERROR,
            # excluded from scoring, instead of a misleading SUCCESS.
            raise MASEvalEnvironmentError(
                "Mini Claude Code's LLM provider returned no response (Ollama API call failed)",
                component="mfca",
            )

        return answer

    def _gather_usage(self) -> Usage:
        """Report Mini Claude Code's real cumulative token count across all
        invocations of this task, sourced from each provider's actual API
        response, not estimated."""
        return TokenUsage(total_tokens=self._cumulative_tokens)

    def _resolve_model_id(self) -> str:
        return self.agent.model
