"""Wires HuggingFace smolagents into the same coding-agent Task/Environment/
Evaluator triad, using MASEval's own SmolAgentAdapter (maseval.interface.agents.
smolagents) instead of a hand-built one -- this is the "get it for free" payoff
of adopting MASEval: no custom message-format conversion or usage-tracking
code needed, MASEval already reads smolagents' agent.memory.steps directly.

Tool parity with Mini Claude Code: smolagents gets the same four tools Mini
Claude Code has (read/list/write/execute), implemented by wrapping Mini
Claude Code's own tool classes directly
(see smolagents_tools.py) rather than reimplementing sandboxing logic --
both agents are judged against the literal same sandbox behavior.
"""

from typing import Any

from maseval.core.benchmark import Benchmark
from maseval.core.model import ModelAdapter
from maseval.interface.agents.smolagents import SmolAgentAdapter

from smolagents import ToolCallingAgent
from smolagents.models import LiteLLMModel

from src.config import Config

from benchmarks.maseval_bench.coding_benchmark import CodingEnvironment, ScriptedUser, select_evaluator
from benchmarks.maseval_bench.mini_swe_agent_adapter import normalize_ollama_host
from benchmarks.maseval_bench.smolagents_tools import build_smolagents_tools
import os


class ContinuingSmolAgentAdapter(SmolAgentAdapter):
    """MASEval's SmolAgentAdapter calls agent.run(query) with no reset
    argument, so it uses smolagents' own default of reset=True -- wiping
    smolagents' memory at the start of every single call. For our multi-turn
    tasks (5 invocations of the same adapter instance), that meant smolagents
    never actually had a real multi-turn conversation: each turn started
    completely fresh, with no memory that a prior turn happened at all
    (confirmed directly: smolagents' run() docstring says reset "whether to
    reset the conversation or keep it going from previous run"). This
    override passes reset=False from the second call onward, so smolagents
    gets genuine conversational continuity, matching what Mini Claude Code
    already does natively."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._has_run_once = False

    def _run_agent(self, query: str) -> str:
        final_answer = self.agent.run(query, reset=not self._has_run_once)
        self._has_run_once = True
        return final_answer


class SmolAgentsCodingBenchmark(Benchmark):
    """Same coding-agent tasks, wrapping smolagents' ToolCallingAgent via
    MASEval's built-in SmolAgentAdapter instead of a hand-built one."""

    def setup_environment(self, agent_data, task, seed_generator):
        return CodingEnvironment(task.environment_data)

    def setup_user(self, agent_data, environment, task, seed_generator):
        turns = task.user_data.get("turns")
        return ScriptedUser(turns) if turns else None

    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        model_name = Config.MODEL_NAME or "gemma4:e2b"
        if Config.PROVIDER == "openai":
            # litellm reads OPENAI_API_KEY from the environment itself, no
            # api_base override needed (that's only for pointing at a local
            # Ollama server or a benchmark proxy in front of one).
            model = LiteLLMModel(model_id=f"openai/{model_name}")
        else:
            host = normalize_ollama_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
            # ollama_chat/ (not plain ollama/) is required for tool-calling support --
            # the plain prefix is litellm's raw generate endpoint, which has no
            # function-calling capability. Same fix already applied for mini-swe-agent.
            model = LiteLLMModel(model_id=f"ollama_chat/{model_name}", api_base=host)
        smol_agent = ToolCallingAgent(model=model, tools=build_smolagents_tools(), max_steps=10)
        adapter = ContinuingSmolAgentAdapter(smol_agent, name="smolagents")
        return [adapter], {"smolagents": adapter}

    def setup_evaluators(self, environment, task, agents, user, seed_generator):
        return [select_evaluator(task, environment, user, "smolagents")]

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        raise NotImplementedError(
            "This spike has no evaluators/simulators that need an LLM judge yet."
        )

    def evaluate(self, evaluators, agents, final_answer, traces):
        results = []
        for ev in evaluators:
            filtered = ev.filter_traces(traces)
            results.append(ev(filtered, final_answer))
        return results

    def run_agents(self, agents, task, environment, query):
        return agents[0].run(query)
