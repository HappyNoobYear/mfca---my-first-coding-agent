"""Minimal MASEval Benchmark wrapping Mini Claude Code for a coding-agent task.

No MASEval-shipped benchmark tests coding tool-use (they're multi-agent
coordination/safety/general-capability), so this Task/Environment/Evaluator
triad is the part we still design ourselves -- mirroring the existing
simple_file_read scenario in test_runner.py so results stay comparable.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from maseval.core.agent import AgentAdapter
from maseval.core.benchmark import Benchmark
from maseval.core.environment import Environment
from maseval.core.evaluator import Evaluator
from maseval.core.history import MessageHistory
from maseval.core.model import ModelAdapter
from maseval.core.task import Task
from maseval.core.user import User

from src.Agent.Agent import Agent, build_system_prompt
from src.API.factory import LLMProviderFactory
from src.config import Config
from src.Tools.ReadCodeTool import ReadCodeTool
from src.Tools.ReadDirectoryTool import ReadDirectoryTool
from src.Tools.ExecuteCodeTool import ExecuteCodeTool
from src.Tools.WriteCodeTool import WriteCodeTool
from src.Tools.WebFetchTool import WebFetchTool

import tempfile

from benchmarks.maseval_bench.mini_claude_code_agent_adapter import MiniClaudeCodeAgentAdapter
from benchmarks.maseval_bench.mini_swe_agent_adapter import MiniSweAgentAdapter, build_default_agent
from benchmarks.maseval_bench.humaneval_tasks import CorrectnessEvaluator, extract_smolagents_tool_calls

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CodingEnvironment(Environment):
    """Mini Claude Code manages its own tools internally (wired inside Agent.__init__,
    not supplied by MASEval), so this environment only tracks task-relevant
    file state for evaluators -- it has no tools of its own to create."""

    def setup_state(self, environment_data: Dict[str, Any]) -> Any:
        return {"file_path": environment_data.get("file_path")}

    def create_tools(self) -> Dict[str, Any]:
        return {}


class ToolUsageEvaluator(Evaluator):
    """Checks whether an agent produced an answer and used the expected tool,
    counting tool calls from the traced message history.

    agent_name selects which registered agent's trace to filter to, so the
    same evaluation logic is reused across Mini Claude Code, mini-swe-agent, or any other
    adapter without duplicating this class (the two-stage filter/evaluate
    pattern MASEval's Evaluator is designed for)."""

    def __init__(self, task: Task, environment: Environment, user: Optional[Any] = None, agent_name: str = "mfca"):
        super().__init__(task, environment, user)
        self.agent_name = agent_name
        self.expected_tool_substring = task.evaluation_data.get("expected_tool_substring", "")
        self.expected_command_keywords = task.evaluation_data.get("expected_command_keywords", [])

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        # Mini Claude Code's tool classes are named e.g. "ReadCodeTool" (lowercases to
        # "readcodetool", containing "readcode") but smolagents' tools are
        # named "read_code" (snake_case) -- the underscore breaks a plain
        # substring match even though it's the same conceptual tool.
        # Confirmed directly: once extract_smolagents_tool_calls() started
        # recovering real tool names, expected_tool_used still silently read
        # False for smolagents because "readcode" is not a substring of
        # "read_code". Strip underscores from both sides before comparing.
        return name.lower().replace("_", "")

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        return traces.get("agents", {}).get(self.agent_name, {})

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        messages = traces.get("messages", [])
        expected_substring = self._normalize_tool_name(self.expected_tool_substring)
        smolagents_calls = extract_smolagents_tool_calls(messages)
        tool_calls = count_tool_calls_from_trace(messages, traces, smolagents_calls)
        expected_tool_used = False

        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                name = tc.get("function", {}).get("name", "")
                if expected_substring in self._normalize_tool_name(name):
                    expected_tool_used = True

        # There's no tool *name* to match for these frameworks, but we do have
        # the actual command text (MiniSweAgentAdapter exposes it as
        # "commands") -- check the command content itself against keywords
        # characteristic of the expected operation (e.g. "cat" for a read,
        # ">" for a write) instead of leaving expected_tool_used permanently
        # False for every single-generic-tool agent.
        commands = traces.get("commands") or []
        if commands and self.expected_command_keywords:
            for cmd in commands:
                if any(kw.lower() in cmd.lower() for kw in self.expected_command_keywords):
                    expected_tool_used = True
                    break

        # smolagents-specific: MASEval's own SmolAgentAdapter never
        # populates a structured tool_calls field on messages, serializing
        # real calls as text instead (see extract_smolagents_tool_calls) --
        # recover them from there rather than leaving this at a permanent 0.
        if smolagents_calls:
            for call in smolagents_calls:
                name = call.get("function", {}).get("name", "")
                if expected_substring in self._normalize_tool_name(name):
                    expected_tool_used = True

        return {
            "answered": bool(final_answer),
            "tool_calls": tool_calls,
            "expected_tool_used": expected_tool_used,
            "actually_executed": _execution_happened(messages, traces, smolagents_calls),
        }


def count_tool_calls_from_trace(
    messages: List[Dict[str, Any]], traces: Dict[str, Any], smolagents_calls: Optional[List[dict]] = None
) -> int:
    """Framework-agnostic tool-call count, factored out of ToolUsageEvaluator
    so callers outside it (e.g. a benchmark runner reporting tool-call counts
    for HumanEval scenarios, which CorrectnessEvaluator doesn't track) can
    reuse the exact same counting logic instead of duplicating it.
    """
    tool_calls = sum(len(msg.get("tool_calls") or []) for msg in messages)

    # Single-tool frameworks (e.g. mini-swe-agent, where every action is a
    # bash command) don't carry named tool_calls on messages the OpenAI
    # way -- their adapter exposes a direct "tool_calls" trace count
    # instead. Prefer that count when present.
    if "tool_calls" in traces and isinstance(traces["tool_calls"], int):
        tool_calls = traces["tool_calls"]

    # smolagents-specific: MASEval's own SmolAgentAdapter never populates a
    # structured tool_calls field on messages, serializing real calls as text
    # instead (see extract_smolagents_tool_calls) -- recover them from there
    # rather than leaving this at a permanent 0.
    if smolagents_calls is None:
        smolagents_calls = extract_smolagents_tool_calls(messages)
    if smolagents_calls:
        tool_calls = len(smolagents_calls)

    return tool_calls


def _execution_happened(messages: list, traces: Dict[str, Any], smolagents_calls: List[dict]) -> bool:
    """Coarse, framework-agnostic signal: did an actual code-execution action
    happen anywhere in the trace, as opposed to only a write.

    Checking this separately from expected_tool_used matters -- confirmed via
    saved transcripts (conversation_memory/benchmark_transcripts/) that an
    agent can write plausible-looking code, never once run it, and still
    confidently assert a result in its final answer (e.g. smolagents claiming
    "the script now prints 2 and 4" on a multi_turn_5 run where it never
    called execute_code at all). Without this, that run scores identically to
    one that genuinely verified its own work.
    """
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            if tc.get("function", {}).get("name", "").lower() == "executecodetool":
                return True
    for cmd in traces.get("commands") or []:
        if "python" in cmd.lower():
            return True
    for call in smolagents_calls:
        if call.get("function", {}).get("name") == "execute_code":
            return True
    return False


def select_evaluator(task: Task, environment: Environment, user: Optional[Any], agent_name: str) -> Evaluator:
    """HumanEval tasks get real correctness testing; the synthetic tasks keep
    the tool-call heuristic. Shared by all three per-agent Benchmark classes
    so the branch isn't duplicated three times."""
    if str(task.metadata.get("scenario", "")).startswith("humaneval"):
        return CorrectnessEvaluator(task, environment, user)
    return ToolUsageEvaluator(task, environment, user, agent_name=agent_name)


class ScriptedUser(User):
    """Plays back a fixed, pre-scripted list of turns -- mirrors test_runner.py's
    multi_turn_5 scenario (a scripted sequence, not an LLM deciding when it's
    satisfied).

    Deliberately never signals early completion via is_done(): MASEval's
    execution_loop checks is_done() right after respond() returns the next
    query and, if done, discards that query without ever running it through
    run_agents(). For a fixed N-turn script we want all N turns run in order,
    so we rely solely on Benchmark(max_invocations=len(turns)) as the loop
    bound and let respond() return "" harmlessly once turns are exhausted
    (that final call's return value is never used, since the range() bound
    ends the loop at the same point).
    """

    def __init__(self, turns: List[str]):
        self._turns = list(turns)
        self._index = 0
        self.messages = MessageHistory()

    def get_initial_query(self) -> str:
        query = self._turns[self._index]
        self.messages.add_message("user", query)
        self._index += 1
        return query

    def respond(self, message: str) -> str:
        self.messages.add_message("assistant", message)
        if self._index >= len(self._turns):
            return ""
        query = self._turns[self._index]
        self.messages.add_message("user", query)
        self._index += 1
        return query

    def is_done(self) -> bool:
        return False

    def gather_traces(self) -> Dict[str, Any]:
        return {
            **super().gather_traces(),
            "messages": self.messages.to_list(),
            "turns_used": self._index,
        }

    def gather_config(self) -> Dict[str, Any]:
        return {**super().gather_config(), "total_turns": len(self._turns)}


class MiniClaudeCodeCodingBenchmark(Benchmark):
    """Minimal MASEval Benchmark wrapping Mini Claude Code for coding-agent tasks.

    For multi-turn tasks (task.user_data["turns"] set), construct this with
    max_invocations=len(turns) -- see build_multi_turn_task() below.
    """

    def setup_environment(self, agent_data, task, seed_generator):
        return CodingEnvironment(task.environment_data)

    def setup_user(self, agent_data, environment, task, seed_generator):
        turns = task.user_data.get("turns")
        return ScriptedUser(turns) if turns else None

    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        provider = LLMProviderFactory.get_provider()
        tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool, WebFetchTool]
        # Real system prompt (CRITICAL MANDATE included) -- not the generic
        # stub used before, which is why Mini Claude Code never reached for WriteCodeTool
        # in prior benchmark runs the way it does in normal use. No project
        # context injected: these are generic tasks unrelated to this repo.
        mini_claude_code_agent = Agent(
            model=Config.MODEL_NAME,
            system_prompt=agent_data.get(
                "system_prompt",
                build_system_prompt(include_project_context=False),
            ),
            tools=tools,
            provider=provider,
            session_id="maseval-spike",
        )
        # name="mfca" is the internal agent key threaded through transcript
        # filenames and results/benchmark_results.json -- see
        # mini_claude_code_agent_adapter.py's class docstring for why it's
        # not renamed to match the display name.
        adapter = MiniClaudeCodeAgentAdapter(mini_claude_code_agent, name="mfca")
        return [adapter], {"mfca": adapter}

    def setup_evaluators(self, environment, task, agents, user, seed_generator):
        return [select_evaluator(task, environment, user, "mfca")]

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


_SIMPLE_READ_FILENAME = "trail_log.md"
_SIMPLE_READ_SANDBOX_PATH = _PROJECT_ROOT / "sandbox_workspace" / _SIMPLE_READ_FILENAME

# Deliberately NOT AGENT_CONTEXT.md's real content. Confirmed via a saved
# transcript (conversation_memory/benchmark_transcripts/) that copying
# AGENT_CONTEXT.md in verbatim backfires specifically for mini-swe-agent:
# that file documents Mini Claude Code's own tools (ReadDirectoryTool, ReadCodeTool,
# WebFetchTool, ...) and gives example commands for them. mini-swe-agent has
# exactly one real tool, plain bash, but after reading the file it started
# hallucinating calls like ReadDirectoryTool(path="...") as if those were
# its own tools, none of which are valid bash, and burned its entire step
# budget on repeated syntax errors instead of ever finishing. The content
# below is unrelated, self-contained prose with no tool names, function-call
# syntax, or anything an agent could mistake for its own capabilities --
# a fair "read and summarize" task regardless of which agent reads it.
_SIMPLE_READ_CONTENT = (
    "# Weekly Trail Log\n\n"
    "This file tracks a hiking log for a small group that meets every Saturday.\n\n"
    "## Members\n"
    "- Priya, joined March 2024, prefers mountain trails\n"
    "- Tomas, joined June 2024, prefers forest loops\n"
    "- Aiko, joined January 2025, prefers coastal paths\n\n"
    "## Rules\n"
    "1. Meet at the trailhead parking lot at 8:00 AM.\n"
    "2. Bring at least 1 liter of water per person.\n"
    "3. Turn back if the group is not at the halfway point by 10:30 AM.\n\n"
    "## Recent Hikes\n"
    "- 2026-06-14: Ridge Loop, 12km, sunny, group of 4\n"
    "- 2026-06-21: Cedar Valley, 8km, light rain, group of 3\n"
    "- 2026-06-28: Ridge Loop, 12km, cloudy, group of 5\n"
)


def build_simple_file_read_task() -> Task:
    """Mirrors test_runner.py's existing simple_file_read scenario so results
    stay comparable to the pre-MASEval harness.

    Originally pointed agents at AGENT_CONTEXT.md via its native Windows
    absolute path (C:\\Users\\...). Confirmed via transcript that this broke
    every agent for a different reason: Mini Claude Code's own system prompt tells it
    "you may read any file under /app/", so it correctly refused a path
    outside that convention instead of calling ReadCodeTool at all; mini-swe-
    agent's bash treats unescaped backslashes as escape characters, silently
    deleting them and mangling the path into a nonexistent run-on string
    ("C:UsersDavid...") -- it retried the identical broken command 5 times
    rather than adapting. Both are real agent behaviors, not benchmark noise,
    but the task itself was unfair: one hard-coded path style can't suit
    Mini Claude Code's Docker-path convention and mini-swe-agent's shell simultaneously.

    Fix: copy a neutral reference file (see _SIMPLE_READ_CONTENT, not
    AGENT_CONTEXT.md -- see that constant's comment for why) into the shared
    sandbox_workspace (writable by this benchmark, already bridged to
    /app/sandbox_workspace on Windows via the existing junction), phrase the
    query with one unambiguous relative filename, and give ReadCodeTool a
    sandbox-relative fallback (see ReadCodeTool.py) so that same bare
    filename resolves correctly for Mini Claude Code/smolagents too. Originally this
    also mentioned the file's "full path" as /app/sandbox_workspace/... for
    tool-using agents, but that's actively wrong for mini-swe-agent:
    confirmed via transcript that Git Bash's MSYS root does not resolve a
    leading "/" the way cmd.exe does (cmd.exe treats /app/... as
    current-drive-relative, resolving via our C:\\app\\sandbox_workspace
    junction; Git Bash instead looks under its own install root), so a
    leading-slash path silently fails there. Giving the model two path
    phrasings to choose between was itself the bug -- one relative filename
    that's simultaneously valid for every agent removes the ambiguity
    instead of hoping the model guesses right.
    """
    _SIMPLE_READ_SANDBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SIMPLE_READ_SANDBOX_PATH.write_text(_SIMPLE_READ_CONTENT, encoding="utf-8")
    query = (
        f"Read the file {_SIMPLE_READ_FILENAME} (it's in your current working directory) "
        f"and summarize what it contains."
    )
    return Task(
        query=query,
        environment_data={"file_path": f"/app/sandbox_workspace/{_SIMPLE_READ_FILENAME}"},
        evaluation_data={
            "expected_tool_substring": "readcode",
            # For single-generic-tool agents (mini-swe-agent): there's no
            # tool *name* to match, but we do have the actual bash command
            # text -- "cat"/"head"/etc are the realistic ways to read a file.
            "expected_command_keywords": ["cat ", "head ", "less ", "more "],
        },
        metadata={"scenario": "simple_file_read"},
    )


def build_multi_turn_task() -> Task:
    """Mirrors test_runner.py's existing multi_turn_5 scenario. Run this task
    through a MiniClaudeCodeCodingBenchmark(max_invocations=5) instance -- max_invocations
    must match len(turns) for ScriptedUser's loop-bound assumption to hold."""
    turns = [
        "Create a simple Python script that prints numbers from 1 to 5",
        "Modify it to print only even numbers",
        "Add a variable to count iterations",
        "Make it more concise",
        "Show me the final code",
    ]
    return Task(
        query=turns[0],
        user_data={"turns": turns},
        evaluation_data={
            "expected_tool_substring": "writecode",
            "expected_command_keywords": [">", "cat >", "echo ", "tee "],
        },
        metadata={"scenario": "multi_turn_5"},
    )


class MiniSweAgentCodingBenchmark(Benchmark):
    """Same coding-agent tasks, wrapping mini-SWE-agent instead of Mini Claude Code.

    Reuses CodingEnvironment/ScriptedUser/ToolUsageEvaluator unchanged --
    exactly the cross-framework reuse this whole MASEval spike was meant to
    validate: the same Task/Environment/Evaluator triad now runs a second,
    structurally different agent (single bash tool vs. five typed tools)
    without duplicating any evaluation logic.
    """

    def setup_environment(self, agent_data, task, seed_generator):
        return CodingEnvironment(task.environment_data)

    def setup_user(self, agent_data, environment, task, seed_generator):
        turns = task.user_data.get("turns")
        return ScriptedUser(turns) if turns else None

    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        # HumanEval tasks are scored by reading solution.py from the shared
        # sandbox_workspace (see humaneval_tasks.CorrectnessEvaluator) -- point
        # mini-swe-agent's workdir there instead of a private tempdir so that
        # check works uniformly across all three agents. simple_file_read's
        # reference file also lives there now (see build_simple_file_read_task)
        # so a bare relative filename resolves correctly via bash's own cwd,
        # without needing any /app-style path translation. Other scenarios
        # keep the private tempdir; they don't depend on any specific location.
        scenario = str(task.metadata.get("scenario", ""))
        if scenario.startswith("humaneval") or scenario == "simple_file_read":
            workdir = str(_PROJECT_ROOT / "sandbox_workspace")
        else:
            workdir = tempfile.mkdtemp(prefix="maseval_mswea_")
        # Multi-turn tasks: preload every turn after the first so
        # ScriptedContinuationAgent chains through the whole scripted
        # sequence in one real run() call, giving mini-swe-agent genuine
        # cross-turn memory (see mini_swe_agent_adapter.py) instead of a
        # fresh, blind conversation per turn.
        turns = task.user_data.get("turns")
        pending_turns = turns[1:] if turns else None
        mini_agent = build_default_agent(workdir, pending_turns=pending_turns)
        adapter = MiniSweAgentAdapter(mini_agent, name="mini-swe-agent")
        return [adapter], {"mini-swe-agent": adapter}

    def setup_evaluators(self, environment, task, agents, user, seed_generator):
        return [select_evaluator(task, environment, user, "mini-swe-agent")]

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
