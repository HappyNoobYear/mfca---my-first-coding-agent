"""MASEval AgentAdapter for mini-SWE-agent.

DefaultAgent.run() resets self.agent.messages to an empty list every time
it's called, so naively calling it once per scripted turn (as this adapter
originally did) discards all prior conversation -- each turn started blind,
confirmed directly via transcript (turn 2 of multi_turn_5 replying "the user
wants to modify an unspecified script... no original code was provided").

That's not a hard limitation of mini-SWE-agent itself, though -- DefaultAgent's
step loop is exception-driven: it raises Submitted when the model wants to
finish, and InteractiveAgent (mini-SWE-agent's own human-in-the-loop REPL
class) intercepts that Submitted *before* it becomes a real exit, and if the
human types a new task instead of quitting, raises UserInterruption instead --
a message with role="user" (not "exit"), which DefaultAgent.run()'s loop just
appends and continues, with the full prior conversation intact. That's a real,
library-native continuation mechanism; it's just gated behind a human prompt.
ScriptedContinuationAgent below reuses the exact same mechanism with a
pre-scripted queue instead of a human, giving multi-turn tasks genuine
cross-turn memory without modifying mini-SWE-agent itself.

Since MASEval only calls get_messages()/_gather_usage() once per task, after
all invocations (turns) complete, this adapter also accumulates messages,
tokens, and tool-call counts itself across calls rather than losing everything
but the last turn.
"""

import os
import platform
import subprocess
from typing import Any, List, Optional

from maseval.core.agent import AgentAdapter
from maseval.core.history import MessageHistory
from maseval.core.usage import TokenUsage, Usage

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted, UserInterruption
from minisweagent.models.litellm_model import LitellmModel

from src.config import Config

_GIT_BASH_CANDIDATES = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
]


def _find_git_bash() -> Optional[str]:
    for path in _GIT_BASH_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


class WindowsBashEnvironment(LocalEnvironment):
    """LocalEnvironment runs commands via subprocess.Popen(cmd, shell=True),
    which on Windows means cmd.exe -- but mini-swe-agent's own prompt template
    and DefaultAgent logic assume a POSIX/bash shell (heredocs like
    `cat <<EOF`, etc). Confirmed directly: every bash-syntax command
    mini-swe-agent issued on this machine failed identically with cmd.exe's
    "'<<' kann syntaktisch an dieser Stelle nicht verarbeitet werden" syntax
    error, the agent never adapted, and it exhausted its step budget without
    ever writing a file -- an environment mismatch, not an agent-quality
    difference. This override runs commands through the Git Bash already
    installed on this machine instead, so mini-swe-agent gets the shell it
    actually expects. Falls back to the default cmd.exe behavior if Git Bash
    isn't found or we're not on Windows.
    """

    def execute(self, action: dict, cwd: str = "", *, timeout: Optional[int] = None) -> dict:
        bash = _find_git_bash()
        if platform.system() != "Windows" or not bash:
            return super().execute(action, cwd, timeout=timeout)

        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            process = subprocess.Popen(
                [bash, "-c", command],
                cwd=cwd,
                env=os.environ | self.config.env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout, _ = process.communicate(timeout=timeout or self.config.timeout)
            output = {"output": stdout, "returncode": process.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

_SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer."
_INSTANCE_TEMPLATE = """Please solve this task: {{task}}

You can execute bash commands to implement the necessary changes.

Every response should include:
1. Reasoning text explaining your analysis and plan
2. At least one bash tool call with your command

Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not combine it with any other command. After this command, you cannot continue working on this task.
"""

_STEP_LIMIT = 15  # bounded explicitly -- see mini_swe_agent_adapter.py in benchmarks/adapters/ for why


def normalize_ollama_host(raw: str) -> str:
    """Same normalization as OllamaProvider.__init__: bare host:port values
    (e.g. the system OLLAMA_HOST=0.0.0.0 used to configure the Ollama server's
    bind address) are not directly connectable client URLs -- litellm needs
    an explicit scheme, and 0.0.0.0 is a listen address, not a client target."""
    if raw and not raw.startswith("http"):
        raw = f"http://{raw}:11434"
    return raw.replace("0.0.0.0", "localhost")


class ScriptedContinuationAgent(DefaultAgent):
    """Gives scripted multi-turn tasks real cross-turn memory by reusing
    DefaultAgent's own Submitted/UserInterruption continuation mechanism --
    the same one InteractiveAgent uses when a human declines to quit and
    types a new task instead (see module docstring). Here the "new task"
    comes from a pre-scripted queue instead of a human prompt, so run() is
    called exactly ONCE for the entire multi-turn sequence and self.messages
    accumulates continuously across every turn, instead of being wiped by a
    fresh run() call per turn.

    Mirrors InteractiveAgent.execute_actions()'s try/except/finally structure
    exactly (catch Submitted before it propagates to run()'s loop; decide
    whether to re-raise it as a real exit or replace it with a continuing
    UserInterruption) -- just with the "ask a human" step replaced by
    "pop the next scripted turn."
    """

    def __init__(self, *args: Any, pending_turns: Optional[List[str]] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pending_turns: List[str] = list(pending_turns or [])

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        try:
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            if self._pending_turns:
                next_task = self._pending_turns.pop(0)
                raise UserInterruption({"role": "user", "content": next_task})
            raise e
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result


def build_default_agent(workdir: str, pending_turns: Optional[List[str]] = None) -> DefaultAgent:
    """Construct a fresh mini-SWE-agent DefaultAgent pointed at whichever
    backend Config.PROVIDER selects.

    For Ollama, OLLAMA_HOST is read fresh (not cached), so a benchmark proxy
    set at runtime is actually picked up -- mirrors the fix in
    src/API/factory.py. For OpenAI, no host override is needed -- litellm
    reads OPENAI_API_KEY from the environment itself.

    pending_turns: remaining scripted turns (after the first) for multi-turn
    tasks -- see ScriptedContinuationAgent. None/empty for single-turn tasks,
    where this behaves identically to a plain DefaultAgent.
    """
    model_name = Config.MODEL_NAME or "gemma4:e2b"
    if Config.PROVIDER == "openai":
        model = LitellmModel(
            model_name=f"openai/{model_name}",
            cost_tracking="ignore_errors",
        )
    else:
        host = normalize_ollama_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model = LitellmModel(
            model_name=f"ollama_chat/{model_name}",
            model_kwargs={"api_base": host, "drop_params": True},
            cost_tracking="ignore_errors",
        )
    env = WindowsBashEnvironment(cwd=workdir)
    # step_limit gates on self.n_calls, which now accumulates across the
    # WHOLE chained multi-turn run (only one real run() call happens, see
    # ScriptedContinuationAgent) instead of resetting per turn -- scale the
    # budget by how many turns are queued so each one still effectively
    # gets _STEP_LIMIT steps, matching the original per-turn intent.
    total_turns = 1 + len(pending_turns or [])
    return ScriptedContinuationAgent(
        model,
        env,
        pending_turns=pending_turns,
        system_template=_SYSTEM_TEMPLATE,
        instance_template=_INSTANCE_TEMPLATE,
        step_limit=_STEP_LIMIT * total_turns,
        cost_limit=0,
    )


class MiniSweAgentAdapter(AgentAdapter):
    """Wraps mini-SWE-agent's DefaultAgent for MASEval."""

    def __init__(self, agent_instance: DefaultAgent, name: str = "mini-swe-agent", **kwargs: Any):
        super().__init__(agent_instance, name, **kwargs)
        self._cumulative_messages: List[dict] = []
        self._cumulative_tokens = 0
        self._cumulative_tool_calls = 0
        self._cumulative_commands: List[str] = []
        # For multi-turn tasks, self.agent is a ScriptedContinuationAgent
        # preloaded with every turn after the first (see setup_agents()),
        # so a single run() call internally chains through the whole
        # scripted sequence. MASEval's execution_loop still calls
        # _run_agent() once per scripted turn regardless (it doesn't know
        # mini-swe-agent already consumed the rest) -- this guard makes
        # every call after the first a no-op that just replays the final
        # result, instead of calling run() again and wiping everything the
        # first call already accumulated.
        self._already_ran = False

    def _run_agent(self, query: str) -> Any:
        if self._already_ran:
            return self._final_answer()
        self._already_ran = True
        # DefaultAgent.run() only resets self.messages, not n_calls/cost/
        # n_consecutive_format_errors -- reset explicitly so a fresh task
        # doesn't inherit leftover budget from a previous task run on the
        # same instance.
        self.agent.n_calls = 0
        self.agent.cost = 0.0
        self.agent.n_consecutive_format_errors = 0
        self.agent.run(query)  # resets self.agent.messages internally before running
        for msg in self.agent.messages:
            extra = msg.get("extra", {}) or {}
            response = extra.get("response")
            if response:
                usage = response.get("usage") or {}
                self._cumulative_tokens += (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
            actions = extra.get("actions") or []
            self._cumulative_tool_calls += len(actions)
            self._cumulative_commands.extend(a.get("command", "") for a in actions)
        self._cumulative_messages.extend(self.agent.messages)
        self.messages = MessageHistory(list(self._cumulative_messages))
        return self._final_answer()

    def _final_answer(self) -> str:
        """DefaultAgent's mandated completion command (`echo
        COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, "do not combine with any
        other command") produces no output after that marker line, so
        `submission` is "" even on a fully successful run -- meaning
        `bool(final_answer)` would read False for every clean completion.
        Meanwhile a failed run's exit message has content="RepeatedFormatError",
        which is truthy -- inverting the signal exactly backwards (success
        reads as unanswered, failure reads as answered). Base "answered" on
        the actual exit_status instead: anything other than a clean
        "Submitted" exit (RepeatedFormatError, LimitsExceeded, ...) is not
        answered, full stop -- the interaction itself failed, regardless of
        whether a correct artifact happens to exist on disk (that's a
        separate, independent check for tasks that have one).

        A clean "Submitted" exit isn't automatically "answered" either,
        though: confirmed via a saved transcript that the agent can hit
        "Submitted" with an empty submission string after a turn where it
        never actually did what was asked (multi_turn_5's "show me the final
        code" turn -- every response after the request was a format-error
        retry, nothing was ever shown, and it still exited cleanly). An
        empty submission on a clean exit isn't proof of nothing, though --
        the mandated completion command itself never carries content, so a
        turn that already displayed its result via an earlier tool call
        (e.g. HumanEval's "verification passed" printed by the script itself)
        would also show up this way.

        Two things had to be gotten right here, both found by testing against
        real saved transcripts rather than reasoning in the abstract:

        1. The tool message immediately before a clean exit is *always* a
           synthetic "action was not executed" placeholder -- ScriptedContinuationAgent's
           execute_actions() raises Submitted mid-batch, so the completion
           command's own "observation" is never recorded, on every single
           clean exit, success or not. Naively checking only the single most
           recent tool message means it's always this placeholder, so
           "answered" would always read False -- confirmed this broke a
           genuinely correct simple_file_read run. Skip past it specifically
           to find the most recent *real* tool result instead.

        2. That search has to stop at the current turn's own boundary, or it
           credits a previous turn's leftover output to a turn where nothing
           new happened at all. Confirmed via the exact "show me the final
           code" transcript above: walking back past its turn boundary reaches
           turn 4's real "2\n4" output and would incorrectly mark it answered.
           ScriptedContinuationAgent injects each new scripted turn as a plain
           "user" message (see execute_actions() above); FormatError retries
           are also role="user" but always carry the literal
           "No tool calls found" text. That difference is what marks a real
           turn boundary versus just another retry within the same turn.
        """
        if not self.agent.messages:
            return ""
        last = self.agent.messages[-1]
        extra = last.get("extra", {}) or {}
        if extra.get("exit_status") != "Submitted":
            return ""
        submission = extra.get("submission") or ""
        if submission.strip():
            return submission
        for msg in reversed(self.agent.messages):
            role = msg.get("role")
            if role == "user":
                if "No tool calls found" not in (msg.get("content") or ""):
                    break  # crossed into a previous turn's boundary -- nothing new here
                continue
            if role != "tool":
                continue
            msg_extra = msg.get("extra", {}) or {}
            if msg_extra.get("exception_info") == "action was not executed":
                continue
            return msg_extra.get("raw_output", "") or ""
        return ""

    def _gather_usage(self) -> Usage:
        return TokenUsage(total_tokens=self._cumulative_tokens)

    def gather_traces(self) -> dict:
        return {
            **super().gather_traces(),
            "tool_calls": self._cumulative_tool_calls,
            "commands": self._cumulative_commands,
        }
