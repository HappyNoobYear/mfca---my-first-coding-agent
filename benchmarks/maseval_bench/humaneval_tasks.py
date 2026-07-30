"""Two tasks adapted from HumanEval (Chen et al., 2021) -- real, independently
verified correctness testing, rather than the synthetic tool-call-counting
heuristic the other tasks use.

Tasks require the agent to write its solution to a fixed file (solution.py)
using its own tools and execute it to self-verify, rather than just typing
code in its final chat answer -- this is what actually makes the task test
agent behavior (tool orchestration) rather than just the underlying model's
raw coding ability. The agent's own self-verification claim is never trusted:
CorrectnessEvaluator independently re-runs its own hidden test assertions
against whatever is actually on disk after the agent finishes, inside the
same sandboxed container everything else in this project uses.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from maseval.core.environment import Environment
from maseval.core.evaluator import Evaluator
from maseval.core.task import Task

from src.Tools.ExecuteCodeTool import ExecuteCodeTool

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SANDBOX_WORKSPACE = _PROJECT_ROOT / "sandbox_workspace"
_SOLUTION_FILENAME = "solution.py"

_SMOLAGENTS_CALLING_TOOLS_RE = re.compile(r"^Calling tools:\n(.*)$", re.DOTALL)


def extract_smolagents_tool_calls(messages: List[dict]) -> List[dict]:
    """MASEval's own SmolAgentAdapter never populates a structured
    tool_calls field on any message -- confirmed directly by dumping a real
    trace: every message has tool_calls=None, and the actual call info is
    serialized as human-readable text instead, e.g.
    role="tool-call", content=[{"type": "text", "text":
    "Calling tools:\n[{'id': ..., 'function': {'name': 'write_code', ...}}]"}].
    The real data is genuinely present, just not exposed as structured data
    the way mfca's/mini-swe-agent's traces are -- parse it back out instead
    of leaving smolagents' tool-call visibility at a permanent 0.
    """
    calls: List[dict] = []
    for msg in messages:
        if msg.get("role") != "tool-call":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            text = block.get("text", "") if isinstance(block, dict) else ""
            match = _SMOLAGENTS_CALLING_TOOLS_RE.match(text)
            if not match:
                continue
            try:
                parsed = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                calls.extend(c for c in parsed if isinstance(c, dict))
    return calls


class CorrectnessEvaluator(Evaluator):
    """Reads the agent's actual solution.py from the shared sandbox (not
    text extracted from its chat answer), runs real hidden test assertions
    against it independently, and separately reports whether the agent's
    trace shows it used tools at all -- the two signals the "agent vs. LLM"
    distinction actually needs: did it behave like an agent, and is the
    result correct, scored independently of each other.
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[Any] = None):
        super().__init__(task, environment, user)
        self.test_code = task.evaluation_data["test_code"]
        self.task_id = task.metadata.get("scenario", task.id)
        # Clean slate before the agent runs (setup_evaluators() is called
        # before execution_loop() starts the agent) -- without this, a
        # leftover solution.py from a *previous* agent's run in this same
        # shared sandbox would be silently misattributed to whichever agent
        # runs next and failed to write anything at all.
        self._solution_path = _SANDBOX_WORKSPACE / _SOLUTION_FILENAME
        if self._solution_path.exists():
            self._solution_path.unlink()

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        return traces  # need the raw traces dict itself, to check any agent's tool_calls

    def _used_tools(self, traces: Dict[str, Any]) -> bool:
        """Coarse, framework-agnostic signal: did the agent make any tool
        calls at all. Deliberately not checking for specific tool names --
        mini-swe-agent's single-bash-tool paradigm has no "write" vs
        "execute" tool names to match against the way mfca/smolagents do."""
        for agent_trace in traces.get("agents", {}).values():
            if agent_trace.get("tool_calls"):
                return True
            messages = agent_trace.get("messages", [])
            for msg in messages:
                if msg.get("tool_calls"):
                    return True
            if extract_smolagents_tool_calls(messages):
                return True
        return False

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        used_tools = self._used_tools(traces)

        if not self._solution_path.exists():
            return {"resolved": False, "reason": "no_file_written", "used_tools": used_tools, "output": ""}

        code = self._solution_path.read_text(encoding="utf-8")
        combined = f"{code}\n\n{self.test_code}\n"
        check_filename = f"_humaneval_{self.task_id}_check.py"
        (_SANDBOX_WORKSPACE / check_filename).write_text(combined, encoding="utf-8")

        output = ExecuteCodeTool(filename=check_filename).execute()
        resolved = "ALL_TESTS_PASSED" in output and "Traceback" not in output and "Error" not in output

        # Clean up so this run's files can't leak into the next agent's check.
        try:
            (_SANDBOX_WORKSPACE / check_filename).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "resolved": resolved,
            "used_tools": used_tools,
            "reason": "ran" if resolved else "assertion_or_error",
            "output": output[:500],
        }


def build_has_close_elements_task() -> Task:
    """Adapted from HumanEval/0 (Chen et al., 2021, 'Evaluating Large Language
    Models Trained on Code'). Requires tool use: the task cannot be completed
    by chat text alone, since scoring only looks at solution.py on disk."""
    query = (
        "Using your available tools, write a Python function called "
        "has_close_elements(numbers, threshold) that takes a list of floats and a "
        "threshold float, and returns True if any two numbers in the list are closer "
        "to each other than the threshold, otherwise returns False. "
        "Save it to a file called solution.py. Then execute solution.py to verify it "
        "produces correct results, for example has_close_elements([1.0, 2.8, 3.0, 4.0, "
        "5.0, 2.0], 0.3) should be True. Report whether your verification passed."
    )
    test_code = (
        "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
        "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
        "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
        "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
        "print('ALL_TESTS_PASSED')"
    )
    return Task(
        query=query,
        evaluation_data={"function_name": "has_close_elements", "test_code": test_code},
        metadata={"scenario": "humaneval_has_close_elements", "source": "HumanEval/0"},
    )


def build_mean_absolute_deviation_task() -> Task:
    """Adapted from HumanEval/4 (Chen et al., 2021)."""
    query = (
        "Using your available tools, write a Python function called "
        "mean_absolute_deviation(numbers) that takes a list of floats and returns the "
        "Mean Absolute Deviation around the mean (the average of the absolute "
        "differences between each number and the mean of the list). "
        "Save it to a file called solution.py. Then execute solution.py to verify it "
        "produces correct results, for example mean_absolute_deviation([1.0, 2.0, 3.0, "
        "4.0]) should be 1.0. Report whether your verification passed."
    )
    test_code = (
        "r1 = mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
        "assert abs(r1 - 1.0) < 1e-6, f'expected 1.0, got {r1}'\n"
        "r2 = mean_absolute_deviation([1.0, 2.0, 3.0])\n"
        "assert abs(r2 - (2.0/3.0)) < 1e-6, f'expected 0.6667, got {r2}'\n"
        "print('ALL_TESTS_PASSED')"
    )
    return Task(
        query=query,
        evaluation_data={"function_name": "mean_absolute_deviation", "test_code": test_code},
        metadata={"scenario": "humaneval_mean_absolute_deviation", "source": "HumanEval/4"},
    )
