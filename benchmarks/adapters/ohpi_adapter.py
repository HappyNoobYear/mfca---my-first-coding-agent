"""Adapter for oh-my-pi agent to benchmark interface.

Uses subprocess/RPC approach to avoid modifying oh-my-pi source.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

from benchmarks.agent_interface import AgentInterface, AgentResult


class OhMyPiAdapter(AgentInterface):
    """Adapter for oh-my-pi agent via subprocess."""

    def __init__(self, ohpi_path: str = None):
        """Initialize adapter.

        Args:
            ohpi_path: Path to oh-my-pi repository. If None, tries to find it.
        """
        self.ohpi_path = Path(ohpi_path) if ohpi_path else self._find_ohpi()
        self.runner_script = None

    async def initialize(self) -> None:
        """Initialize oh-my-pi agent."""
        if not self.ohpi_path or not self.ohpi_path.exists():
            raise RuntimeError(
                f"oh-my-pi not found at {self.ohpi_path}. "
                "Please clone from https://github.com/can1357/oh-my-pi.git"
            )

        self.runner_script = self._create_runner_script()

    async def process_task(self, task_description: str) -> AgentResult:
        """Execute a single task via subprocess."""
        try:
            if not self.runner_script:
                await self.initialize()

            result = await self._run_via_subprocess(task_description)
            return result
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                tokens_used=0,
                tool_calls=0,
                turns_completed=0,
                error=str(e),
            )

    async def process_multi_turn(self, turns: List[str]) -> AgentResult:
        """Execute multi-turn conversation via subprocess."""
        try:
            if not self.runner_script:
                await self.initialize()

            combined_prompt = "\n".join(
                [f"Turn {i+1}: {turn}" for i, turn in enumerate(turns)]
            )
            result = await self._run_via_subprocess(combined_prompt)
            result.turns_completed = len(turns)
            return result
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                tokens_used=0,
                tool_calls=0,
                turns_completed=0,
                error=str(e),
            )

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    async def _run_via_subprocess(self, prompt: str) -> AgentResult:
        """Run oh-my-pi via subprocess and capture output."""
        # Create a wrapper script that runs oh-my-pi and returns metrics
        import tempfile
        import os

        wrapper_code = f'''
import sys
sys.path.insert(0, "{self.ohpi_path}")

# Import oh-my-pi components (adjust based on actual API)
try:
    # This is a placeholder - actual oh-my-pi API will differ
    from main import Agent  # or whatever the actual import is
    agent = Agent()
    result = agent.run("{prompt}")

    output = {{
        "success": True,
        "output": str(result),
        "tokens": 0,  # oh-my-pi may not expose this
        "tool_calls": 0
    }}
except ImportError:
    # Fallback: try to execute oh-my-pi as a script
    output = {{
        "success": False,
        "output": "",
        "tokens": 0,
        "tool_calls": 0,
        "error": "oh-my-pi import failed"
    }}

import json
print(json.dumps(output))
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_code)
            wrapper_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, wrapper_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return AgentResult(
                    success=data.get("success", False),
                    output=data.get("output", ""),
                    tokens_used=data.get("tokens", 0),
                    tool_calls=data.get("tool_calls", 0),
                    turns_completed=1,
                    metadata={"provider": "oh-my-pi"},
                )
            else:
                return AgentResult(
                    success=False,
                    output="",
                    tokens_used=0,
                    tool_calls=0,
                    turns_completed=0,
                    error=result.stderr,
                )
        finally:
            os.unlink(wrapper_file)

    def _find_ohpi(self) -> Path:
        """Try to find oh-my-pi installation."""
        # Try common locations
        candidates = [
            Path("/tmp/oh-my-pi"),
            Path.home() / "oh-my-pi",
            Path.home() / "projects" / "oh-my-pi",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _create_runner_script(self) -> Path:
        """Create a runner script for oh-my-pi."""
        # Placeholder - will be implemented once we understand oh-my-pi API
        return self.ohpi_path / "main.py"
