"""Adapter for smolagents to benchmark interface."""

from typing import List

try:
    from smolagents import CodeAgent, tool
    from smolagents.models import OllamaModel
except ImportError:
    # Fallback if imports fail
    CodeAgent = None
    tool = None
    OllamaModel = None

from benchmarks.agent_interface import AgentInterface, AgentResult


class SmolagentsAdapter(AgentInterface):
    """Adapter for smolagents agent."""

    def __init__(self):
        self.agent = None
        self.model = None
        self.total_tokens = 0

    async def initialize(self) -> None:
        """Initialize smolagents agent."""
        if not CodeAgent:
            raise RuntimeError("smolagents not available. Install with: pip install smolagents")

        try:
            # Try to use Ollama model if available
            try:
                self.model = OllamaModel(model_name="gemma4:e2b", base_url="http://localhost:11434")
                self.agent = CodeAgent(model=self.model)
            except Exception:
                # Fallback: use CodeAgent with default model
                self.agent = CodeAgent()

            self.total_tokens = 0
        except Exception as e:
            raise RuntimeError(f"Failed to initialize smolagents: {e}")

    async def process_task(self, task_description: str) -> AgentResult:
        """Execute a single task."""
        try:
            if not self.agent:
                await self.initialize()

            result = self.agent.run(task_description)

            # Approximate token count (smolagents doesn't expose this easily)
            tokens_used = self._estimate_tokens(task_description + str(result))

            return AgentResult(
                success=True,
                output=str(result),
                tokens_used=tokens_used,
                tool_calls=0,  # smolagents doesn't expose this easily
                turns_completed=1,
                metadata={"provider": "smolagents", "model": "default"},
            )
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
        """Execute multi-turn conversation."""
        try:
            if not self.agent:
                await self.initialize()

            final_output = ""
            total_tokens = 0

            for turn in turns:
                result = self.agent.run(turn)
                final_output = str(result)
                total_tokens += self._estimate_tokens(turn + str(result))

            return AgentResult(
                success=True,
                output=final_output,
                tokens_used=total_tokens,
                tool_calls=0,
                turns_completed=len(turns),
                metadata={"provider": "smolagents", "turns": len(turns)},
            )
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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate: ~4 chars per token)."""
        return len(text) // 4
