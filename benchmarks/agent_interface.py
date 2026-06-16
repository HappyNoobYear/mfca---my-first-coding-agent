"""Unified interface for agent comparison testing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AgentResult:
    """Standardized result from agent execution."""

    success: bool
    output: str
    tokens_used: int
    tool_calls: int
    turns_completed: int
    error: str = None
    metadata: Dict[str, Any] = None


class AgentInterface(ABC):
    """Abstract base class for all agents to implement."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with tools and configuration."""
        pass

    @abstractmethod
    async def process_task(self, task_description: str) -> AgentResult:
        """Execute a single task and return standardized result."""
        pass

    @abstractmethod
    async def process_multi_turn(self, turns: List[str]) -> AgentResult:
        """Execute multi-turn conversation and return aggregated result."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def get_name(self) -> str:
        """Return agent name."""
        return self.__class__.__name__.replace("Adapter", "")
