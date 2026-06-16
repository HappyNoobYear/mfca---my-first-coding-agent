"""Adapter for mfca agent to benchmark interface."""

import asyncio
import json
from typing import List

from src.Agent.Agent import Agent
from src.API.factory import LLMProviderFactory
from src.config import Config
from src.Tools.ReadCodeTool import ReadCodeTool
from src.Tools.ReadDirectoryTool import ReadDirectoryTool
from src.Tools.ExecuteCodeTool import ExecuteCodeTool
from src.Tools.WriteCodeTool import WriteCodeTool
from src.Tools.WebFetchTool import WebFetchTool

from benchmarks.agent_interface import AgentInterface, AgentResult


class MFCAAdapter(AgentInterface):
    """Adapter for mfca agent."""

    def __init__(self):
        self.agent = None
        self.session_id = "benchmark"

    async def initialize(self) -> None:
        """Initialize mfca agent with tools."""
        provider = LLMProviderFactory.get_provider()
        model = Config.MODEL_NAME

        tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool, WebFetchTool]

        system_prompt = (
            "You are a helpful coding assistant that calls tools to answer questions. "
            "Use the tools available to you to complete tasks efficiently. "
            "Be concise and direct in your responses."
        )

        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            provider=provider,
            session_id=self.session_id,
        )

    async def process_task(self, task_description: str) -> AgentResult:
        """Execute a single task."""
        try:
            if not self.agent:
                await self.initialize()

            response = await self.agent.agent_loop(task_description)

            # Get token stats from memory manager
            stats = self.agent.memory_manager.get_stats(self.agent.messages)

            return AgentResult(
                success=True,
                output=response,
                tokens_used=stats["total_tokens"],
                tool_calls=self._count_tool_calls(),
                turns_completed=1,
                metadata={"provider": "mfca", "model": Config.MODEL_NAME},
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

            total_tokens = 0
            total_tool_calls = 0

            for turn in turns:
                response = await self.agent.agent_loop(turn)
                stats = self.agent.memory_manager.get_stats(self.agent.messages)
                total_tokens = stats["total_tokens"]
                total_tool_calls += self._count_tool_calls()

            return AgentResult(
                success=True,
                output=response,
                tokens_used=total_tokens,
                tool_calls=total_tool_calls,
                turns_completed=len(turns),
                metadata={
                    "provider": "mfca",
                    "model": Config.MODEL_NAME,
                    "compression_enabled": Config.COMPRESSION_ENABLED,
                },
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
        if self.agent:
            self.agent.memory_manager.save_conversation(
                self.session_id, self.agent.messages
            )

    def _count_tool_calls(self) -> int:
        """Count tool calls in recent message history."""
        count = 0
        for msg in self.agent.messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                count += len(msg["tool_calls"])
        return count
