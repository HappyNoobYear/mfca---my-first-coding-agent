# src/API/openai_provider.py
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.API.interface import ILLMProvider
from src.API.schemas import LLMResponse, ToolCallMessage


class OpenAIProvider(ILLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool_calls arguments from dict back to JSON string for OpenAI API."""
        normalized = []
        for msg in messages:
            msg_copy = msg.copy()
            if "tool_calls" in msg_copy and msg_copy["tool_calls"]:
                normalized_calls = []
                for tc in msg_copy["tool_calls"]:
                    tc_copy = tc.copy()
                    if "function" in tc_copy and isinstance(tc_copy["function"].get("arguments"), dict):
                        tc_copy["function"] = tc_copy["function"].copy()
                        tc_copy["function"]["arguments"] = json.dumps(tc_copy["function"]["arguments"])
                    normalized_calls.append(tc_copy)
                msg_copy["tool_calls"] = normalized_calls
            normalized.append(msg_copy)
        return normalized

    def generate(self, model_name: str, memory: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[
        LLMResponse]:
        """
        Implements the ILLMProvider contract using the OpenAI Client Library.
        """
        # Normalize messages to ensure tool_calls arguments are JSON strings
        normalized_memory = self._normalize_messages(memory)

        kwargs = {
            "model": model_name,
            "messages": normalized_memory,
        }

        if tools:
            kwargs["tools"] = tools

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0].message
            answer = choice.content or ""

            formatted_tool_calls = []
            if choice.tool_calls:
                for tc in choice.tool_calls:
                    # OpenAI passes tool arguments as an unparsed JSON string, so we deserialize it
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = tc.function.arguments

                    print(f"\n[TOOL CALL] OpenAI wants to use: {tc.function.name}")
                    print(f"[ARGUMENTS] {args}")

                    formatted_tool_calls.append(
                        ToolCallMessage(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args
                        )
                    )

            return LLMResponse(
                answer=answer,
                tool_calls=formatted_tool_calls,
                raw_response=response
            )

        except Exception as e:
            logging.error(f"OpenAI API Exception: {str(e)}")
            return None