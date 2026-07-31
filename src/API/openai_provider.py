# src/API/openai_provider.py
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.API.interface import LLMInterface
from src.API.schemas import LLMResponse, ToolCallMessage


class OpenAIProvider(LLMInterface):
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
        Implements the LLMInterface using the OpenAI Client Library.
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

            # Real API-reported usage, same idea as Ollama's prompt_eval_count/eval_count
            usage = response.usage
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

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
                raw_response=response,
                tokens_used=tokens,
            )

        except Exception as e:
            logging.error(f"OpenAI API Exception: {str(e)}")
            return None

    # OpenAI has no live introspection endpoint for context window size (unlike
    # Ollama's /api/show), so this is a static table of known models. Matched
    # by prefix since OpenAI model names carry dated suffixes (e.g. "gpt-4o-2024-08-06").
    _CONTEXT_WINDOWS = {
        "gpt-4o": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
        "o1": 200_000,
        "o3": 200_000,
    }
    _FALLBACK_CONTEXT_WINDOW = 8_192

    def get_context_window(self, model_name: str) -> int:
        """Return the known context window for model_name, or a conservative fallback."""
        for prefix, window in self._CONTEXT_WINDOWS.items():
            if model_name.startswith(prefix):
                return window
        return self._FALLBACK_CONTEXT_WINDOW