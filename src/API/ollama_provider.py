# src/API/ollama_provider.py
import os
import requests
import json
import re
import logging
from typing import List, Dict, Any, Optional

from src.Timer import timer
from src.API.interface import LLMInterface
from src.API.schemas import LLMResponse, ToolCallMessage


class OllamaProvider(LLMInterface):
    def __init__(self, host: str = None):
        raw = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Normalize bare IPs/hostnames to a full URL with port.
        if raw and not raw.startswith("http"):
            raw = f"http://{raw}:11434"
        # 0.0.0.0 is a listen address, not a client address
        # connect to localhost.
        raw = raw.replace("0.0.0.0", "localhost")
        self.ollama_host = raw
        self.url = f"{self.ollama_host}/api/chat"
        self.show_url = f"{self.ollama_host}/api/show"
        self._context_window_cache: Dict[str, int] = {}

    @timer
    def generate(self, model_name: str, memory: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[
        LLMResponse]:
        """
        Implements the LLMInterface using the Ollama REST API logic.
        """
        # Determine streaming based on if tools are being passed
        stream_mode = not bool(tools)

        payload = {
            "model": model_name,
            "messages": memory,
            "tools": tools,
            "stream": stream_mode
        }

        try:
            response = requests.post(self.url, json=payload, stream=stream_mode)
        except Exception as e:
            logging.error(f"Connection error to Ollama: {str(e)}")
            return None

        # Handle different status codes
        match response.status_code:
            case 200:
                logging.debug("Status Code 200: Successfully called the model.")
            case 404:
                logging.debug("Status Code 404: Check if model exists.")
                return None
            case 500:
                logging.debug("Status Code 500: Server error.")
                return None
            case _:
                logging.debug(f"Unexpected error: {response.status_code}")
                return None

        # Scenario A: Streaming Text (No Tools Used)
        if stream_mode:
            full_text = ""
            last_chunk = {}
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    content_piece = chunk.get("message", {}).get("content", "")
                    print(content_piece, end="", flush=True)
                    full_text += content_piece
                    last_chunk = chunk

            thinking = re.findall(r"<think>(.*?)</think>", full_text, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            tokens = last_chunk.get("prompt_eval_count", 0) + last_chunk.get("eval_count", 0)

            return LLMResponse(
                answer=answer.strip(),
                thinking=thinking[0].strip() if thinking else None,
                raw_response=full_text,
                tokens_used=tokens,
            )

        # Scenario B: JSON response (Tools Used)
        else:
            result = response.json()
            message = result.get("message", {})
            raw_tool_calls = message.get("tool_calls", [])
            answer = message.get("content", "") or ""
            tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)

            # Convert Ollama's inner format to our standard ToolCallMessage format
            formatted_tool_calls = []
            for tool in raw_tool_calls:
                name = tool["function"]["name"]
                args = tool["function"]["arguments"]
                print(f"\n[TOOL CALL] Model wants to use: {name}")
                print(f"[ARGUMENTS] {args}")

                formatted_tool_calls.append(
                    ToolCallMessage(
                        id=tool.get("id", ""),  # Ollama sometimes omits call IDs depending on version
                        name=name,
                        arguments=args
                    )
                )

            return LLMResponse(
                answer=answer,
                tool_calls=formatted_tool_calls,
                raw_response=result,
                tokens_used=tokens,
            )

    # Conservative fallback if /api/show is unreachable or the response has no
    # recognizable context-length field (e.g. an unusual or very old model).
    _FALLBACK_CONTEXT_WINDOW = 4096

    def get_context_window(self, model_name: str) -> int:
        """Query Ollama's /api/show for the model's real context length.

        The context-length key is prefixed by the model's architecture (e.g.
        "llama.context_length", "gemma3.context_length"), so rather than
        hardcoding architecture names we scan model_info for any key ending
        in "context_length" and take the first match.
        """
        if model_name in self._context_window_cache:
            return self._context_window_cache[model_name]

        try:
            resp = requests.post(self.show_url, json={"model": model_name}, timeout=10)
            if resp.status_code == 200:
                model_info = resp.json().get("model_info", {})
                for key, value in model_info.items():
                    if key.endswith("context_length") and isinstance(value, int):
                        self._context_window_cache[model_name] = value
                        return value
        except Exception as e:
            logging.warning(
                f"Could not determine context window for '{model_name}' via /api/show "
                f"({e}); falling back to {self._FALLBACK_CONTEXT_WINDOW}."
            )

        # Deliberately not cached: a transient failure (Ollama still
        # starting up, a momentary network issue) should not lock this
        # provider instance into the fallback for the rest of its life.
        # Only a genuine successful lookup gets cached, so the next call
        # can retry instead of repeating a stale failure forever.
        return self._FALLBACK_CONTEXT_WINDOW