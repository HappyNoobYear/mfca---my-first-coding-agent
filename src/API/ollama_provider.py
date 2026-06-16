# src/API/ollama_provider.py
import os
import requests
import json
import re
import logging
from typing import List, Dict, Any, Optional

from src.Timer import timer
from src.API.interface import ILLMProvider
from src.API.schemas import LLMResponse, ToolCallMessage


class OllamaProvider(ILLMProvider):
    def __init__(self, host: str = None):
        self.ollama_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.url = f"{self.ollama_host}/api/chat"

    @timer
    def generate(self, model_name: str, memory: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[
        LLMResponse]:
        """
        Implements the ILLMProvider contract using your legacy Ollama REST API logic.
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
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    content_piece = chunk.get("message", {}).get("content", "")
                    print(content_piece, end="", flush=True)
                    full_text += content_piece

            thinking = re.findall(r"<think>(.*?)</think>", full_text, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)

            # Pack results nicely into the unified schema layout
            return LLMResponse(
                answer=answer.strip(),
                thinking=thinking[0].strip() if thinking else None,
                raw_response=full_text
            )

        # Scenario B: JSON response (Tools Used)
        else:
            result = response.json()
            message = result.get("message", {})
            raw_tool_calls = message.get("tool_calls", [])
            answer = message.get("content", "") or ""

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
                raw_response=result
            )