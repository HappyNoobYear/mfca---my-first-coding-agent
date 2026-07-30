from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.API.schemas import LLMResponse


class LLMInterface(ABC):
    """Abstract interface for LLM providers (OpenAI, Ollama, etc.)."""

    @abstractmethod
    def generate(self, model_name: str, memory: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[LLMResponse]:
        """
        Generate a response from the LLM.

        Args:
            model_name: The name of the model to use (e.g., 'gpt-4o', 'gemma4:e2b').
            memory: Chat history as a list of message dicts with 'role' and 'content' keys.
            tools: Optional list of function-calling tool schemas.

        Returns:
            An LLMResponse containing the answer, tool calls, and raw response, or None on error.
        """
        pass

    @abstractmethod
    def get_context_window(self, model_name: str) -> int:
        """
        Return the model's real context window size in tokens.

        Used to derive the history-compression threshold from the actual
        model in use instead of a fixed constant, so a small local model and
        a large cloud model don't share the same compression trigger point.
        """
        pass