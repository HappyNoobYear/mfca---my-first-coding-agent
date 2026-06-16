from typing import List, Dict, Any
import json


class TokenCounter:
    """Counts tokens in messages for different LLM providers."""

    def __init__(self, model_name: str, provider: str):
        """
        Initialize token counter for a specific model and provider.

        Args:
            model_name: e.g., "gpt-4o" or "gemma4:e2b"
            provider: "openai" or "ollama"
        """
        self.model_name = model_name
        self.provider = provider
        self.tiktoken_enc = None

        if provider == "openai":
            try:
                import tiktoken
                self.tiktoken_enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                # Fallback if tiktoken fails or model not found
                self.tiktoken_enc = None

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += self.count_message(msg)
        return total

    def count_message(self, message: Dict[str, Any]) -> int:
        """Count tokens in a single message dict."""
        if self.provider == "openai" and self.tiktoken_enc:
            return self._count_openai(message)
        else:
            return self._count_approximate(message)

    def _count_openai(self, message: Dict[str, Any]) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        content = message.get("content", "")
        if isinstance(content, str):
            return len(self.tiktoken_enc.encode(content))
        # If content is dict (shouldn't happen in our format), estimate
        return len(self.tiktoken_enc.encode(json.dumps(content)))

    def _count_approximate(self, message: Dict[str, Any]) -> int:
        """Approximate token count: ~4 characters = 1 token."""
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content) // 4 + 1
        # If content is complex, estimate via JSON
        return len(json.dumps(content)) // 4 + 1
