from typing import List, Dict, Any
from src.Memory.TokenCounter import TokenCounter
from src.Memory.Compressor import HistoryCompressor
from src.Memory.ExternalMemoryBackend import ExternalMemoryBackend


class MemoryManager:
    """
    Orchestrates conversation memory: token counting, compression, and persistence.
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        max_conversation_tokens: int = 50000,
        recent_turns_to_keep: int = 10,
        compression_enabled: bool = True,
        external_memory_dir: str = "./conversation_memory",
    ):
        """
        Initialize memory manager.

        Args:
            model_name: LLM model name (e.g., "gpt-4o", "gemma4:e2b")
            provider: LLM provider ("openai" or "ollama")
            max_conversation_tokens: Token threshold for compression (default 50k)
            recent_turns_to_keep: Number of recent turns to preserve (default 10)
            compression_enabled: Whether to enable auto-compression (default True)
            external_memory_dir: Directory for storing conversation history
        """
        self.token_counter = TokenCounter(model_name, provider)
        self.compressor = HistoryCompressor(self.token_counter)
        self.external_memory = ExternalMemoryBackend(external_memory_dir)

        self.max_conversation_tokens = max_conversation_tokens
        self.recent_turns_to_keep = recent_turns_to_keep
        self.compression_enabled = compression_enabled
        self.model_name = model_name
        self.provider = provider

    def check_and_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check token count and compress history if needed.

        Call this after each LLM response to trim the message history.

        Args:
            messages: Current message history

        Returns:
            Potentially compressed message list
        """
        if not self.compression_enabled:
            return messages

        return self.compressor.compress(
            messages,
            max_tokens=self.max_conversation_tokens,
            recent_turns_to_keep=self.recent_turns_to_keep,
        )

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens in message history.

        Args:
            messages: Message history to count

        Returns:
            Total token count
        """
        return self.token_counter.count_tokens(messages)

    def save_conversation(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Persist conversation to external storage.

        Args:
            session_id: Unique identifier for this session
            messages: Message history to save
        """
        self.external_memory.save(session_id, messages)

    def load_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load previous conversation from external storage.

        Args:
            session_id: Session identifier to load

        Returns:
            Message history, or empty list if not found
        """
        return self.external_memory.load(session_id)

    def list_sessions(self) -> List[str]:
        """
        List all saved conversation sessions.

        Returns:
            List of session identifiers
        """
        return self.external_memory.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a saved conversation.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False otherwise
        """
        return self.external_memory.delete(session_id)

    def get_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get memory statistics for debugging.

        Args:
            messages: Message history

        Returns:
            Dict with stats (token_count, message_count, etc.)
        """
        return {
            "total_tokens": self.count_tokens(messages),
            "message_count": len(messages),
            "compression_enabled": self.compression_enabled,
            "max_tokens": self.max_conversation_tokens,
            "recent_turns_to_keep": self.recent_turns_to_keep,
        }
