from typing import List, Dict, Any
from src.Memory.TokenCounter import TokenCounter


class HistoryCompressor:
    """Compresses conversation history using sliding window + summary strategy."""

    def __init__(self, token_counter: TokenCounter):
        """
        Initialize compressor.

        Args:
            token_counter: TokenCounter instance for measuring tokens
        """
        self.token_counter = token_counter

    def compress(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 50000,
        recent_turns_to_keep: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Compress conversation history using sliding window + summary.

        Keeps the last N complete turns in full context. If total tokens exceed
        max_tokens, replaces older turns with a single summary turn.

        Args:
            messages: Full message history
            max_tokens: Token threshold (default 50k)
            recent_turns_to_keep: Number of complete turns to keep (default 10)

        Returns:
            Compressed message list
        """
        if not messages or len(messages) < 2:
            return messages

        total_tokens = self.token_counter.count_tokens(messages)

        # No compression needed
        if total_tokens <= max_tokens:
            return messages

        # Separate system message (always kept)
        system_msg = None
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            other_messages = messages[1:]
        else:
            other_messages = messages

        if not other_messages:
            return messages

        # Identify recent complete turns (user + assistant + tool responses)
        recent_turn_indices = self._get_recent_complete_turns(
            other_messages, recent_turns_to_keep
        )

        if not recent_turn_indices:
            # Can't identify turns, return original
            return messages

        # Split into old and recent
        old_messages = other_messages[: recent_turn_indices[0]]
        recent_messages = other_messages[recent_turn_indices[0] :]

        # Create summary of old messages
        if old_messages:
            summary_msg = self._create_summary_turn(old_messages)
            compressed = [system_msg] if system_msg else []
            compressed.append(summary_msg)
            compressed.extend(recent_messages)
            return compressed

        # No old messages to compress
        result = [system_msg] if system_msg else []
        result.extend(other_messages)
        return result

    def _get_recent_complete_turns(
        self, messages: List[Dict[str, Any]], num_turns: int
    ) -> List[int]:
        """
        Get indices of the most recent N complete turns.

        A turn = one user message + one assistant message + optional tool messages.

        Returns:
            List of indices where turns start
        """
        turn_starts = []
        i = len(messages) - 1

        # Work backwards from the end
        while i >= 0 and len(turn_starts) < num_turns:
            msg = messages[i]

            if msg.get("role") == "user":
                # Found a user message, this is a turn start
                turn_starts.insert(0, i)
                i -= 1
            else:
                i -= 1

        return turn_starts if turn_starts else []

    def _create_summary_turn(self, old_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary turn from old messages.

        Returns:
            A message dict representing the summary
        """
        # Build a narrative summary of what happened
        summary_lines = []
        summary_lines.append(
            "[Summary of earlier conversation turns - older context compressed for token efficiency]"
        )

        turn_count = 0
        for msg in old_messages:
            if msg.get("role") == "user":
                turn_count += 1
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                summary_lines.append(f"Turn {turn_count} (user): {content}")

            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                summary_lines.append(f"  → Agent responded: {content}")

        summary_text = "\n".join(summary_lines)

        return {
            "role": "assistant",
            "content": summary_text,
        }
