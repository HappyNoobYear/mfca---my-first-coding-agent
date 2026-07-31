from typing import List, Dict, Any, Optional
from src.Memory.TokenCounter import TokenCounter


class HistoryCompressor:
    """Compresses conversation history using sliding window + summary strategy."""

    def __init__(
        self,
        token_counter: TokenCounter,
        llm_provider: Optional[Any] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize compressor.

        Args:
            token_counter: TokenCounter instance for measuring tokens
            llm_provider: Optional LLMInterface used to write the old-turns
                summary with a real model call instead of truncating each
                message. If not given, summarization falls back to the
                deterministic, truncation-based version.
            model_name: Model to use for that summarization call.
        """
        self.token_counter = token_counter
        self.llm_provider = llm_provider
        self.model_name = model_name

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

        Tries a real model call first, since a truncated concatenation of
        each message cuts at a fixed character count with no regard for
        where the important part of a message actually is, and truncating
        a tool call's raw arguments (e.g. a full code body) produces a
        broken code fragment, not a useful shorter version. Falls back to
        the deterministic version if no provider was given, or if the call
        fails, so compression never crashes the agent outright.

        Returns:
            A message dict representing the summary
        """
        transcript = self._render_transcript(old_messages)

        if self.llm_provider is not None and transcript.strip():
            model_summary = self._summarize_with_model(transcript)
            if model_summary:
                return {"role": "assistant", "content": model_summary}

        return {"role": "assistant", "content": self._fallback_summary(old_messages)}

    def _render_transcript(self, old_messages: List[Dict[str, Any]]) -> str:
        """Render old messages into one plain-text transcript for the model
        to summarize. Tool calls and tool results are rendered as short
        facts (tool name, short arguments, result text) rather than passed
        through raw, since a tool call's arguments can contain a full code
        body that would dominate the transcript without adding anything a
        summary needs.
        """
        lines = []
        for msg in old_messages:
            role = msg.get("role")
            if role == "user":
                lines.append(f"User: {msg.get('content', '')}")
            elif role == "assistant":
                content = msg.get("content", "")
                if content:
                    lines.append(f"Agent: {content}")
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) or {}
                    args = self._short_args(fn.get("arguments"))
                    lines.append(f"Agent called {fn.get('name', 'a tool')}({args})")
            elif role == "tool":
                output = (msg.get("content", "") or "")[:300]
                lines.append(f"Tool result: {output}")
        return "\n".join(lines)

    @staticmethod
    def _short_args(args: Any) -> str:
        """Render only the short arguments of a tool call, e.g. a filename,
        and skip long ones, e.g. a full code body, that would swamp the
        transcript without telling a summary anything useful.
        """
        parts = []
        for key, value in (args or {}).items():
            value_str = str(value)
            if len(value_str) <= 60:
                parts.append(f"{key}={value_str}")
        return ", ".join(parts)

    def _summarize_with_model(self, transcript: str) -> Optional[str]:
        """One model call per compression, not per message. Returns None on
        any failure so the caller falls back to the deterministic summary
        instead of the agent crashing over a summarization error.
        """
        try:
            response = self.llm_provider.generate(
                model_name=self.model_name,
                memory=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following part of a conversation between a "
                            "user and a coding agent. Keep concrete facts: files created "
                            "or modified, code written, commands run, results, and any "
                            "constraints the user gave. Be concise."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                tools=[],
            )
        except Exception:
            return None

        if response is None or not response.answer:
            return None
        return response.answer

    def _fallback_summary(self, old_messages: List[Dict[str, Any]]) -> str:
        """Deterministic summary used when no model is available or the
        summarization call fails. Tool activity is still recorded as short
        facts rather than dropped, matching what the model prompt asks for,
        so this differs from the model-written version only in how it
        turns text into something shorter, not in what it tries to keep.
        """
        lines = [
            "[Summary of earlier conversation turns - older context compressed for token efficiency]"
        ]

        turn_count = 0
        for msg in old_messages:
            role = msg.get("role")
            if role == "user":
                turn_count += 1
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                lines.append(f"Turn {turn_count} (user): {content}")

            elif role == "assistant":
                content = msg.get("content", "")
                if content:
                    if len(content) > 150:
                        content = content[:150] + "..."
                    lines.append(f"  -> Agent responded: {content}")
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) or {}
                    args = self._short_args(fn.get("arguments"))
                    lines.append(f"  -> Agent called {fn.get('name', 'a tool')}({args})")

            elif role == "tool":
                output = (msg.get("content", "") or "")[:150]
                lines.append(f"  -> Tool result: {output}")

        return "\n".join(lines)
