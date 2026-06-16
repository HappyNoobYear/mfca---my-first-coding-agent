from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ToolCallMessage:
    """Represents a single tool call from the LLM."""
    id: str  # Unique call ID for tracking in conversation history
    name: str  # Tool name (e.g., 'readcodetool')
    arguments: Dict[str, Any]  # Parsed arguments for the tool


@dataclass
class LLMResponse:
    """Unified response format from any LLM provider."""
    answer: str  # The text response from the model
    tool_calls: List[ToolCallMessage] = field(default_factory=list)  # Function calls the model wants to make
    thinking: Optional[str] = None  # Extended thinking/reasoning (if model supports it)
    raw_response: Any = None  # Original provider response object for debugging