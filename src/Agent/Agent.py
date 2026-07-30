import asyncio
from pathlib import Path

from src.Tools import BaseTool
from src.config import Config
from src.API.factory import LLMProviderFactory
from src.Tools.ReadCodeTool import ReadCodeTool
from src.Tools.ReadDirectoryTool import ReadDirectoryTool
from src.Tools.ExecuteCodeTool import ExecuteCodeTool
from src.Tools.WriteCodeTool import WriteCodeTool
from src.Tools.WebFetchTool import WebFetchTool
from src.API.schemas import LLMResponse, ToolCallMessage
from src.API.interface import LLMInterface
from src.Memory.MemoryManager import MemoryManager


class Agent:

    def __init__(self, model: str, system_prompt: str, tools: list[BaseTool], provider: LLMInterface, session_id: str = "default"):
        """
        Initiates the Agent with a system prompt, memory layout, and tools.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.__name__.lower(): t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]
        self.provider = provider
        self.session_id = session_id

        # Derive the compression threshold from the model's real context
        # window rather than a fixed constant, so a small local model and a
        # large cloud model don't share the same compression trigger point.
        # Falls back to Config.MAX_CONVERSATION_TOKENS if the provider can't
        # determine the model's context window (e.g. Ollama /api/show unreachable).
        try:
            context_window = self.provider.get_context_window(model)
            max_conversation_tokens = int(context_window * Config.COMPRESSION_CONTEXT_RATIO)
        except Exception:
            max_conversation_tokens = Config.MAX_CONVERSATION_TOKENS

        # Memory Management: Token counting, compression, and persistence
        self.memory_manager = MemoryManager(
            model_name=model,
            provider=Config.PROVIDER,
            max_conversation_tokens=max_conversation_tokens,
            recent_turns_to_keep=Config.RECENT_TURNS_TO_KEEP,
            compression_enabled=Config.COMPRESSION_ENABLED,
            external_memory_dir=Config.EXTERNAL_MEMORY_DIR,
            llm_provider=self.provider,
        )

        # Core Chat Memory: Seeded with the system instructions
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Accumulated real API token count for the most recent agent_loop() call
        self._last_tokens_used: int = 0

    # Keywords that signal each tool is relevant to the current task.
    _TOOL_KEYWORDS: dict[str, set[str]] = {
        "readcodetool":      {"read", "show", "view", "open", "file", "code", "content",
                              "look", "explain", "what", "see", "display", "check", "print"},
        "readdirectorytool": {"directory", "folder", "list", "files", "structure",
                              "ls", "find", "exist", "tree"},
        "writecodetool":     {"write", "create", "make", "modify", "update", "change",
                              "add", "implement", "fix", "edit", "build", "concise",
                              "refactor", "script", "save"},
        "executecodetool":   {"run", "execute", "test", "output", "result", "compute",
                              "print", "numbers"},
        "webfetchtool":      {"web", "http", "https", "url", "fetch", "download",
                              "api", "request", "website"},
    }

    def _select_schemas(self, user_prompt: str) -> list:
        """Return only the tool schemas relevant to the current prompt + recent context."""
        if not Config.SELECTIVE_TOOLS_ENABLED:
            return self.tool_schemas

        # Build context from the current prompt and the last few messages.
        context_parts = [user_prompt.lower()]
        for msg in self.messages[-4:]:
            if isinstance(msg.get("content"), str):
                context_parts.append(msg["content"].lower())
        context = " ".join(context_parts)

        selected = [
            schema for schema in self.tool_schemas
            if any(
                kw in context
                for kw in self._TOOL_KEYWORDS.get(
                    schema["function"]["name"], set()
                )
            )
        ]
        return selected if selected else self.tool_schemas

    async def agent_loop(self, user_prompt: str) -> str:
        """
        Processes a single user prompt through an execution loop until
        the model decides it has finished calling tools.
        """
        # Reset per-call token counter
        self._last_tokens_used = 0

        # Append the new message to persistent chat history
        self.messages.append({"role": "user", "content": user_prompt})

        # Select schemas once for this turn so inner tool-call iterations are consistent.
        active_schemas = self._select_schemas(user_prompt)

        while True:
            # Check and compress history if needed
            self.messages = self.memory_manager.check_and_compress(self.messages)

            response: LLMResponse = self.provider.generate(
                model_name=self.model,
                memory=self.messages,
                tools=active_schemas
            )

            # Accumulate real API token counts across all inner iterations
            if response is not None:
                self._last_tokens_used += response.tokens_used

            # API Error Fallback: Check if response is completely empty (None)
            if response is None:
                error_msg = "Error: Did not receive a valid response from the Ollama API."
                self.messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            assistant_msg = {
                "role": "assistant",
                "content": response.answer,
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                            # Note: Ollama takes dict, OpenAI provider handles its string conversion internally
                        }
                    } for tc in response.tool_calls
                ]

            self.messages.append(assistant_msg)

            # If no tools to process, yield the final textual answer
            if not response.tool_calls:
                return response.answer

            # Execute requested tools sequentially
            for tc in response.tool_calls:
                tool_output = self.execute_tool(tc)

                # Append tool outcome back into conversation memory context
                self.messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "tool_call_id": tc.id
                })

    def execute_tool(self, tool_call: ToolCallMessage):
        """Executes a tool call from the model."""
        tool_class = self.tools.get(tool_call.name.lower())

        if tool_class:
            try:
                # Unpack arguments directly into the class constructor
                tool = tool_class(**tool_call.arguments)
                return tool.execute()
            except Exception as e:
                return f"Error executing tool {tool_call.name}: {str(e)}"

        return f"Tool {tool_call.name} not found."

    async def start_chat(self):
        """Launches an interactive loop for continuous live chat."""
        print("\n Coding Agent Initialized. Type 'exit' or 'quit' to stop.")
        print("-" * 60)

        try:
            while True:
                try:
                    user_prompt = input("\nYou: ")

                    if user_prompt.strip().lower() in ["exit", "quit"]:
                        print("Goodbye!")
                        break

                    if not user_prompt.strip():
                        continue

                    print("\nThinking...")
                    response = await self.agent_loop(user_prompt)

                    print(f"\nAgent: {response}")
                    print("-" * 60)

                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye!")
                    break
        finally:
            # Save conversation before exiting
            self.memory_manager.save_conversation(self.session_id, self.messages)
            stats = self.memory_manager.get_stats(self.messages)
            print(f"\n[Session saved] Tokens: {stats['total_tokens']}, Messages: {stats['message_count']}")


def build_system_prompt(include_project_context: bool = True) -> str:
    """The real system prompt mfca runs with -- including the CRITICAL MANDATE
    that tells the model it must use WriteCodeTool/ExecuteCodeTool for file
    operations. Shared by main() and the benchmark harness so they can't
    silently drift apart the way they did before (the benchmark was passing
    a stripped-down generic prompt missing this mandate entirely, which is
    why mfca in benchmarks never reached for WriteCodeTool the way it does
    in normal use).

    include_project_context controls whether AGENT_CONTEXT.md (this repo's
    own documentation) gets injected -- relevant for real usage, irrelevant
    noise for generic benchmark tasks that aren't about this repository.
    """
    agent_context = ""
    if include_project_context:
        context_path = Path(__file__).parent.parent / "AGENT_CONTEXT.md"
        if context_path.exists():
            with open(context_path, 'r', encoding='utf-8') as f:
                agent_context = f.read()

    return (
        "You are a helpful coding assistant that calls tools to answer questions. "
        + ("Here is your project context:\n\n" + agent_context + "\n\n" if agent_context else "")
        + "CRITICAL MANDATE: When writing or executing files, you MUST use /app/sandbox_workspace/ as the directory. "
        "WriteCodeTool and ExecuteCodeTool only work with files in that directory — writing anywhere else will fail. "
        "Always write to paths like '/app/sandbox_workspace/my_script.py'. "
        "You may read any file under /app/ (e.g. /app/src/...) using ReadCodeTool. "
        "Do not create new subdirectories.\n\n"
        "WebFetchTool: Use this to fetch HTTP/HTTPS content (public APIs, web data). "
        "Returns JSON with status_code, headers, and body (max 100KB). "
        "Supports GET and POST. SSRF protection blocks internal IPs (127.0.0.1, 192.168.*, 10.*, etc.). "
        "Timeout: 10s default, max 30s. If response is truncated, focus on the first 100KB and make another request if needed."
    )


# Configuration setup
async def main():
    # 1. Resolve provider and model configuration implicitly
    provider = LLMProviderFactory.get_provider()
    model = Config.MODEL_NAME

    # 2. Collect systemic execution tools
    tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool, WebFetchTool]

    # 3. Build the real system prompt, including project context
    system_prompt = build_system_prompt(include_project_context=True)

    # 4. Instantiate the Agent with our clean abstractions
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        provider=provider,
        session_id="default"
    )

    # 5. Start processing
    await agent.start_chat()


if __name__ == "__main__":
    asyncio.run(main())