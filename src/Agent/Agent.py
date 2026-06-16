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
from src.API.interface import ILLMProvider
from src.Memory.MemoryManager import MemoryManager


class Agent:

    def __init__(self, model: str, system_prompt: str, tools: list[BaseTool], provider: ILLMProvider, session_id: str = "default"):
        """
        Initiates the Agent with a system prompt, memory layout, and tools.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.__name__.lower(): t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]
        self.provider = provider
        self.session_id = session_id

        # Memory Management: Token counting, compression, and persistence
        self.memory_manager = MemoryManager(
            model_name=model,
            provider=Config.PROVIDER,
            max_conversation_tokens=Config.MAX_CONVERSATION_TOKENS,
            recent_turns_to_keep=Config.RECENT_TURNS_TO_KEEP,
            compression_enabled=Config.COMPRESSION_ENABLED,
            external_memory_dir=Config.EXTERNAL_MEMORY_DIR
        )

        # Core Chat Memory: Seeded with the system instructions
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    async def agent_loop(self, user_prompt: str) -> str:
        """
        Processes a single user prompt through an execution loop until
        the model decides it has finished calling tools.
        """
        # Append the new message to persistent chat history
        self.messages.append({"role": "user", "content": user_prompt})

        while True:
            # Check and compress history if needed
            self.messages = self.memory_manager.check_and_compress(self.messages)

            response: LLMResponse = self.provider.generate(
                model_name=self.model,
                memory=self.messages,
                tools=self.tool_schemas
            )

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


# Configuration setup
async def main():
    # 1. Resolve provider and model configuration implicitly
    provider = LLMProviderFactory.get_provider()
    model = Config.MODEL_NAME

    # 2. Collect systemic execution tools
    tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool, WebFetchTool]

    # 3. Load AGENT_CONTEXT.md for project documentation
    agent_context = ""
    context_path = Path(__file__).parent.parent / "AGENT_CONTEXT.md"
    if context_path.exists():
        with open(context_path, 'r', encoding='utf-8') as f:
            agent_context = f.read()

    system_prompt = (
        "You are a helpful coding assistant that calls tools to answer questions. "
        "Here is your project context:\n\n"
        + agent_context + "\n\n"
        + "CRITICAL MANDATE: Your execution environment workspace is strictly at /app. "
        "Whenever you use WriteCodeTool, ReadCodeTool, or ReadDirectoryTool, you MUST use absolute paths starting with '/app/'. "
        "For example: '/app/sandbox_workspace/Print.py'—NEVER use relative paths like 'sandbox_workspace/Print.py'. "
        "If you need to modify a file, overwrite the exact absolute file path you read from. Do not create new directories.\n\n"
        "WebFetchTool: Use this to fetch HTTP/HTTPS content (public APIs, web data). "
        "Returns JSON with status_code, headers, and body (max 100KB). "
        "Supports GET and POST. SSRF protection blocks internal IPs (127.0.0.1, 192.168.*, 10.*, etc.). "
        "Timeout: 10s default, max 30s. If response is truncated, focus on the first 100KB and make another request if needed."
    )

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