import asyncio
import os

from src.API.API import call_ollama
from src.Tools import BaseTool
from src.Tools.ReadCodeTool import ReadCodeTool
from src.Tools.ReadDirectoryTool import ReadDirectoryTool
from src.Tools.ExecuteCodeTool import ExecuteCodeTool
from src.Tools.WriteCodeTool import WriteCodeTool


class Agent:

    def __init__(self, model: str, system_prompt: str, tools: list[BaseTool]):
        """
        Initiates the Agent with a system prompt, memory layout, and tools.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.__name__.lower(): t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]

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
            response = call_ollama(model_name=self.model,
                                   tools_used=self.tool_schemas,
                                   memory=self.messages)

            # API Error Fallback: Check if response is completely empty (None)
            if response is None:
                error_msg = "Error: Did not receive a valid response from the Ollama API."
                self.messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            tool_calls = response.get("tool_calls", [])
            assistant_msg = {
                "role": "assistant",
                "content": response.get("answer") or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            self.messages.append(assistant_msg)

            # If no tools to process, yield the final textual answer
            if not tool_calls:
                return response.get("answer", "")

            # Execute requested tools sequentially
            else:
                for t_call in tool_calls:
                    call_id = t_call.get("id")
                    tool_output = self.execute_tool(t_call)

                    self.messages.append({
                        "role": "tool",
                        "content": tool_output,
                        "tool_call_id": call_id
                    })

    def execute_tool(self, tool_call: dict):
        """Executes a tool call from the model."""
        function_name = tool_call.get("function", {}).get("name")
        args = tool_call.get("function", {}).get("arguments", {})

        tool_class = self.tools.get(function_name)

        if tool_class:
            try:
                tool = tool_class(**args)
                return tool.execute()
            except Exception as e:
                return f"Error executing tool {function_name}: {str(e)}"

        return f"Tool {function_name} not found."

    async def start_chat(self):
        """Launches an interactive loop for continuous live chat."""
        print("\n🤖 Coding Agent Initialized. Type 'exit' or 'quit' to stop.")
        print("-" * 60)

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

# todo later build chat that asks for model and user prompt
# todo move system prompt to own file?
# model = "gemma4:e2b"
# system_prompt = (
#     "You are a helpful coding assistant that calls tools like ReadCodeTool or ReadDirectoryTool to answer user questions. "
#     "CRITICAL MANDATE: Your execution environment workspace is strictly at /app. "
#     "Whenever you use WriteCodeTool, ReadCodeTool, or ReadDirectoryTool, you MUST use absolute paths starting with '/app/'. "
#     "For example, you must use '/app/sandbox_workspace/Print.py'—NEVER use relative paths like 'sandbox_workspace/Print.py'. "
#     "If you need to modify a file, overwrite the exact absolute file path you read from. Do not create new directories."
# )
# tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool]
# TODO move this into tests
# test ReadCodeTool
# user_prompt = r"What does the Timer file (C:\Users\David\Desktop\Studium\Master\Module\SS 2026\AMT\mfca---my-first-coding-agent\src\Timer.py) do? Use the ReadCodeTool to read the file and answer the question."
# test ReadDirectoryTool
# user_prompt = r"Show me the files in the directory (C:\Users\David\Desktop\Studium\Master\Module\SS 2026\AMT\mfca---my-first-coding-agent\src\). Use the ReadDirectoryTool. If you see a file called Timer.py analyze it usig the ReadCodeTool."
# test execute tool
# user_prompt = "Write a quick python script that prints 'Hello from Docker Sandbox!' and run it using your ExecuteCodeTool."
# test write code tool
# user_prompt = "Write a quick python script that prints 'Hello from Docker Sandbox!' and save it as hello.py using your WriteCodeTool. Then use ExecuteCodeTool to run the script."
# user_prompt = "What is your name?"
# print to loggin prompt
# user_prompt = "Locate Print.py inside the sandbox_workspace directory and improve the code. I want it to use logging instead of print."
# test modifying code ability
# user_prompt = "Locate Print.py inside the sandbox_workspace directory. I want it to change the name of the function to test_logging"
# user_prompt = "Locate Print.py inside the sandbox_workspace directory and execute it."

# test_agent = agent = Agent(model, system_prompt, tools)
# response = asyncio.run(test_agent.agent_loop(user_prompt))
# print(response)

# asyncio.run(test_agent.start_chat())

# this no longer works
# call docker compose up --build from the terminal instead


# Configuration setup
model = "gemma4:e2b"
system_prompt = (
    "You are a helpful coding assistant that calls tools like ReadCodeTool or ReadDirectoryTool to answer user questions. "
    "CRITICAL MANDATE: Your execution environment workspace is strictly at /app. "
    "Whenever you use WriteCodeTool, ReadCodeTool, or ReadDirectoryTool, you MUST use absolute paths starting with '/app/'. "
    "For example, you must use '/app/sandbox_workspace/Print.py'—NEVER use relative paths like 'sandbox_workspace/Print.py'. "
    "If you need to modify a file, overwrite the exact absolute file path you read from. Do not create new directories."
)
tools = [ReadCodeTool, ReadDirectoryTool, ExecuteCodeTool, WriteCodeTool]

# Instantiation and execution entry point
if __name__ == "__main__":
    test_agent = Agent(model, system_prompt, tools)
    asyncio.run(test_agent.start_chat())