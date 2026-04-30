import asyncio

from src.API.API import call_ollama
from src.Tools import BaseTool
from src.Tools.ReadCodeTool import ReadCodeTool

class Agent:

    def __init__(self, model: str, system_prompt: str, tools: list[BaseTool]):
        """
        Initiates the Agent with a system prompt and a list of tools.
        :param model: Model the agent should use
        :param system_prompt: System prompt the agent should follow
        :param tools: List of tools the agent should use
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.__name__.lower(): t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]

    async def agent_loop(self, user_prompt: str) -> str:
        """
        Steps:
        1. Call the model with the system prompt and tools
        2. Check if the model wants to call a tool
        2.1 If yes, parse the tool calls
        2.2 If no, return the model response
        3. Execute the tool calls and get results
        4. Call the model again with the tool results
        :param user_prompt: User prompt for the model
        :return: response from the model
        """
        # memory of the conversation
        # TODO this needs some optimization for large conversations
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        while True:
            response = call_ollama(model_name=self.model,
                                   tools_used=self.tool_schemas,
                                   memory=messages)

            tool_calls = response.get("tool_calls", [])
            assistant_msg = {
                "role": "assistant",
                "content": response.get("answer") or "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            messages.append(assistant_msg)

            # if there are no tool calls, we can return the answer directly
            # TODO check if this could loop indefinetly if the model keeps tools
            #  max loop counter?
            if not tool_calls:
                return response.get("answer", "")

            # if there are tool calls, we need to execute them and call the model again with the results
            else:

                # process tool calls
                for t_call in tool_calls:
                    call_id = t_call.get("id")
                    response = self.execute_tool(t_call)
                    messages.append({"role": "tool",
                                     "content": response,
                                     "tool_call_id": call_id
                                     })

    def execute_tool(self, tool_call: dict):
        """Executes a tool call from the model.
        :param tool_call: dictionary that contains the tool call
        :return: response from the tool execution"""

        # find the function and its arguments
        function_name = tool_call.get("function", {}).get("name")
        args = tool_call.get("function", {}).get("arguments", {})

        # look up the class
        tool_class = self.tools.get(function_name)

        if tool_class:
            # create an instance of the tool
            tool = tool_class(**args)
            # execute the tool and get the response
            return tool.execute()

        # TODO deal with missing functions
        return f"Tool {function_name} not found."


model = "gemma4:e2b"
system_prompt = "You are a helpful coding assistant that calls tools like ReadCodeTool to answer user questions."
tools = [ReadCodeTool]
user_prompt = r"What does the Timer file (C:\Users\David\Desktop\Studium\Master\Module\SS 2026\AMT\mfca---my-first-coding-agent\src\Timer.py) do? Use the ReadCodeTool to read the file and answer the question."
test_agent = agent = Agent(model, system_prompt, tools)
response = asyncio.run(test_agent.agent_loop(user_prompt))
print(response)
