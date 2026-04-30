import requests
import json
import re
import logging
import time

from src.Timer import timer


# installed models: gemma4:e2b, gemma3:1b, gemma4
# gemma3:1b does not support tool calls, but is super quick
# gemma4:e2b is slower, but supports tool calls

def test_function():
    print("This is a test function to demonstrate tool calls.")
    return "TEST"


@timer
def call_ollama(
        model_name: str ="gemma4:e2b",
        tools_used: list = [],
        memory: list[dict] = []) -> dict[str, list | str]:
    """
    Calls ollama via REST API and prints the response.
    You can specify the model and system prompt.
    :param model_name: name of the model you want to use, default is "gemma4"
    :param tools_used: list of tools used by the model, default is empty list
    :param memory: list of previous messages in the conversation, default is empty list
    :return: response in the format:
    {"tool_calls": tool_calls, "thinking": thinking, "answer": answer, "full_text": full_text}
    """
    # todo: clean up function
    # todo: improve code clarity of return

    # standard API call for ollama
    url = "http://localhost:11434/api/chat"
    for memo in memory:
        print(f"{memo['role'].upper()}: {memo['content']}")

    # give model, system prompt and user prompt
    payload = {
        "model": model_name,
        "messages": memory,
        "tools": tools_used,
        "stream": False
    }

    response = requests.post(url, json=payload, stream=False)

    # handle different status codes
    match response.status_code:
        case 200:
            logging.debug("Status Code 200: Successfully called the model.")
        case 404:
            logging.debug("Status Code 404: Check if model exists.")
            return None
        case 500:
            logging.debug("Status Code 500: Server error.")
            return None
        case _:
            logging.debug(f"❌ Unexpected error: {response.status_code}")
            return None

    if not tools_used:
        full_text = ""

        # create full text from stream
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                content_piece = chunk.get("message", {}).get("content", "")
                print(content_piece, end="", flush=True)
                full_text += content_piece

        # filter thinking and answer from the full text
        thinking = re.findall(r"<think>(.*?)</think>", full_text, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
        logging.info(f"thinking: {thinking[0].strip() if thinking else 'N/A'}\n")
        logging.info(f"answer: {answer.strip() if answer else 'N/A'}")

        return {"tool_calls": [], "thinking": thinking, "answer": answer, "full_text": full_text}

    else:
        result = response.json()
        message = result.get("message", {})

        tool_calls = message.get("tool_calls", [])
        answer = message.get("content", "")

        for tool in tool_calls:
            name = tool["function"]["name"]
            args = tool["function"]["arguments"]
            print(f"\n[TOOL CALL] Model wants to use: {name}")
            print(f"[ARGUMENTS] {args}")

            # In a real agent, you would execute your python function here
            # and send the result back to Ollama for the final answer.
        return {
            "tool_calls": tool_calls,
            "thinking": [],
            "answer": answer,
            "full_text": answer
        }

example_tools = [{
    "type": "function",
    "function": {
        "name": "test_function",
        "description": "Used to test tool calls",
    }
}]
model_name = "gemma4:e2b"
user_prompt = "What is the current stock price of NVDA?"
system_prompt = "You are a helpful assistant that can provide stock prices using the get_stock_price function."
# todos:
# - deal with non thinking models
logging.basicConfig(level=logging.DEBUG)
call_ollama(model_name, [], [])
# call_ollama(model_name="gemma4")
