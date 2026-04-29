import requests
import json
import re

from src.Timer import timer


@timer
def call_ollama(model_name="gemma4:e2b", system_propmt="You are a Python assisstant", user_promt="Tell me who you are"):
    """
    Calls ollama via REST API and prints the response.
    You can specify the model and system prompt.
    :param model_name: name of the model you want to use, default is "gemma4"
    :param system_propmt: system prompt for the model
    :param user_promt: user prompt for the model
    :return: response
    """
    # standard API call for ollama
    url = "http://localhost:11434/api/chat"

    # give model, system prompt and user prompt
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_propmt},
            {"role": "user", "content": user_promt}
        ]
    }

    # print response line by line
    response = requests.post(url, json=payload, stream=True)
    # print("Recieved Response")
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(data.get("message", {}).get("content", ""), end="")

    # Optionally extract "thinking" part
    # content = response['message']['content']
    # thinking = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    # answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # print("🧠 Thought process:\n", thinking[0].strip() if thinking else "N/A")
    # print("\n✅ Final answer:\n", answer.strip())

    return response


# test gemma4, gemma3 and what happens when you call a model that doesnt exist --> todo: work around failure
call_ollama(model_name="gemma3:1b")
# call_ollama(model_name="gemma4:e2b")
# call_ollama(model_name="gemma4")
# call_ollama(model_name="gemma5")
