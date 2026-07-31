#!/usr/bin/env python3
"""Test script to validate both OpenAI and Ollama providers."""
import os
import sys
from src.config import Config
from src.API.factory import LLMProviderFactory
from src.API.schemas import LLMResponse

def test_provider(provider_name: str):
    """Test a specific provider."""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name.upper()} Provider")
    print(f"{'='*60}")

    os.environ["LLM_PROVIDER"] = provider_name
    Config.PROVIDER = provider_name

    try:
        provider = LLMProviderFactory.get_provider()
        print(f"[OK] Provider initialized: {type(provider).__name__}")
        print(f"[OK] Model: {Config.MODEL_NAME}")

        # Test basic generation without tools
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from [provider]!' in one sentence."}
        ]

        response = provider.generate(
            model_name=Config.MODEL_NAME,
            memory=messages,
            tools=[]
        )

        if response is None:
            print(f"[FAIL] Provider returned None")
            return False

        print(f"[OK] Response received: {response.answer[:80]}...")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        return False

if __name__ == "__main__":
    Config.validate()

    # Test current provider
    current = Config.PROVIDER
    success = test_provider(current)

    if success:
        print(f"\n[SUCCESS] {current.upper()} provider works!")
    else:
        print(f"\n[ERROR] {current.upper()} provider failed!")
        sys.exit(1)
