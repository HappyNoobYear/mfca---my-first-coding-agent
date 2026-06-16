# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the root directory and load the .env file
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class Config:
    """Centralized configuration manager."""

    PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
    MODEL_NAME = os.getenv("LLM_MODEL")

    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Ollama Settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Token Management & History Compression
    MAX_TOKENS_PER_MESSAGE = int(os.getenv("MAX_TOKENS_PER_MESSAGE", "2000"))
    MAX_CONVERSATION_TOKENS = int(os.getenv("MAX_CONVERSATION_TOKENS", "50000"))
    RECENT_TURNS_TO_KEEP = int(os.getenv("RECENT_TURNS_TO_KEEP", "10"))
    COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
    EXTERNAL_MEMORY_DIR = os.getenv("EXTERNAL_MEMORY_DIR", "./conversation_memory")

    # WebFetchTool Configuration
    WEBFETCH_ENABLED = os.getenv("WEBFETCH_ENABLED", "true").lower() == "true"
    WEBFETCH_MAX_RESPONSE_SIZE = int(os.getenv("WEBFETCH_MAX_RESPONSE_SIZE", "102400"))  # 100KB
    WEBFETCH_DEFAULT_TIMEOUT = int(os.getenv("WEBFETCH_DEFAULT_TIMEOUT", "10"))  # seconds
    WEBFETCH_MAX_TIMEOUT = int(os.getenv("WEBFETCH_MAX_TIMEOUT", "30"))  # seconds
    WEBFETCH_FOLLOW_REDIRECTS = os.getenv("WEBFETCH_FOLLOW_REDIRECTS", "true").lower() == "true"
    WEBFETCH_BLOCKLIST_INTERNAL_IPS = os.getenv("WEBFETCH_BLOCKLIST_INTERNAL_IPS", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validates critical environment configuration at runtime initialization."""
        if cls.PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("❌ Initialization Error: OPENAI_API_KEY is missing in the environment.")

        # Automatically assign defaults based on provider if no model was explicitly configured
        if not cls.MODEL_NAME:
            cls.MODEL_NAME = "gpt-4o" if cls.PROVIDER == "openai" else "gemma4:e2b"