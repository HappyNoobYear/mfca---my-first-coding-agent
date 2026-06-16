from src.config import Config
from src.API.interface import ILLMProvider
from src.API.ollama_provider import OllamaProvider
from src.API.openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory to safely instantiate LLM providers based on application configuration."""

    @staticmethod
    def get_provider() -> ILLMProvider:
        """
        Instantiate and return an LLM provider based on LLM_PROVIDER config.

        Returns:
            An ILLMProvider instance (OpenAIProvider or OllamaProvider).

        Raises:
            ValueError: If LLM_PROVIDER is not 'openai' or 'ollama'.
        """
        # Validate settings first
        Config.validate()

        if Config.PROVIDER == "openai":
            # Injecting the token securely from central configuration
            return OpenAIProvider(api_key=Config.OPENAI_API_KEY)

        elif Config.PROVIDER == "ollama":
            return OllamaProvider(host=Config.OLLAMA_HOST)

        else:
            raise ValueError(
                f"Unsupported LLM provider: {Config.PROVIDER}. "
                f"Valid options are: 'openai', 'ollama'"
            )