from pydantic import BaseModel


class BaseTool(BaseModel):
    """Base class for all tools. Generates OpenAI-compatible function-calling schemas from Pydantic models."""

    @classmethod
    def to_schema(cls) -> dict:
        """
        Returns an OpenAI-compatible function-calling schema.
        The function name is derived from the class name (lowercased).
        The description comes from the class docstring.
        """
        return {
            'type': 'function',
            'function': {
                'name': cls.__name__.lower(),
                'description': cls.__doc__ or '',
                'parameters': cls.model_json_schema(),
            },
        }

