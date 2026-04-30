from pydantic import BaseModel


class BaseTool(BaseModel):
    @classmethod
    def to_schema(cls) -> dict:
        """Returns the schema for this tool."""
        return {
            'type': 'function',
            'function': {
                'name': cls.__name__.lower(),
                # pull the description from the class docstring
                'parameters': cls.model_json_schema(),
            },
        }

