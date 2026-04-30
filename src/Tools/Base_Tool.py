from pydantic import BaseModel


class BaseTool(BaseModel):
    @classmethod
    def to_schema(cls) -> dict:
        """Returns the schema for this tool."""
        return {
            'type': 'function',
            'function': {
                'name': cls.__name__.lower(),
                # We pull the description from the class docstring!
                'parameters': cls.model_json_schema(),
            },
        }


class GetStockPrice(BaseTool):
    """Fetches the current trading price for a given stock ticker."""
    symbol: str

    def execute(self) -> str:
        """Executes the tool and returns the stock price."""
        # TODO implement this function
        return "The current stock price of NVDA is $500."


example_class = GetStockPrice
print(example_class.to_schema())
