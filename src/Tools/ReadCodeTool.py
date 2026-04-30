from src.Tools.BaseTool import BaseTool


class ReadCodeTool(BaseTool):
    """Reads Code from file"""
    file_path: str

    def execute(self) -> str:
        """Reads the code from the given file path and returns it as a string.
        :return: The content of the code file as a string."""
        with open(self.file_path, 'r') as file:
            code = file.read()
        return code