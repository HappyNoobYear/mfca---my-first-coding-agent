from src.Tools.BaseTool import BaseTool
from os import listdir
from os.path import isfile, join


class ReadDirectoryTool(BaseTool):
    """Reads Directory from path"""
    directory_path: str

    def execute(self) -> str:
        """Reads the code from the given file path and returns it as a string.
        :return: The content of the code file as a string."""
        directory = listdir(self.directory_path)
        return str(directory)
