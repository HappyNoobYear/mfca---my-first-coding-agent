from src.Tools.BaseTool import BaseTool
from os import listdir
from os.path import isfile, join


class ReadDirectoryTool(BaseTool):
    """Reads Directory from path"""
    directory_path: str

    def execute(self) -> str:
        """Reads the directory from the given directory path and returns its contents.
        :return: List of files and folders as a string representation."""
        import os
        resolved_path = self.directory_path

        # Auto-resolve absolute paths (like /src/...) by checking if they exist under the /app workspace
        if os.path.isabs(resolved_path) and not resolved_path.startswith("/app"):
            test_path = os.path.join("/app", resolved_path.lstrip("/"))
            if os.path.exists(test_path) and os.path.isdir(test_path):
                resolved_path = test_path

        try:
            directory = listdir(resolved_path)
            return str(directory)
        except FileNotFoundError:
            return f"Error: Directory '{self.directory_path}' not found. Please verify the directory path (e.g. check if you should prefix with /app or use a relative path)."
        except Exception as e:
            return f"Error listing directory '{self.directory_path}': {str(e)}"
