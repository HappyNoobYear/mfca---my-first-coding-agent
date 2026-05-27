from src.Tools.BaseTool import BaseTool


class ReadCodeTool(BaseTool):
    """Reads Code from file"""
    file_path: str

    def execute(self) -> str:
        """Reads the code from the given file path and returns it as a string.
        :return: The content of the code file as a string."""
        import os
        resolved_path = self.file_path
        
        # Auto-resolve absolute paths (like /src/...) by checking if they exist under the /app workspace
        if os.path.isabs(resolved_path) and not resolved_path.startswith("/app"):
            test_path = os.path.join("/app", resolved_path.lstrip("/"))
            if os.path.exists(test_path):
                resolved_path = test_path

        try:
            with open(resolved_path, 'r', encoding='utf-8') as file:
                code = file.read()
            return code
        except FileNotFoundError:
            return f"Error: File '{self.file_path}' not found. Please verify the file path (e.g. check if you should prefix with /app or use a relative path)."
        except Exception as e:
            return f"Error reading file '{self.file_path}': {str(e)}"