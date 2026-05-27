import os
from src.Tools.BaseTool import BaseTool


class WriteCodeTool(BaseTool):
    """Writes Python code into the secure shared workspace."""
    filename: str
    code_content: str

    def execute(self) -> str:
        # Resolve the absolute path under /workspace
        # normpath resolves any relative segments like '.' or '..'
        target_path = os.path.normpath(os.path.join("/app", self.filename))

        # Prevent path traversal outside the safe sandbox directory
        if not target_path.startswith("/app"):
            return f"Security Error: Access denied. The path must stay inside the isolated workspace. Attempted: {self.filename}"

        # Create parent directories if a nested file structure is requested
        parent_dir = os.path.dirname(target_path)
        if parent_dir and parent_dir != "/workspace":
            os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(self.code_content)
            return f"Successfully wrote code to secure workspace: {self.filename}"
        except Exception as e:
            return f"Error writing file to sandbox: {str(e)}"