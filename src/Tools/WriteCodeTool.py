import os
import posixpath
from src.Tools.BaseTool import BaseTool


class WriteCodeTool(BaseTool):
    """Writes code files to the secure sandbox workspace."""
    filename: str
    code_content: str

    def execute(self) -> str:
        """Writes the code content to a file in the sandbox workspace.
        :return: Success or error message."""
        # Normalise the incoming filename: strip any /app/ or sandbox_workspace/
        # prefix so the model can pass either form and we always land in the right place.
        clean = self.filename
        if clean.startswith("/app/"):
            clean = clean[5:]
        if clean.startswith("/"):
            clean = clean[1:]
        if clean.startswith("sandbox_workspace/"):
            clean = clean[len("sandbox_workspace/"):]

        # Use posixpath, not os.path: this target is always a Linux-style
        # absolute path (the production deployment always runs inside a Linux
        # container). os.path.normpath would convert "/" to "\" when this
        # process itself runs on Windows (e.g. under the benchmark harness),
        # silently breaking the startswith() check below on every call.
        target_path = posixpath.normpath(posixpath.join("/app/sandbox_workspace", clean))

        # Prevent path traversal outside the sandbox
        if not target_path.startswith("/app/sandbox_workspace"):
            return f"Security Error: Access denied. Files must be written inside /app/sandbox_workspace/. Attempted: {self.filename}"

        # Create parent directories if a nested file structure is requested
        parent_dir = posixpath.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(self.code_content)
            return f"Successfully wrote code to secure workspace: {self.filename}"
        except Exception as e:
            return f"Error writing file to sandbox: {str(e)}"