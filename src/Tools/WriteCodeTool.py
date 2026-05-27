import os
from src.Tools.BaseTool import BaseTool


class WriteCodeTool(BaseTool):
    """Writes Python code into the secure shared workspace."""
    filename: str
    code_content: str

    def execute(self) -> str:
        # Force all files to be written inside the shared workspace
        secure_path = os.path.join("/workspace", os.path.basename(self.filename))
        with open(secure_path, 'w', encoding='utf-8') as f:
            f.write(self.code_content)
        return f"Successfully wrote code to secure workspace: {self.filename}"