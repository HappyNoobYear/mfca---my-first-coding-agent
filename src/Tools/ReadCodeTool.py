import os
from src.Tools.BaseTool import BaseTool


def _resolve_case_insensitive(path: str) -> str:
    """Try exact path first, then case-insensitive match in parent directory."""
    if os.path.exists(path):
        return path

    parent_dir = os.path.dirname(path)
    filename = os.path.basename(path)

    if not os.path.exists(parent_dir):
        return path

    try:
        for entry in os.listdir(parent_dir):
            if entry.lower() == filename.lower():
                return os.path.join(parent_dir, entry)
    except (OSError, PermissionError):
        pass

    return path


class ReadCodeTool(BaseTool):
    """Reads code from a file in the workspace."""
    file_path: str

    def execute(self) -> str:
        """Reads the code from the given file path and returns it as a string.
        :return: The content of the code file as a string."""
        resolved_path = self.file_path

        # Auto-resolve absolute paths (like /src/...) by checking if they exist under the /app workspace
        if os.path.isabs(resolved_path) and not resolved_path.startswith("/app"):
            test_path = os.path.join("/app", resolved_path.lstrip("/"))
            if os.path.exists(test_path):
                resolved_path = test_path

        # Auto-resolve bare relative filenames (e.g. "hello.py") by checking the
        # shared sandbox workspace, since that's where WriteCodeTool/ExecuteCodeTool
        # always write/run files -- a plain filename mentioned without a path is
        # far more likely to mean "the one in the sandbox" than one relative to
        # this process's own working directory.
        #
        # Deliberately NOT gated on "not os.path.exists(resolved_path)" -- that
        # looked safe but wasn't: this process's CWD is normally the project
        # root, which happens to contain a same-named file at the top level
        # (AGENT_CONTEXT.md). On a case-insensitive filesystem (Windows, macOS
        # default), os.path.exists("agent_context.md") matches that real file
        # via case-folding alone, so the guard silently "succeeded" and this
        # whole fallback never ran -- meaning a bare filename intended for the
        # sandbox could silently resolve to an unrelated same-named file at
        # the project root instead. Checking the sandbox path unconditionally,
        # before ever looking at the bare path, removes that collision.
        if not os.path.isabs(resolved_path):
            test_path = os.path.join("/app/sandbox_workspace", resolved_path)
            if os.path.exists(test_path):
                resolved_path = test_path

        # Case-insensitive path resolution
        resolved_path = _resolve_case_insensitive(resolved_path)

        try:
            with open(resolved_path, 'r', encoding='utf-8') as file:
                code = file.read()
            return code
        except FileNotFoundError:
            return f"Error: File '{self.file_path}' not found. Please verify the file path (e.g. check if you should prefix with /app or use a relative path)."
        except Exception as e:
            return f"Error reading file '{self.file_path}': {str(e)}"