import os
from src.Tools.BaseTool import BaseTool


def _resolve_case_insensitive_dir(path: str) -> str:
    """Try exact path first, then case-insensitive match for directory components."""
    if os.path.exists(path) and os.path.isdir(path):
        return path

    # Try case-insensitive resolution of the last directory component
    parent_dir = os.path.dirname(path)
    dirname = os.path.basename(path)

    if not os.path.exists(parent_dir):
        return path

    try:
        for entry in os.listdir(parent_dir):
            if entry.lower() == dirname.lower() and os.path.isdir(os.path.join(parent_dir, entry)):
                return os.path.join(parent_dir, entry)
    except (OSError, PermissionError):
        pass

    return path


class ReadDirectoryTool(BaseTool):
    """Lists files and directories at a given path."""
    directory_path: str

    def execute(self) -> str:
        """Reads the directory from the given directory path and returns its contents.
        :return: List of files and folders as a string representation."""
        resolved_path = self.directory_path

        # Auto-resolve absolute paths (like /src/...) by checking if they exist under the /app workspace
        if os.path.isabs(resolved_path) and not resolved_path.startswith("/app"):
            test_path = os.path.join("/app", resolved_path.lstrip("/"))
            if os.path.exists(test_path) and os.path.isdir(test_path):
                resolved_path = test_path

        # Case-insensitive directory resolution
        resolved_path = _resolve_case_insensitive_dir(resolved_path)

        try:
            directory = sorted(os.listdir(resolved_path))
            return str(directory)
        except FileNotFoundError:
            return f"Error: Directory '{self.directory_path}' not found. Please verify the directory path (e.g. check if you should prefix with /app or use a relative path)."
        except Exception as e:
            return f"Error listing directory '{self.directory_path}': {str(e)}"
