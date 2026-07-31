"""Real tools for smolagents, mirroring Mini Claude Code's own five tools.

Rather than reimplementing read/write/execute logic for smolagents, these
wrap Mini Claude Code's own (now Windows-safe, posixpath-fixed) tool classes directly --
same sandbox_workspace path, same sandbox_worker container, same security
checks. This is a stronger fairness guarantee than parallel reimplementation
would be: both agents are judged against the literal same sandboxing
behavior, just exposed through smolagents' @tool convention instead of
Mini Claude Code's BaseTool/Pydantic convention.
"""

from smolagents import tool

from src.Tools.ReadCodeTool import ReadCodeTool
from src.Tools.ReadDirectoryTool import ReadDirectoryTool
from src.Tools.WriteCodeTool import WriteCodeTool
from src.Tools.ExecuteCodeTool import ExecuteCodeTool


@tool
def read_code(file_path: str) -> str:
    """Reads the content of a file.

    Args:
        file_path: Path to the file to read.
    """
    return ReadCodeTool(file_path=file_path).execute()


@tool
def read_directory(directory_path: str) -> str:
    """Lists the files and directories at a given path.

    Args:
        directory_path: Path to the directory to list.
    """
    return ReadDirectoryTool(directory_path=directory_path).execute()


@tool
def write_code(filename: str, code_content: str) -> str:
    """Writes code content to a file in the secure sandbox workspace.

    Args:
        filename: Name of the file to write, relative to the sandbox workspace.
        code_content: The full contents to write to the file.
    """
    return WriteCodeTool(filename=filename, code_content=code_content).execute()


@tool
def execute_code(filename: str) -> str:
    """Executes a Python file inside the isolated, network-restricted sandbox container.

    Args:
        filename: Name of the file to execute. Must already have been written
            to the sandbox workspace via write_code.
    """
    return ExecuteCodeTool(filename=filename).execute()


def build_smolagents_tools():
    """Tool instances for smolagents' ToolCallingAgent(tools=...)."""
    return [read_code, read_directory, write_code, execute_code]
