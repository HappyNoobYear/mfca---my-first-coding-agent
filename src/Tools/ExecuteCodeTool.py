import docker
from src.Tools.BaseTool import BaseTool


class ExecuteCodeTool(BaseTool):
    """Executes a Python file inside the network-isolated sandbox worker container."""
    filename: str

    def execute(self) -> str:
        """
        Sends an execution command to the perpetually running sandbox_worker container.
        The file must have already been written to the shared /workspace directory
        by the WriteCodeTool.
        """
        try:
            # 1. Connect to the internal Docker daemon socket
            # (passed into the agent container via docker-compose)
            client = docker.from_env()

            # 2. Get the running sandbox worker container
            # Docker Compose names containers using the pattern: [folder_name]-[service_name]-1
            # Assuming your root folder is named 'mfca---my-first-coding-agent'
            container_name = "mfca---my-first-coding-agent-sandbox_worker-1"
            sandbox = client.containers.get(container_name)

            # 3. Clean the filename to prevent path traversal attacks
            # Enforces that the agent can only target files inside the shared /workspace
            import os
            safe_filename = os.path.basename(self.filename)

            # 4. Run the code inside the sandbox container
            # We enforce a 5-second timeout and run as the non-root 'agentuser' (UID 1000)
            exec_command = f"timeout 5 python /workspace/{safe_filename}"

            exec_result = sandbox.exec_run(
                cmd=exec_command,
                user="1000",  # Matches the UID defined in docker-compose
                workdir="/workspace"
            )

            # 5. Capture and decode the results
            output = exec_result.output.decode("utf-8")

            # If the timeout command triggered (exit code 124), catch it explicitly
            if exec_result.exit_code == 124:
                return "Error: Code execution timed out (maximum 5 seconds allowed)."

            if not output and exec_result.exit_code != 0:
                return f"Execution failed with exit code {exec_result.exit_code}."

            return output

        except docker.errors.NotFound:
            return (
                f"Error: Sandbox container '{container_name}' not found. "
                "Make sure you started your project using 'docker compose up'."
            )
        except Exception as e:
            return f"An error occurred while communicating with the sandbox: {str(e)}"