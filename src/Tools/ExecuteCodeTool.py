import docker
import os
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
            container_name = "mfca---my-first-coding-agent-sandbox_worker-1"
            try:
                sandbox = client.containers.get(container_name)
            except docker.errors.NotFound:
                sandbox = None
                for c in client.containers.list():
                    if "sandbox_worker" in c.name:
                        sandbox = c
                        container_name = c.name
                        break
                if not sandbox:
                    return (
                        "Error: Sandbox container 'sandbox_worker' not found. "
                        "Make sure you started your project using 'docker compose up'."
                    )

            # 3. Clean and validate the target path inside the sandbox container
            clean_filename = self.filename
            if clean_filename.startswith("/app/"):
                clean_filename = clean_filename[5:]  # Removes the "/app/" prefix
            elif clean_filename.startswith("/"):
                clean_filename = clean_filename[1:]  # Removes any leading slash

            # STRIP THE PREFIX: Because sandbox_worker maps ./sandbox_workspace straight to /workspace
            if clean_filename.startswith("sandbox_workspace/"):
                clean_filename = clean_filename[len("sandbox_workspace/"):]

            # This will now correctly evaluate to exactly "/workspace/Print.py"
            target_path = os.path.normpath(os.path.join("/workspace", clean_filename))

            # Prevent execution attempts outside the isolated workspace
            if not target_path.startswith("/workspace"):
                return f"Security Error: Access denied. The path must stay inside the isolated workspace. Attempted: {self.filename}"
            # 4. Run the code inside the sandbox container
            # We enforce a 5-second timeout and run as the non-root 'agentuser' (UID 1000)
            exec_command = f"timeout 5 python {target_path}"

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

        except Exception as e:
            return f"An error occurred while communicating with the sandbox: {str(e)}"