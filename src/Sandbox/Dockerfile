# Use a slim, official base image
FROM python:3.11-slim

# Install any basic tools your agent might need
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the agent
RUN useradd -m -u 1000 agentuser

# Set up a working directory
WORKDIR /workspace
RUN chown agentuser:agentuser /workspace

# Switch to the non-root user
USER agentuser

# (Optional) Pre-install common libraries the agent might use
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container alive waiting for commands
CMD ["tail", "-f", "/dev/null"]