FROM python:3.11-slim

WORKDIR /app

# Install git or any other tools the controller needs
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy your project files into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run your agent loop
CMD ["python", "-m", "src.Agent.Agent"]