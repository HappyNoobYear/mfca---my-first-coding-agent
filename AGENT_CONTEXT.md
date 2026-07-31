# Mini Claude Code
## Project Context & Agent Instructions

### Project Overview
Master's thesis project for Advanced Media Technologies (SS 2026). Building an autonomous AI agent capable of:
- Multi-turn conversation with persistent memory across sessions
- Tool-calling to interact with files and execute code
- Isolated Docker sandbox for safe code execution (5s timeout, no network)
- Network requests via WebFetchTool (HTTP GET/POST from agent, not sandbox)
- Support for multiple LLM providers (OpenAI, Ollama local inference)
- Automatic history compression to handle long conversations

### Architecture
- **Agent Controller**: Orchestrates the agent loop with chat memory and tool invocation
- **Sandbox Worker**: Isolated Docker container for code execution (runs as non-root user, 5s timeout limit)
- **Memory System**: Token-aware history compression with sliding window + summary strategy
- **Tools**: ReadCodeTool, ReadDirectoryTool, WriteCodeTool, ExecuteCodeTool, WebFetchTool
- **Network**: WebFetchTool runs in agent_controller (has network), sandbox_worker is isolated (no network)
- **Providers**: OpenAI (e.g. gpt-4o, gpt-4o-mini) or Ollama (e.g. gemma4:e2b), selected via configuration

### Key Capabilities
1. **Read Files** — Use `ReadCodeTool` to examine source code with case-insensitive fallback
2. **List Directories** — Use `ReadDirectoryTool` to explore workspace structure
3. **Write Files** — Use `WriteCodeTool` to create or modify code in sandbox
4. **Execute Code** — Use `ExecuteCodeTool` to run Python scripts (5s timeout, isolated)
5. **Fetch Web Data** — Use `WebFetchTool` to call APIs and fetch HTTP/HTTPS content (10s timeout, 100KB limit)
6. **Multi-turn Memory** — Conversation history persists within session
7. **History Compression** — Old conversation turns are automatically summarized to prevent context overflow
8. **Session Persistence** — Conversations can be saved/loaded across sessions

### Building & Running

**Prerequisites:**
- Docker and Docker Compose
- Ollama (for local inference) OR OpenAI API key

**Setup:**
```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your provider/model/API key
```

**Start the Agent:**
```bash
# Build Docker image
docker compose build

# Run agent (interactive mode)
docker compose run --rm agent_controller
```

**Environment Variables** (see `.env.example` for the full list with defaults):
```bash
# Provider selection
LLM_PROVIDER=openai|ollama

# Model name
LLM_MODEL=gpt-4o              # for openai
LLM_MODEL=gemma4:e2b          # for ollama

# OpenAI setup
OPENAI_API_KEY=sk-proj-...

# Ollama setup
OLLAMA_HOST=http://localhost:11434

# Memory/Compression settings
RECENT_TURNS_TO_KEEP=5           # keep last N complete turns in full context
COMPRESSION_ENABLED=true         # enable history compression
MAX_CONVERSATION_TOKENS=4000     # fallback only, used if the model's real context window can't be determined

# WebFetchTool settings
WEBFETCH_ENABLED=true                    # enable HTTP requests
WEBFETCH_MAX_RESPONSE_SIZE=102400        # max response size (100KB)
WEBFETCH_DEFAULT_TIMEOUT=10              # default timeout (seconds)
WEBFETCH_MAX_TIMEOUT=30                  # hard timeout limit
WEBFETCH_BLOCKLIST_INTERNAL_IPS=true     # block SSRF (internal IP ranges)
```

### Workspace & File Paths
- **Agent workspace**: `/app` (absolute root in Docker)
- **Sandbox workspace**: `/app/sandbox_workspace` (shared with sandbox container)
- **Write/execute paths**: `WriteCodeTool` and `ExecuteCodeTool` accept either an absolute `/app/sandbox_workspace/...` path or a bare relative filename (e.g. `hello.py`), both resolve to the same sandbox directory
- **Example**: `/app/sandbox_workspace/hello.py` and `hello.py` both refer to the same file

### What You Should Know
- **Network Isolation**: Sandbox worker has NO network access. Use WebFetchTool for HTTP requests (runs in agent_controller)
- **WebFetchTool SSRF Protection**: Blocks internal IPs (127.0.0.0/8, 192.168.*, 10.*, 172.16.*/12) to prevent attacks
- **WebFetchTool Limits**: 10-30s timeout, 100KB response limit, supports GET/POST only
- **5 Second Timeout**: Code execution times out after 5 seconds (prevent infinite loops)
- **Non-root User**: Sandbox runs as `agentuser` (UID 1000), not root
- **Case Insensitive**: File tools resolve case mismatches (Print.py works for print.py)
- **Tool Descriptions**: LLM receives descriptions of all tools to decide when to use them

### Testing the Agent
Once started, try these commands:

1. **List files:**
   ```
   You: "Show me files in /app/sandbox_workspace"
   ```

2. **Read a file:**
   ```
   You: "Read print.py from /app/sandbox_workspace"
   ```

3. **Write and execute code:**
   ```
   You: "Write a hello.py that prints 'Hello from the sandbox' and execute it"
   ```

4. **Multi-turn conversation:**
   ```
   You: "Create a Python script that counts to 10"
   You: "Now modify it to count backwards from 10"
   You: "Execute the modified version"
   ```

5. **Fetch web data (API call):**
   ```
   You: "What's the current Bitcoin price? Use the CoinGecko API."
   ```

6. **Fetch and parse JSON:**
   ```
   You: "Check the status of this public API: https://api.example.com/status"
   ```

### How History Compression Works
When compression runs:
1. The context manager tracks the conversation's total token count.
2. Once that count crosses a threshold, a fraction (0.5 by default) of the model's real context window read through the provider, not a fixed number:
   - **Keep**: The most recent complete turns untouched (`RECENT_TURNS_TO_KEEP`, 5 by default)
   - **Summarize**: All older turns into a single message, written by an actual model call when one is available, falling back to a deterministic truncation-based summary otherwise
3. The compressed history is used for the next LLM call

### Architecture Notes
- **Tool Schemas**: Tools are Pydantic models with `to_schema()` that generate OpenAI-compatible function schemas
- **Provider Interface**: `LLMInterface` is the shared abstract class both `OpenAIProvider` and `OllamaProvider` implement
- **Token Counting**: OpenAI uses `tiktoken` (accurate), Ollama uses an approximation (~4 chars per token)
- **Message Format**: Standard OpenAI message format (role, content, optional tool_calls)

### Key Files
- `src/Agent/Agent.py` — Main agent loop and chat logic
- `src/API/openai_provider.py` — OpenAI client integration
- `src/API/ollama_provider.py` — Ollama REST API integration
- `src/Memory/MemoryManager.py` — History compression and persistence
- `src/Tools/` — Tool implementations
- `docker-compose.yml` — Container orchestration
- `.env` — Configuration (DO NOT commit, use .env.example)

### Troubleshooting

**Agent says "Tool not found":**
- LLM returned a tool name with wrong case
- Solution: Case-insensitive lookup is now automatic

**File not found errors:**
- Check the filename's case is correct (tools do case-insensitive fallback)

**Context limit exceeded:**
- History compression should trigger automatically
- Check: is `COMPRESSION_ENABLED=true`?

**Connection to Ollama failed:**
- Check: Is Ollama running? `ollama list`
- Check: Is `OLLAMA_HOST` correct? (should be `http://localhost:11434`, or `http://host.docker.internal:11434` from inside a container)

**OpenAI API errors:**
- Check: Is `OPENAI_API_KEY` set correctly?
- Check: Do you have API quota remaining?

### Future Enhancements
- [x] Internet request handling (WebFetchTool with SSRF protection)
- [x] Benchmark harness comparing this agent against other open-source coding-agent frameworks
- [ ] Extended thinking support for complex reasoning
- [ ] Multi-session conversation recall
- [ ] Custom function definitions (user-provided tools)
- [ ] Cost tracking for OpenAI usage
- [ ] HTML parsing tool for web scraping
- [ ] JavaScript execution for dynamic websites
