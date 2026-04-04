# malcolm

A transparent monitoring proxy for LLM API calls. malcolm sits between your LLM client tools (like [Claude Code](https://claude.ai/code) or [opencode](https://github.com/opencode-ai/opencode)) and the actual model backend, logging every request and response for inspection.

## Why?

Tools like `claude` and `opencode` construct complex prompts with system instructions, tool definitions, and conversation history. malcolm lets you see exactly what gets sent to the model — useful for debugging, understanding tool behavior, and optimizing prompts.

## Quick start

```bash
# Install
uv pip install -e .

# Configure the backend to proxy to
export MALCOLM_TARGET_URL="https://openrouter.ai/api/v1"
export MALCOLM_TARGET_API_KEY="sk-or-..."

# Start the proxy
malcolm
```

Then point your LLM tool to malcolm:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8900/v1
export OPENAI_API_KEY=dummy  # malcolm handles real auth
claude --model openai/anthropic/claude-sonnet-4-20250514
```

Browse logged requests at [http://127.0.0.1:8900/logs](http://127.0.0.1:8900/logs).

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MALCOLM_TARGET_URL` | *(required)* | Backend API base URL |
| `MALCOLM_TARGET_API_KEY` | *(empty)* | API key for the backend. If empty, forwards the client's Authorization header |
| `MALCOLM_HOST` | `127.0.0.1` | Listen address |
| `MALCOLM_PORT` | `8900` | Listen port |
| `MALCOLM_STORAGE_ENABLED` | `true` | Enable/disable SQLite persistence |
| `MALCOLM_DB_PATH` | `malcolm.db` | SQLite database file path |
| `MALCOLM_LOG_LEVEL` | `info` | Log level |

See [docs/configuration.md](docs/configuration.md) for details.

## How it works

```mermaid
flowchart LR
    Client["Your LLM tool<br><span style="color: lightgray;">claude / opencode / curl</span>"]
    Malcolm["malcolm<br><span style="color: lightgray;">localhost:8900</span>"]
    Backend["Real LLM API"]
    DB[("SQLite<br><span style="color: lightgray;">malcolm.db</span>")]

    Client -- request --> Malcolm
    Malcolm -- forward --> Backend
    Backend -- response --> Malcolm
    Malcolm -- response --> Client
    Malcolm -. logs .-> DB
```

malcolm exposes an OpenAI-compatible API. It captures the full request, forwards it to the configured backend, captures the full response (including streaming), and stores everything in a local SQLite database for later inspection.

See [docs/architecture.md](docs/architecture.md) for the full architecture overview.

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run the proxy
uv run malcolm
```

## License

MIT
