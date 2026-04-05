# Configuration

malcolm is configured through environment variables (all prefixed with `MALCOLM_`) and CLI arguments (all prefixed with `--malcolm-`). CLI arguments take precedence over environment variables.

```bash
# These two are equivalent:
MALCOLM_TARGET_URL=http://localhost:11434/v1 uv run malcolm
uv run malcolm --malcolm-target-url=http://localhost:11434/v1

# CLI arguments override environment variables:
MALCOLM_PORT=8900 uv run malcolm --malcolm-port=9000  # listens on 9000
```

## Required

### `MALCOLM_TARGET_URL`

The base URL of the LLM API backend to proxy requests to. This should include the path prefix if the API uses one (e.g., `/v1`).

Examples:
```bash
# OpenRouter
MALCOLM_TARGET_URL=https://openrouter.ai/api/v1

# OpenAI
MALCOLM_TARGET_URL=https://api.openai.com/v1

# Local Ollama
MALCOLM_TARGET_URL=http://localhost:11434/v1

# LM Studio
MALCOLM_TARGET_URL=http://localhost:1234/v1
```

## Optional

### `MALCOLM_TARGET_API_KEY`

**Default:** *(empty)*

API key to use when authenticating with the backend. Sent as `Authorization: Bearer <key>`.

If left empty, malcolm forwards the client's `Authorization` header as-is. This makes malcolm a pure transparent proxy where the client provides its own credentials.

### `MALCOLM_HOST`

**Default:** `127.0.0.1`

The address malcolm listens on. Use `0.0.0.0` to accept connections from other machines.

### `MALCOLM_PORT`

**Default:** `8900`

The port malcolm listens on.

### `MALCOLM_STORAGE_ENABLED`

**Default:** `true`

When `true`, malcolm stores all requests and responses in a SQLite database. When `false`, malcolm still proxies everything and logs to stdout, but doesn't persist to disk. The log viewer will show no records when storage is disabled.

### `MALCOLM_DB_PATH`

**Default:** `malcolm.db`

Path to the SQLite database file. Can be absolute or relative to the working directory.

### `MALCOLM_LOG_LEVEL`

**Default:** `info`

Log level for console output. Valid values: `debug`, `info`, `warning`, `error`, `critical`.

### `MALCOLM_TRANSLATE`

**Default:** *(empty)*

Enables protocol translation between Anthropic and OpenAI API formats. When set, malcolm translates requests and responses on the fly, allowing clients that speak one protocol to use backends that speak the other.

Valid values:
- `anthropic_to_openai` — Client sends Anthropic format (`/v1/messages`), backend expects OpenAI format (`/v1/chat/completions`). Useful for running Claude Code against OpenAI or Ollama backends.
- `openai_to_anthropic` — Client sends OpenAI format (`/v1/chat/completions`), backend expects Anthropic format (`/v1/messages`). Useful for running OpenAI-compatible tools against Anthropic's API.

When empty or unset, malcolm acts as a transparent proxy with no format conversion.

## Example `.env` file

```bash
MALCOLM_TARGET_URL=https://openrouter.ai/api/v1
MALCOLM_TARGET_API_KEY=sk-or-v1-abc123
MALCOLM_PORT=8900
MALCOLM_STORAGE_ENABLED=true
MALCOLM_DB_PATH=malcolm.db
MALCOLM_LOG_LEVEL=info
MALCOLM_TRANSLATE=
```
