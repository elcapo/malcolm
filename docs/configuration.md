# Configuration

malcolm is configured through environment variables (all prefixed with `MALCOLM_`), CLI arguments (all prefixed with `--malcolm-`), and a YAML config file for the transform pipeline. CLI arguments take precedence over environment variables.

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

When `true`, malcolm stores all requests and responses in a SQLite database. When `false`, malcolm still proxies everything and logs to stdout, but doesn't persist to disk. The TUI will show no records when storage is disabled.

### `MALCOLM_DB_PATH`

**Default:** `malcolm.db`

Path to the SQLite database file. Can be absolute or relative to the working directory.

### `MALCOLM_LOG_LEVEL`

**Default:** `info`

Log level for console output. Valid values: `debug`, `info`, `warning`, `error`, `critical`.

### `MALCOLM_CONFIG_FILE`

**Default:** `malcolm.yaml`

Path to the YAML config file that defines the transform pipeline.

## Transform pipeline

Transforms are configured in `malcolm.yaml` (or whatever path `MALCOLM_CONFIG_FILE` points to). Each transform is a pluggable module that can modify requests and responses as they pass through the proxy.

The list order defines the pipeline order: transforms are applied in order on the request side and in reverse order on the response side.

```yaml
transforms:
  - ghostkey
  - translation:
      direction: anthropic_to_openai
```

Transforms without configuration are listed as plain strings. Transforms with configuration are listed as single-key dicts mapping the transform name to its config.

### `ghostkey`

Scans outgoing requests for known secret patterns (API keys, tokens, JWTs, etc.) and replaces them with format-preserving fakes before they reach the backend. Responses are transparently restored so the client always sees real values.

This prevents secrets accidentally sent in LLM messages (e.g. `.env` file contents read by a coding agent) from reaching the upstream API.

Both the original (with real secrets) and obfuscated versions are stored in the database — the original in the `requests` table and the obfuscated version in the `request_transforms` table. Use `t` in the TUI to toggle between views.

The secret dictionary lives in memory only — it is not persisted to disk and resets when malcolm restarts.

No configuration required.

### `translation`

Enables protocol translation between Anthropic and OpenAI API formats. Translates requests and responses on the fly, allowing clients that speak one protocol to use backends that speak the other.

**Config:**

| Key | Required | Description |
|---|---|---|
| `direction` | yes | `anthropic_to_openai` or `openai_to_anthropic` |

- `anthropic_to_openai` — Client sends Anthropic format (`/v1/messages`), backend expects OpenAI format (`/v1/chat/completions`). Useful for running Claude Code against OpenAI or Ollama backends.
- `openai_to_anthropic` — Client sends OpenAI format (`/v1/chat/completions`), backend expects Anthropic format (`/v1/messages`). Useful for running OpenAI-compatible tools against Anthropic's API.

In addition to the built-in transforms listed above, malcolm discovers external transforms installed as pip packages via the `malcolm.transforms` entry point group. Any installed plugin becomes available in `malcolm.yaml` just like a built-in. See [malcolm-proxy/malcolm-transform-example](https://github.com/malcolm-proxy/malcolm-transform-example) for a working reference implementation, and [src/malcolm/transforms/README.md](../src/malcolm/transforms/README.md) for the full protocol contract.

## Example `.env` file

```bash
MALCOLM_TARGET_URL=https://openrouter.ai/api/v1
MALCOLM_TARGET_API_KEY=sk-or-v1-abc123
MALCOLM_PORT=8900
MALCOLM_STORAGE_ENABLED=true
MALCOLM_DB_PATH=malcolm.db
MALCOLM_LOG_LEVEL=info
```

## Example `malcolm.yaml`

```yaml
transforms:
  - ghostkey
  - translation:
      direction: anthropic_to_openai
```
