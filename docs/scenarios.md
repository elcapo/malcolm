# Usage Scenarios

This guide shows how to configure Malcolm with common combinations of LLM clients and model backends.

In every scenario, Malcolm acts as a transparent proxy: the client talks to Malcolm, and Malcolm forwards requests to the real backend. The general setup is always the same:

1. Start Malcolm pointing at the backend.
2. Configure the client to point at Malcolm instead of the backend.

## Ollama

Start Malcolm with Ollama as the target:

```bash
uv run malcolm --malcolm-target-url=http://localhost:11434/v1
```

No API key is needed — Ollama runs locally without authentication.

### Using Ollama's models in Claude Code

```shell
ANTHROPIC_AUTH_TOKEN=ollama \
ANTHROPIC_BASE_URL=http://127.0.0.1:8900 \
ANTHROPIC_API_KEY="" \
claude --model qwen3-coder:30b
```

Claude Code expects Anthropic-style auth variables. Setting `ANTHROPIC_AUTH_TOKEN` to a dummy value and clearing `ANTHROPIC_API_KEY` prevents it from trying to authenticate against Anthropic's servers. The model name must match one available in your Ollama instance.

### Using Ollama's models in OpenCode

In your OpenCode configuration file (`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "malcolm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (malcolm)",
      "options": {
        "baseURL": "http://127.0.0.1:8900"
      },
      "models": {
        "qwen3-coder:30b": {
          "name": "Qwen 3 (Ollama via Malcolm)"
        }
      }
    }
  }
}
```

OpenCode uses the `@ai-sdk/openai-compatible` provider to talk to any OpenAI-compatible endpoint. Point `baseURL` at Malcolm and declare the models you want to use.

## OpenAI

Start Malcolm with OpenAI as the target:

```bash
uv run malcolm \
  --malcolm-target-url=https://api.openai.com/v1 \
  --malcolm-target-api-key=sk-...
```

### Using OpenAI's models in Claude Code

Claude Code speaks the Anthropic protocol (`/v1/messages`), not the OpenAI protocol (`/v1/chat/completions`). To bridge this gap, enable Malcolm's protocol translation:

```bash
uv run malcolm \
  --malcolm-target-url=https://api.openai.com/v1 \
  --malcolm-target-api-key=sk-... \
  --malcolm-translate=anthropic_to_openai
```

Then point Claude Code at Malcolm as if it were an Anthropic backend:

```shell
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_BASE_URL=http://127.0.0.1:8900 \
ANTHROPIC_API_KEY="" \
claude --model gpt-4.1
```

Malcolm receives Anthropic-format requests, translates them to OpenAI format, forwards them, and translates the OpenAI responses back to Anthropic format before returning them to Claude Code.

### Using OpenAI's models in OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "malcolm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenAI (malcolm)",
      "options": {
        "baseURL": "http://127.0.0.1:8900/v1"
      },
      "models": {
        "gpt-4.1": {
          "name": "GPT-4.1 (via Malcolm)"
        }
      }
    }
  }
}
```

If Malcolm is not managing the API key (i.e., `MALCOLM_TARGET_API_KEY` is unset), add `"apiKey"` inside `"options"` with your OpenAI key.

## Anthropic

Start Malcolm with Anthropic as the target:

```bash
uv run malcolm \
  --malcolm-target-url=https://api.anthropic.com/v1 \
  --malcolm-target-api-key=sk-ant-...
```

### Using Anthropic's models in Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8900 \
ANTHROPIC_API_KEY=dummy \
claude
```

Claude Code natively speaks the Anthropic API, so this is the most straightforward scenario. Malcolm forwards requests directly to Anthropic's servers.

As with OpenAI, you can alternatively leave `MALCOLM_TARGET_API_KEY` unset and provide the real key via the client:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8900 \
ANTHROPIC_API_KEY=sk-ant-... \
claude
```

### Using Anthropic's models in OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "malcolm": {
      "npm": "@ai-sdk/anthropic",
      "name": "Anthropic (malcolm)",
      "options": {
        "baseURL": "http://127.0.0.1:8900"
      },
      "models": {
        "claude-sonnet-4-20250514": {
          "name": "Claude Sonnet (via Malcolm)"
        }
      }
    }
  }
}
```

Here OpenCode uses the `@ai-sdk/anthropic` provider instead of the generic OpenAI-compatible one, since Anthropic has its own API format. If Malcolm is not managing the key, add `"apiKey"` in `"options"`.

## Viewing logs

Regardless of the scenario, browse logged requests at [http://127.0.0.1:8900/logs](http://127.0.0.1:8900/logs) or query the JSON API at `/api/logs`.

## Tips

- **API key placement**: If you set `--malcolm-target-api-key`, Malcolm injects the key into every forwarded request and clients can use dummy credentials. If you leave it unset, clients must provide their own valid key — Malcolm will forward it as-is.
- **Model names**: Use the exact model identifier the backend expects. For Ollama, this is the tag you pulled (e.g., `qwen3-coder:30b`). For OpenAI and Anthropic, these are the official model IDs.
- **Port conflicts**: If port 8900 is taken, use `--malcolm-port` to choose a different one and update the client URLs accordingly.
- **CLI vs environment**: All `--malcolm-*` arguments have a corresponding `MALCOLM_*` environment variable. CLI arguments take precedence. Use environment variables (or a `.env` file) for persistent configuration and CLI arguments for one-off overrides.
