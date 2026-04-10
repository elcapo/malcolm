# Transforms

Transforms are pluggable modules that modify requests and responses as they pass through the Malcolm proxy. Each transform is a self-contained package inside `src/malcolm/transforms/`.

## Architecture

```
transforms/
    __init__.py              # REGISTRY + build_pipeline
    _base.py                 # Transform protocol
    ghostkey/
        __init__.py          # GhostKeyTransform + create()
        engine.py            # obfuscation engine
    translation/
        __init__.py          # TranslationTransform + create()
        engine.py            # format conversion functions
```

The pipeline is defined in `malcolm.yaml`:

```yaml
transforms:
  - ghostkey
  - translation:
      direction: anthropic_to_openai
```

- List order = pipeline order
- Plain string (`ghostkey`) = no config needed
- Dict (`translation: {direction: ...}`) = with config

Transforms are applied in **forward order** for requests and **reverse order** for responses/streams.

## Creating a custom transform

### 1. Create the package

Create a new directory in `src/malcolm/transforms/`, e.g. `ratelimit/`:

```
transforms/
    ratelimit/
        __init__.py    # transform class + create()
        engine.py      # business logic (optional)
```

### 2. Implement the protocol

In `__init__.py`, implement the four protocol methods and a `create(config: dict)` factory:

```python
from __future__ import annotations


class RateLimitTransform:
    name = "ratelimit"

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm

    def transform_request(self, body: dict) -> dict:
        # Modify or inspect the request body before it reaches the backend.
        # Return the (possibly modified) body dict.
        return body

    def transform_response(self, body: dict, model: str = "") -> dict:
        # Modify or inspect the response body before it reaches the client.
        # Return the (possibly modified) body dict.
        return body

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        # Process a single SSE line during streaming responses.
        # `state` is a per-stream dict you can use to accumulate data.
        # Return a list of output lines (usually just [line]).
        return [line]

    def rewrite_path(self, path: str) -> str:
        # Optionally rewrite the URL path (e.g. /v1/messages -> /v1/chat/completions).
        # Return the path unchanged if no rewriting is needed.
        return path


def create(config: dict) -> RateLimitTransform:
    rpm = config.get("rpm", 60)
    return RateLimitTransform(rpm)
```

The four protocol methods:

| Method | When it runs | Direction |
|---|---|---|
| `transform_request` | Before forwarding to backend | Forward order (first in list runs first) |
| `transform_response` | Before returning to client | Reverse order (last in list runs first) |
| `transform_stream_line` | For each SSE line in streaming responses | Reverse order |
| `rewrite_path` | Before building the target URL | Forward order |

### 3. Register it

Add one line to the `REGISTRY` dict in `src/malcolm/transforms/__init__.py`:

```python
from malcolm.transforms.ratelimit import create as _create_ratelimit

REGISTRY: dict[str, Callable[[dict], Transform]] = {
    "ghostkey": _create_ghostkey,
    "translation": _create_translation,
    "ratelimit": _create_ratelimit,
}
```

### 4. Use it

Add it to `malcolm.yaml`:

```yaml
transforms:
  - ghostkey
  - ratelimit:
      rpm: 100
```

### 5. Add tests

Create tests covering your transform's behavior and its factory function.

## Publishing a transform as a pip package

Transforms do not need to live inside the Malcolm repo. Any installed pip package can register a transform via the `malcolm.transforms` entry point group, and Malcolm will discover it at startup.

Minimum `pyproject.toml` for an external transform:

```toml
[project]
name = "malcolm-transform-ratelimit"
version = "0.1.0"

[project.entry-points."malcolm.transforms"]
ratelimit = "malcolm_transform_ratelimit:create"
```

The value on the right is an `import.path:attribute` reference to the `create(config: dict)` factory. The name on the left is how the transform is referenced in `malcolm.yaml`.

Once installed (`uv pip install malcolm-transform-ratelimit`), the transform becomes available just like a built-in:

```yaml
transforms:
  - ratelimit:
      rpm: 100
```

Precedence rules:

- Built-in names (`ghostkey`, `translation`) cannot be shadowed by external plugins. A collision is logged and the external registration is ignored.
- If two external plugins claim the same name, the first one discovered wins.
- A plugin whose `create` factory fails to import is logged as a warning and skipped — it will not crash Malcolm's startup.

External transforms should depend only on the Python standard library and their own runtime needs. Do not depend on `malcolm` itself.

A complete working example lives at [`examples/malcolm-transform-example/`](../../../examples/malcolm-transform-example/). It is an independent pip package (own `pyproject.toml`, own venv, own tests) that implements a simple `header_logger` pass-through transform. Install it with `uv pip install -e examples/malcolm-transform-example` to see entry point discovery in action, or copy it as a starting point for your own transform.

## Guidelines

- Transforms must be **self-contained** — no imports from `malcolm.formats`, `malcolm.models`, or other Malcolm modules outside the `transforms` package.
- Heavy logic goes in `engine.py`, the `__init__.py` is a thin adapter.
- The `create(config: dict)` factory receives the YAML config dict for this transform. Raise `ValueError` with a clear message if required config is missing.
