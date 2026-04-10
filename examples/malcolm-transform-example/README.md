# malcolm-transform-example

Example external transform for [Malcolm](https://github.com/malcolm-proxy/malcolm) demonstrating the entry point discovery mechanism.

This package is **independent from `malcolm` itself** — it has its own `pyproject.toml` and declares a `malcolm.transforms` entry point that Malcolm discovers at startup whenever this package is installed in the same Python environment. Nothing in the core proxy imports from this package.

## What the transform does

`header_logger` is a pass-through transform: it logs structural information about each request and response (body key set, body size, model, path) without modifying anything. It exists as a template and as a live test of the entry point discovery mechanism — not to do anything useful in production.

## Install (development mode)

From the root of the malcolm repo:

```bash
uv pip install -e examples/malcolm-transform-example
```

Or from anywhere into a venv where `malcolm` is already installed:

```bash
uv pip install -e /path/to/malcolm/examples/malcolm-transform-example
```

After installation, confirm the entry point is visible:

```bash
python -c "from importlib.metadata import entry_points; print([ep.name for ep in entry_points(group='malcolm.transforms')])"
```

You should see `['header_logger']` (plus any other external transforms you have installed).

## Use

Add it to your `malcolm.yaml`:

```yaml
transforms:
  - header_logger
```

Or with custom configuration:

```yaml
transforms:
  - header_logger:
      prefix: "[my-proxy]"
      logger_name: "my.custom.logger"
```

When you start malcolm, the startup log will show the transform in the active pipeline:

```
malcolm.transforms INFO transform pipeline: ['header_logger']
```

And each request will produce log lines like:

```
malcolm_transform_example INFO [header_logger] request keys=['messages', 'model'] size=234
malcolm_transform_example INFO [header_logger] response model=gpt-4o size=1012
```

## Configuration

| Key | Default | Description |
|---|---|---|
| `prefix` | `[header_logger]` | String prepended to every log line produced by this transform. |
| `logger_name` | `malcolm_transform_example` | Python logger name the transform writes to. Useful if you want to route its output somewhere specific. |

## Writing your own transform

Use this package as a template:

1. Copy the directory to a new location (inside this repo or anywhere else).
2. Rename the package: update `name` in `pyproject.toml`, rename `src/malcolm_transform_example/` to your package name, and update the test imports.
3. Update the entry point in `pyproject.toml` to point at your new package's `create` factory:
   ```toml
   [project.entry-points."malcolm.transforms"]
   your_transform_name = "your_package:create"
   ```
4. Implement the four protocol methods on your transform class: `transform_request`, `transform_response`, `transform_stream_line`, `rewrite_path`.
5. Implement `create(config: dict)` as the factory that Malcolm will call at pipeline assembly time. Raise `ValueError` with a clear message if required configuration is missing.
6. Install it into the same venv as `malcolm` — the new transform becomes available in `malcolm.yaml` automatically.

See the main [transforms README](../../src/malcolm/transforms/README.md) for the full protocol contract and precedence rules.

## Running the tests

From inside this directory:

```bash
cd examples/malcolm-transform-example
uv sync
uv run pytest
```

The tests don't require `malcolm` to be installed — they verify the transform's behaviour in isolation.
