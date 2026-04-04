# malcolm

LLM API monitoring proxy — transparent man-in-the-middle that logs requests/responses between LLM client tools and model backends.

## Project conventions

- **Tests are mandatory**: Every module must have corresponding tests in `tests/`. Run `uv run pytest` before delivering any work.
- **Documentation in English**: All documentation (README.md, docs/) is written in English.
- **Update docs before delivering**: When completing a feature, update README.md and relevant docs/ files to reflect the changes.
- **Use `uv`** for dependency management and running commands.
- **Source layout**: Code lives in `src/malcolm/`. Tests in `tests/`.

## Tech stack

- Python 3.12+
- FastAPI + uvicorn
- httpx (async HTTP client)
- aiosqlite (async SQLite)
- pydantic-settings (configuration)
- pytest + pytest-asyncio (testing)

## Running

```bash
uv run malcolm          # start the proxy
uv run pytest           # run tests
```
