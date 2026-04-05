import argparse

import uvicorn

from malcolm.app import create_app
from malcolm.config import Settings


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(
        prog="malcolm",
        description="LLM API monitoring proxy",
    )
    parser.add_argument("--malcolm-target-url", dest="target_url")
    parser.add_argument("--malcolm-target-api-key", dest="target_api_key")
    parser.add_argument("--malcolm-host", dest="host")
    parser.add_argument("--malcolm-port", dest="port", type=int)
    parser.add_argument("--malcolm-storage-enabled", dest="storage_enabled")
    parser.add_argument("--malcolm-db-path", dest="db_path")
    parser.add_argument("--malcolm-log-level", dest="log_level")
    parser.add_argument("--malcolm-translate", dest="translate")

    args = parser.parse_args()
    # Return only the explicitly provided arguments
    return {k: v for k, v in vars(args).items() if v is not None}


def main():
    overrides = _parse_args()
    settings = Settings(**overrides)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
