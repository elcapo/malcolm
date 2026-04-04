import uvicorn

from malcolm.config import Settings


def main():
    settings = Settings()
    uvicorn.run(
        "malcolm.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
