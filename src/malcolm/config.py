from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MALCOLM_", env_file=".env", extra="ignore"
    )

    target_url: str
    target_api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 8900
    storage_enabled: bool = True
    db_path: str = "malcolm.db"
    log_level: str = "info"
    config_file: str = "malcolm.yaml"
