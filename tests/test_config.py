import pytest

from malcolm.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MALCOLM_TARGET_API_KEY", "sk-test-123")
    monkeypatch.setenv("MALCOLM_PORT", "9000")
    monkeypatch.setenv("MALCOLM_STORAGE_ENABLED", "false")

    settings = Settings()

    assert settings.target_url == "https://api.example.com/v1"
    assert settings.target_api_key == "sk-test-123"
    assert settings.port == 9000
    assert settings.storage_enabled is False


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "https://api.example.com/v1")

    settings = Settings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8900
    assert settings.storage_enabled is True
    assert settings.db_path == "malcolm.db"
    assert settings.log_level == "info"
    assert settings.target_api_key == ""
    assert settings.config_file == "malcolm.yaml"


def test_settings_config_file(monkeypatch):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MALCOLM_CONFIG_FILE", "custom.yaml")

    settings = Settings()

    assert settings.config_file == "custom.yaml"


def test_settings_requires_target_url(monkeypatch):
    monkeypatch.delenv("MALCOLM_TARGET_URL", raising=False)

    with pytest.raises(Exception):
        Settings(_env_file=None)
