import sys

from malcolm.cli import _parse_args


def test_parse_no_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["malcolm"])
    assert _parse_args() == {}


def test_parse_target_url(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["malcolm", "--malcolm-target-url=http://localhost:11434/v1"])
    result = _parse_args()
    assert result == {"target_url": "http://localhost:11434/v1"}


def test_parse_multiple_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "malcolm",
        "--malcolm-target-url=http://localhost:11434/v1",
        "--malcolm-port=9000",
        "--malcolm-config-file=custom.yaml",
    ])
    result = _parse_args()
    assert result == {
        "target_url": "http://localhost:11434/v1",
        "port": 9000,
        "config_file": "custom.yaml",
    }


def test_cli_args_override_env(monkeypatch):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://from-env.com")
    monkeypatch.setenv("MALCOLM_PORT", "8900")
    monkeypatch.setattr(sys, "argv", ["malcolm", "--malcolm-port=9999"])

    from malcolm.config import Settings

    overrides = _parse_args()
    settings = Settings(**overrides)

    assert settings.target_url == "http://from-env.com"  # from env
    assert settings.port == 9999  # overridden by CLI
