"""Tests for malcolm.transforms.ghostkey.engine — secret obfuscation."""

import json

import pytest

from malcolm.transforms.ghostkey import engine as ghostkey

from malcolm.transforms.ghostkey.engine import (
    is_sensitive_file,
    obfuscate,
    reset_session,
    restore,
    scan_env_content,
    scan_request,
    scan_tokens,
    session_stats,
)


@pytest.fixture(autouse=True)
def _clean_session():
    """Reset the global secret dictionary between tests."""
    reset_session()
    yield
    reset_session()


# ── Token pattern detection ──────────────────────────────────────────────────

class TestScanTokens:
    def test_anthropic_key(self):
        token = "sk-ant-api03-" + "a" * 20
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_openai_project_key(self):
        token = "sk-proj-" + "B" * 40
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_github_pat(self):
        token = "github_pat_" + "x" * 82
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_github_token_ghp(self):
        token = "ghp_" + "a" * 36
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_aws_access_key(self):
        token = "AKIA" + "A" * 16
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_google_api_key(self):
        token = "AIza" + "b" * 35
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_stripe_live_key(self):
        token = "sk_live_" + "c" * 24
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_slack_token(self):
        token = "xoxb-" + "d" * 20
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_huggingface_token(self):
        token = "hf_" + "e" * 34
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_jwt(self):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456"
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_bare_hex_not_matched(self):
        # Bare hex strings (git SHAs, MD5/SHA256 hashes, UUIDs without dashes)
        # are too ambiguous to treat as secrets — only service-prefixed tokens
        # are registered.
        scan_tokens("a" * 32)
        scan_tokens("e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0")  # git SHA-ish
        assert session_stats()["secrets_protected"] == 0

    def test_short_string_ignored(self):
        scan_tokens("short")
        assert session_stats()["secrets_protected"] == 0

    def test_same_token_registered_once(self):
        token = "AKIA" + "X" * 16
        scan_tokens(token)
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1

    def test_sendgrid_key(self):
        token = "SG." + "a" * 22 + "." + "b" * 43
        scan_tokens(token)
        assert session_stats()["secrets_protected"] >= 1

    def test_npm_token(self):
        token = "npm_" + "c" * 36
        scan_tokens(token)
        assert session_stats()["secrets_protected"] == 1


# ── Format-preserving fake generation ────────────────────────────────────────

class TestFakeGeneration:
    def test_prefix_preserved(self):
        token = "sk-ant-" + "a" * 20
        scan_tokens(token)
        fake = obfuscate(token)
        # _natural_prefix preserves up to the first separator ("sk-")
        assert fake.startswith("sk-")
        assert fake != token

    def test_length_preserved(self):
        token = "AKIA" + "B" * 16
        scan_tokens(token)
        fake = obfuscate(token)
        assert len(fake) == len(token)

    def test_charset_preserved(self):
        token = "ghp_" + "aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV1wX2y"
        scan_tokens(token)
        fake = obfuscate(token)
        # After the prefix, each char should match the original charset
        for orig, repl in zip(token[4:], fake[4:]):
            if orig.isdigit():
                assert repl.isdigit()
            elif orig.isupper():
                assert repl.isupper()
            elif orig.islower():
                assert repl.islower()
            else:
                assert repl == orig  # separators preserved


# ── Obfuscate / restore round-trip ───────────────────────────────────────────

class TestObfuscateRestore:
    def test_round_trip(self):
        token = "sk-ant-" + "z" * 20
        scan_tokens(token)
        text = f"my key is {token} here"
        obfuscated = obfuscate(text)
        assert token not in obfuscated
        restored = restore(obfuscated)
        assert restored == text

    def test_multiple_tokens(self):
        t1 = "AKIA" + "A" * 16
        t2 = "ghp_" + "b" * 36
        text = f"aws={t1} github={t2}"
        scan_tokens(text)
        obfuscated = obfuscate(text)
        assert t1 not in obfuscated
        assert t2 not in obfuscated
        assert restore(obfuscated) == text

    def test_no_secrets_passthrough(self):
        text = "just a normal message"
        assert obfuscate(text) == text
        assert restore(text) == text

    def test_obfuscate_idempotent(self):
        token = "sk-ant-" + "m" * 20
        scan_tokens(token)
        text = f"key={token}"
        once = obfuscate(text)
        twice = obfuscate(once)
        assert once == twice


# ── Sensitive file detection ─────────────────────────────────────────────────

class TestSensitiveFile:
    @pytest.mark.parametrize("name", [
        ".env", ".env.local", ".env.production",
        "secrets.json", "credentials.yaml",
        "service-account.json", ".npmrc", ".pypirc",
        "terraform.tfvars", "id_rsa",
    ])
    def test_known_sensitive_files(self, name):
        assert is_sensitive_file(name) is True

    @pytest.mark.parametrize("name", [
        "server.key", "cert.pem", "store.p12", "key.pfx",
    ])
    def test_sensitive_extensions(self, name):
        assert is_sensitive_file(name) is True

    def test_path_with_sensitive_file(self):
        assert is_sensitive_file("/home/user/project/.env") is True

    def test_non_sensitive(self):
        assert is_sensitive_file("app.py") is False
        assert is_sensitive_file("README.md") is False

    def test_env_variant(self):
        assert is_sensitive_file(".env.custom") is True


# ── ENV content scanning ─────────────────────────────────────────────────────

class TestEnvScanning:
    def test_parses_env_values(self):
        content = 'API_KEY="my-secret-key-value-1234"\nDB_HOST=localhost'
        scan_env_content(content)
        assert session_stats()["secrets_protected"] == 2  # both values >= 8 chars

    def test_skips_comments(self):
        content = "# API_KEY=secret123456\nPORT=3000"
        scan_env_content(content)
        assert session_stats()["secrets_protected"] == 0

    def test_strips_quotes(self):
        content = "SECRET='supersecretvalue'"
        scan_env_content(content)
        fake = obfuscate("supersecretvalue")
        assert fake != "supersecretvalue"


# ── scan_request ─────────────────────────────────────────────────────────────

class TestScanRequest:
    def test_scans_tokens_in_body(self):
        token = "sk-ant-" + "q" * 20
        scan_request(json.dumps({"messages": [], "key": token}))
        assert session_stats()["secrets_protected"] == 1

    def test_scans_env_anthropic_format(self):
        """Anthropic: tool_use with sensitive path + tool_result triggers env scan."""
        body = json.dumps({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "read_1", "name": "Read",
                     "input": {"path": "/home/user/.env"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "read_1",
                     "content": "API_KEY=supersecretvalue12345\nDB_URL=postgres://secret"},
                ]},
            ],
            "system": "You are a helpful assistant.",
        })
        scan_request(body)
        assert session_stats()["secrets_protected"] >= 1

    def test_scans_env_openai_format(self):
        """OpenAI tool result with sensitive filename triggers env scanning."""
        body = json.dumps({
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": "{\"path\": \".env.local\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "SECRET_KEY=mysupersecretvalue\nDB=localhost"},
            ],
        })
        scan_request(body)
        assert session_stats()["secrets_protected"] >= 1

    def test_no_env_scan_without_sensitive_file(self):
        """KEY=VALUE content is NOT scanned if no sensitive filename is present."""
        body = json.dumps({
            "messages": [{
                "role": "tool",
                "tool_call_id": "c1",
                "content": "NAME=JohnDoeSmith\nAGE=25",
            }],
        })
        scan_request(body)
        assert session_stats()["secrets_protected"] == 0

    def test_invalid_json_no_crash(self):
        scan_request("not valid json {{{")
        assert session_stats()["secrets_protected"] == 0


# ── Session stats ────────────────────────────────────────────────────────────

class TestSessionStats:
    def test_starts_empty(self):
        assert session_stats() == {"secrets_protected": 0}

    def test_counts_registered(self):
        scan_tokens("AKIA" + "Z" * 16)
        scan_tokens("ghp_" + "a" * 36)
        assert session_stats()["secrets_protected"] == 2

    def test_reset_clears(self):
        scan_tokens("AKIA" + "Z" * 16)
        reset_session()
        assert session_stats()["secrets_protected"] == 0


