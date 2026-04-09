"""
ghostkey.py — secret obfuscation engine for Malcolm.

Scans request bodies for secret values (API keys, tokens, credentials)
and replaces them with format-preserving fakes.  Restores real values
in responses so the client always sees originals while the upstream API
never receives them.

Used by the transform pipeline (``GhostKeyTransform`` in transforms/ghostkey.py).
Enable by adding ``ghostkey`` to the transforms list in ``malcolm.yaml``.
"""

import json
import logging
import random
import re
import string
import threading

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MIN_SECRET_LEN = 8
SEPARATORS     = set("-_.:/@#")

SENSITIVE_FILES = {
    ".env", ".env.local", ".env.production", ".env.staging",
    ".env.development", ".env.test", ".env.backup",
    "secrets.yml", "secrets.yaml", "secrets.json",
    "credentials.yml", "credentials.yaml", "credentials.json",
    "credentials", "service-account.json",
    ".npmrc", ".pypirc", ".netrc", ".docker/config.json",
    "terraform.tfvars", "terraform.tfvars.json",
    "database.yml", "secrets.rb",
    "id_rsa", "id_ed25519",
}

_RAW_PATTERNS = [
    r"sk-[a-zA-Z0-9\-_]{20,}",
    r"github_pat_[a-zA-Z0-9_]{82}",
    r"gh[poas]_[a-zA-Z0-9]{36}",
    r"ghr_[a-zA-Z0-9]{36}",
    r"ASIA[0-9A-Z]{16}",
    r"AKIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z\-_]{35}",
    r"ya29\.[0-9A-Za-z\-_]{20,}",
    r"[rp]k_live_[a-zA-Z0-9]{24,}",
    r"sk_(?:live|test)_[a-zA-Z0-9]{24,}",
    r"SK[0-9a-fA-F]{32}",
    r"AC[0-9a-fA-F]{32}",
    r"xox[bposa]-[0-9a-zA-Z\-]{10,}",
    r"hf_[a-zA-Z0-9]{34,}",
    r"dp\.[sp]t\.[a-zA-Z0-9]{43}",
    r"hv[sb]\.[a-zA-Z0-9]{24,}",
    r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}",
    r"key-[0-9a-zA-Z]{32}",
    r"npm_[a-zA-Z0-9]{36}",
    r"vercel_[a-zA-Z0-9]{24,}",
    r"sbp_[a-zA-Z0-9]{40}",
    r"pscale_tkn_[a-zA-Z0-9\-_]{32,}",
    r"railway_[a-zA-Z0-9]{32,}",
    r"eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+",
    r"(?<=Bearer )[a-zA-Z0-9\-_=+/]{20,}",
    r"\b[0-9a-f]{32,64}\b",
]

TOKEN_PATTERNS = [re.compile(p) for p in _RAW_PATTERNS]

# ── Session dictionary ────────────────────────────────────────────────────────

_real_to_fake: dict[str, str] = {}
_fake_to_real: dict[str, str] = {}
_dict_lock = threading.Lock()


def session_stats() -> dict:
    with _dict_lock:
        return {"secrets_protected": len(_real_to_fake)}


def reset_session() -> None:
    """Clear all registered secrets. Useful for testing."""
    with _dict_lock:
        _real_to_fake.clear()
        _fake_to_real.clear()


# ── Fake generation ───────────────────────────────────────────────────────────

def _same_charset(ch: str) -> str:
    if ch.isdigit():  return random.choice(string.digits)
    if ch.isupper():  return random.choice(string.ascii_uppercase)
    if ch.islower():  return random.choice(string.ascii_lowercase)
    return ch


def _natural_prefix(real: str) -> int:
    for i, ch in enumerate(real):
        if ch in SEPARATORS:
            return i + 1
    return min(5, len(real))


def _make_fake(real: str) -> str:
    n = _natural_prefix(real)
    return real[:n] + "".join(_same_charset(ch) for ch in real[n:])


def _register(real: str) -> str:
    with _dict_lock:
        if real in _real_to_fake:
            return _real_to_fake[real]
        fake = _make_fake(real)
        _real_to_fake[real] = fake
        _fake_to_real[fake] = real
        log.info("ghostkey: registered %s***", real[:4])
        return fake


# ── Sensitive file helpers ────────────────────────────────────────────────────

def is_sensitive_file(name: str) -> bool:
    name = name.lower()
    if any(name.endswith(s) or s in name for s in SENSITIVE_FILES):
        return True
    basename = name.split("/")[-1].split("\\")[-1]
    if basename.startswith(".env"):
        return True
    return any(name.endswith(ext) for ext in (".pem", ".key", ".p12", ".pfx"))


# ── Scanning ──────────────────────────────────────────────────────────────────

def scan_tokens(text: str) -> None:
    for pattern in TOKEN_PATTERNS:
        for m in pattern.finditer(text):
            token = m.group(0)
            if len(token) >= MIN_SECRET_LEN and token not in _real_to_fake:
                _register(token)


def scan_env_content(content: str) -> None:
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if len(value) >= MIN_SECRET_LEN and value not in _real_to_fake:
            _register(value)


def _extract_strings(obj) -> list[str]:
    """Recursively extract all string values from a JSON structure."""
    strings: list[str] = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_extract_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(_extract_strings(value))
    return strings


def _extract_tool_arguments(messages: list) -> list[str]:
    """Extract tool call argument strings from messages (OpenAI & Anthropic)."""
    args: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # OpenAI: tool_calls[].function.arguments
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("arguments"):
                args.append(fn["arguments"])
        # Anthropic: content[].{type: "tool_use"}.input
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                inp = block.get("input")
                if isinstance(inp, dict):
                    args.append(json.dumps(inp))
    return args


def _extract_tool_results(messages: list) -> list[str]:
    """Extract tool result content from messages (OpenAI & Anthropic)."""
    results: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # OpenAI: role=tool, content=string
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            results.append(msg["content"])
        # Anthropic: content[].{type: "tool_result"}.content
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                c = block.get("content", "")
                if isinstance(c, str):
                    results.append(c)
    return results


def scan_request(body: str) -> None:
    scan_tokens(body)
    try:
        data = json.loads(body)
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return

        # Check tool call arguments for sensitive file paths
        has_sensitive_file = False
        for arg_str in _extract_tool_arguments(messages):
            for s in _extract_strings(json.loads(arg_str)) if arg_str.startswith("{") else _extract_strings(arg_str):
                if is_sensitive_file(s):
                    has_sensitive_file = True

        # If a sensitive file was read, scan tool result contents
        if has_sensitive_file:
            for result in _extract_tool_results(messages):
                scan_env_content(result)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass


# ── Obfuscate / restore ───────────────────────────────────────────────────────

def obfuscate(text: str) -> str:
    with _dict_lock:
        snapshot = dict(_real_to_fake)
    for real, fake in snapshot.items():
        if real in text:
            text = text.replace(real, fake)
    return text


def restore(text: str) -> str:
    with _dict_lock:
        snapshot = dict(_fake_to_real)
    for fake in sorted(snapshot, key=len, reverse=True):
        if fake in text:
            text = text.replace(fake, snapshot[fake])
    return text


