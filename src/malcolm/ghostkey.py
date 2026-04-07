"""
ghostkey.py — secret obfuscation middleware for Malcolm.

Intercepts requests before they reach the LLM backend and replaces
secret values with format-preserving fakes. Restores them in responses
so logs and tool calls always show real values locally, while the
upstream API never sees them.

Enable via environment variable:
    MALCOLM_GHOSTKEY_ENABLED=true

Wire into Malcolm's FastAPI app:
    from malcolm.ghostkey import GhostKeyMiddleware
    if settings.ghostkey_enabled:
        app.add_middleware(GhostKeyMiddleware)
"""

import json
import logging
import random
import re
import string
import threading

from starlette.types import ASGIApp, Message as ASGIMessage, Receive, Scope, Send

from malcolm.formats import _find_request_parser

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
    r"sk-ant-[a-zA-Z0-9\-_]{20,}",
    r"sk-proj-[a-zA-Z0-9\-_]{40,}",
    r"sk-[a-zA-Z0-9]{48}",
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


def scan_request(body: str) -> None:
    scan_tokens(body)
    try:
        data = json.loads(body)
        parser = _find_request_parser(data)
        if parser is None:
            return
        messages = parser.parse_request_messages(data)

        # Collect file paths from tool_call arguments
        has_sensitive_file = False
        for msg in messages:
            for tc in msg.tool_calls:
                for s in _extract_strings(json.loads(tc.arguments)) if tc.arguments else []:
                    if is_sensitive_file(s):
                        has_sensitive_file = True

        # If a sensitive file was read, scan tool result contents
        if has_sensitive_file:
            for msg in messages:
                if msg.tool_result:
                    scan_env_content(msg.tool_result)
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


# ── Middleware ────────────────────────────────────────────────────────────────

class GhostKeyMiddleware:
    """
    Pure ASGI middleware — obfuscates secrets in LLM requests,
    restores them in responses. Transparent to Malcolm's logging.

    Handles both regular and streaming (SSE) responses: streaming
    responses are processed chunk-by-chunk to preserve incremental
    delivery.

    Usage in Malcolm's app setup:
        if settings.ghostkey_enabled:
            app.add_middleware(GhostKeyMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # ── Read and obfuscate request body ──────────────────────────
        body_chunks: list[bytes] = []
        while True:
            msg = await receive()
            body_chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break

        raw_body = b"".join(body_chunks)
        body_text = raw_body.decode(errors="replace")

        scan_request(body_text)
        clean = obfuscate(body_text)

        if clean != body_text:
            log.info("ghostkey: obfuscated secrets in outgoing request")

        clean_bytes = clean.encode()
        body_sent = False

        async def receive_with_clean_body() -> ASGIMessage:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": clean_bytes}
            # After body is sent, return disconnect on further reads
            return {"type": "http.disconnect"}

        # ── Intercept and restore response body ──────────────────────
        async def send_with_restore(message: ASGIMessage) -> None:
            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    text = chunk.decode(errors="replace")
                    restored = restore(text)
                    message = {**message, "body": restored.encode()}
            await send(message)

        await self.app(scope, receive_with_clean_body, send_with_restore)
