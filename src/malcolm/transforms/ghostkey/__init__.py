"""GhostKey transform — obfuscates secrets in requests and restores them in responses."""

from __future__ import annotations

import json

from malcolm.transforms.ghostkey.engine import obfuscate, restore, scan_request


class GhostKeyTransform:
    name = "ghostkey"
    stores_snapshot = True

    def transform_request(self, body: dict) -> dict:
        text = json.dumps(body)
        scan_request(text)
        clean = obfuscate(text)
        if clean == text:
            return body
        return json.loads(clean)

    def transform_response(self, body: dict, model: str = "") -> dict:
        text = json.dumps(body)
        restored = restore(text)
        if restored == text:
            return body
        return json.loads(restored)

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        return [restore(line)]

    def rewrite_path(self, path: str) -> str:
        return path


def create(config: dict) -> GhostKeyTransform:
    return GhostKeyTransform()
