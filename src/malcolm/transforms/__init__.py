"""
transforms — pluggable request/response transform pipeline for Malcolm.

Each transform lives in its own module and exposes a ``create(config)``
factory.  The pipeline is assembled at startup via ``build_pipeline()``
using the transform list defined in ``malcolm.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import yaml

from malcolm.transforms._base import Transform
from malcolm.transforms.ghostkey import GhostKeyTransform
from malcolm.transforms.ghostkey import create as _create_ghostkey
from malcolm.transforms.translation import TranslationTransform
from malcolm.transforms.translation import create as _create_translation

logger = logging.getLogger("malcolm.transforms")

__all__ = [
    "Transform",
    "GhostKeyTransform",
    "TranslationTransform",
    "build_pipeline",
    "REGISTRY",
]

REGISTRY: dict[str, Callable[[dict], Transform]] = {
    "ghostkey": _create_ghostkey,
    "translation": _create_translation,
}


def _load_transform_list(config_file: str) -> list[dict[str, dict]]:
    """Load the transforms list from a YAML config file.

    Each entry is either a plain string (no config) or a single-key dict
    (name → config dict).  Returns a normalised list of ``{name: config}``
    dicts.
    """
    path = Path(config_file)
    if not path.exists():
        return []

    data = yaml.safe_load(path.read_text()) or {}
    raw = data.get("transforms", [])

    result: list[dict[str, dict]] = []
    for entry in raw:
        if isinstance(entry, str):
            result.append({entry: {}})
        elif isinstance(entry, dict):
            for name, cfg in entry.items():
                result.append({name: cfg if isinstance(cfg, dict) else {}})
    return result


def build_pipeline(config_file: str = "malcolm.yaml") -> list[Transform]:
    """Assemble the transform pipeline from the YAML config file."""
    entries = _load_transform_list(config_file)

    pipeline: list[Transform] = []
    for entry in entries:
        for name, config in entry.items():
            factory = REGISTRY.get(name)
            if factory is None:
                raise ValueError(
                    f"Unknown transform: {name!r}. "
                    f"Available: {sorted(REGISTRY)}"
                )
            pipeline.append(factory(config))

    if pipeline:
        logger.info("transform pipeline: %s", [t.name for t in pipeline])

    return pipeline
