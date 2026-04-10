"""
transforms — pluggable pipeline of Transforms and Annotators for Malcolm.

Each plugin lives in its own module and exposes a ``create(config)`` factory.
Plugins come in two shapes:

* **Transforms** mutate request/response bodies, stream lines and paths.
* **Annotators** observe traffic and produce structured :class:`Annotation`
  objects for storage and TUI rendering.

A single plugin may implement either or both protocols.  The pipeline is
assembled at startup via ``build_pipeline()`` using the list defined in
``malcolm.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, Iterable

import yaml

from malcolm.transforms._base import Annotation, Annotator, Transform
from malcolm.transforms.ghostkey import GhostKeyTransform
from malcolm.transforms.ghostkey import create as _create_ghostkey
from malcolm.transforms.llm_annotator import LLMAnnotator
from malcolm.transforms.llm_annotator import create as _create_llm_annotator
from malcolm.transforms.translation import TranslationTransform
from malcolm.transforms.translation import create as _create_translation

logger = logging.getLogger("malcolm.transforms")

ENTRY_POINT_GROUP = "malcolm.transforms"


@dataclass
class Pipeline:
    """Assembled pipeline of transforms and annotators."""

    transforms: list[Transform] = field(default_factory=list)
    annotators: list[Annotator] = field(default_factory=list)


__all__ = [
    "Annotation",
    "Annotator",
    "Transform",
    "GhostKeyTransform",
    "LLMAnnotator",
    "TranslationTransform",
    "Pipeline",
    "build_pipeline",
    "REGISTRY",
]

REGISTRY: dict[str, Callable[[dict], object]] = {
    "ghostkey": _create_ghostkey,
    "llm_annotator": _create_llm_annotator,
    "translation": _create_translation,
}


def _is_transform(plugin: object) -> bool:
    return hasattr(plugin, "transform_request")


def _is_annotator(plugin: object) -> bool:
    return hasattr(plugin, "annotate_request")


def _discover_entry_points(entries: Iterable) -> None:
    """Register external plugins exposed via the ``malcolm.transforms`` entry point group.

    Existing entries in ``REGISTRY`` win (built-ins cannot be shadowed, and the
    first external registration for a given name beats later ones). A plugin
    whose ``load()`` raises is logged and skipped, never propagated.
    """
    for ep in entries:
        name = ep.name
        if name in REGISTRY:
            logger.warning(
                "external plugin %r shadowed by existing registration, ignoring",
                name,
            )
            continue
        try:
            factory = ep.load()
        except Exception as exc:
            logger.warning("failed to load external plugin %r: %s", name, exc)
            continue
        REGISTRY[name] = factory


_discover_entry_points(entry_points(group=ENTRY_POINT_GROUP))


def _load_transform_list(config_file: str) -> list[dict[str, dict]]:
    """Load the plugin list from a YAML config file.

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


def build_pipeline(config_file: str = "malcolm.yaml") -> Pipeline:
    """Assemble the pipeline from the YAML config file.

    Each plugin in the config is instantiated and classified into
    :attr:`Pipeline.transforms` and/or :attr:`Pipeline.annotators` based on
    the methods it implements.  A plugin that implements both protocols
    appears in both lists.
    """
    entries = _load_transform_list(config_file)

    pipeline = Pipeline()
    for entry in entries:
        for name, config in entry.items():
            factory = REGISTRY.get(name)
            if factory is None:
                raise ValueError(
                    f"Unknown plugin: {name!r}. "
                    f"Available: {sorted(REGISTRY)}"
                )
            plugin = factory(config)
            matched = False
            if _is_transform(plugin):
                pipeline.transforms.append(plugin)
                matched = True
            if _is_annotator(plugin):
                pipeline.annotators.append(plugin)
                matched = True
            if not matched:
                raise ValueError(
                    f"Plugin {name!r} is neither a Transform nor an Annotator"
                )

    if pipeline.transforms:
        logger.info("transforms: %s", [t.name for t in pipeline.transforms])
    if pipeline.annotators:
        logger.info("annotators: %s", [a.name for a in pipeline.annotators])

    return pipeline
