"""YAML config loading with dotted-key overrides (`--set retrieval.top_k=8`)."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(os.environ.get("ROBORAG_CONFIG", "configs/default.yaml"))


def load_config(path: Path | str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    if os.environ.get("ROBORAG_MODEL"):
        cfg.setdefault("generation", {})["model"] = os.environ["ROBORAG_MODEL"]
    return cfg


def with_overrides(cfg: dict, **sections: dict) -> dict:
    out = copy.deepcopy(cfg)
    for name, values in sections.items():
        out.setdefault(name, {}).update(values)
    return out
