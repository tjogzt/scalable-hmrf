"""YAML-backed configuration loader.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-20
Updated: 2026-10-20

Business constants (datasets, slice lists, seeds, parameter grids, palettes)
live in ``config/defaults.yaml``. Scripts call :func:`load_config` once and read
values from the returned mapping; nothing is hard-coded in analysis code.

Environment overrides
---------------------
HMRF_CONFIG  path of an alternative YAML file (full replacement)
HMRF_SEED    integer that overrides ``project.seed`` (reproducibility runs)

Usage
-----
    from hmrf.config import load_config

    cfg = load_config()
    betas = cfg["experiments"]["h9"]["betas"]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from hmrf.paths import repo_root


def _expand(value: Any, root: Path) -> Any:
    """Recursively expand ``${ROOT}`` placeholders inside strings."""
    if isinstance(value, str):
        value = value.replace("${ROOT}", str(root))
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, root) for key, item in value.items()}
    return value


def load_config(path: str | Path | None = None) -> dict:
    """Load and validate the project configuration.

    Args:
        path: Optional explicit YAML path. Defaults to ``HMRF_CONFIG`` or
            ``<root>/config/defaults.yaml``.

    Returns:
        Nested mapping with ``project``, ``paths``, ``datasets``,
        ``experiments`` and ``figures`` sections. ``${ROOT}`` placeholders are
        expanded and the seed is overridable through ``HMRF_SEED``.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If a required top-level section is missing.
    """
    root = repo_root()
    config_path = Path(path or os.environ.get("HMRF_CONFIG") or root / "config" / "defaults.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = _expand(config, root)

    for section in ("project", "paths", "datasets", "experiments"):
        if section not in config:
            raise ValueError(f"configuration is missing the '{section}' section")

    seed_override = os.environ.get("HMRF_SEED")
    if seed_override:
        config["project"]["seed"] = int(seed_override)
    return config
