"""Filesystem conventions for the Scalable-HMRF repository.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-20
Updated: 2026-10-20

All path handling is centralised here so that scripts never embed absolute
paths. The repository root is discovered by walking upwards until
``config/defaults.yaml`` is found, which makes the whole tree relocatable.

Environment overrides
---------------------
HMRF_ROOT        explicit repository root (highest priority)
HMRF_XENIUM_DIR  external Xenium output directory (not shipped with the repo)

Usage
-----
    from hmrf.paths import repo_root, data_dir, results_dir, figures_dir

    results = results_dir("h9-dlpfc12-benchmark")
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root.

    Args:
        start: File or directory used as the starting point of the upward
            search. Defaults to this module's location.

    Returns:
        Absolute path of the directory that contains ``config/defaults.yaml``.

    Raises:
        RuntimeError: If no configuration file is found in any ancestor.
    """
    env = os.environ.get("HMRF_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    current = Path(start or __file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config" / "defaults.yaml").exists():
            return candidate
    raise RuntimeError(
        "repository root not found: no config/defaults.yaml in any parent "
        f"of {current}. Set HMRF_ROOT to override."
    )


def data_dir() -> Path:
    """Return the raw-data directory (``data/``, excluded from version control)."""
    return repo_root() / "data"


def results_dir(experiment: str) -> Path:
    """Return the results directory of one experiment and create it if needed.

    Args:
        experiment: Experiment directory name, e.g. ``h9-dlpfc12-benchmark``.

    Returns:
        ``<root>/experiments/<experiment>/results`` (created on demand).
    """
    path = repo_root() / "experiments" / experiment / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    """Return the figure output directory and create it if needed."""
    path = repo_root() / "figures" / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def xenium_dir(required: bool = True) -> Path | None:
    """Resolve the external Xenium output directory.

    Args:
        required: When True, raise if the environment variable is unset.

    Returns:
        Absolute path, or None when ``required=False`` and the variable is
        not configured.

    Raises:
        RuntimeError: If ``required=True`` and ``HMRF_XENIUM_DIR`` is unset.
    """
    value = os.environ.get("HMRF_XENIUM_DIR")
    if not value:
        if required:
            raise RuntimeError(
                "HMRF_XENIUM_DIR is not set; export it to the folder that "
                "contains the Xenium 'outs' directory."
            )
        return None
    return Path(value).expanduser().resolve()
