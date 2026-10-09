"""Project path resolution.

Every path in the codebase is derived from a single project root so the repo can
be moved or copied without editing anything. Resolution order for the **code
root** (where ``configs/`` lives):

1. The nearest ancestor of this file containing both ``pyproject.toml`` and
   ``configs/``.
2. The current working directory.
3. ``src/anti_uav/utils/paths.py`` -> four levels up.

Two environment overrides exist, and they do different jobs on purpose:

``ANTI_UAV_ROOT``
    Relocates everything. Rarely needed - the code root is found from ``__file__``
    anyway.

``ANTI_UAV_DATA_DIR``
    Relocates only the mutable trees (``data/`` and ``artifacts/``). This is the
    useful one: it lets a test or a smoke script write frames to a scratch
    directory while still reading the real ``configs/``. It is also how you point
    the 100 GB dataset tree at a different drive without duplicating the repo.

All mutable artefacts live under ``data/`` (datasets) and ``artifacts/`` (runs,
exports, tracks). Both are gitignored, so a stray ``git add -A`` cannot commit
100 GB of frames.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ENV_ROOT = "ANTI_UAV_ROOT"
_ENV_DATA_DIR = "ANTI_UAV_DATA_DIR"

#: Where each symbolic key lives, relative to its root.
_CODE_SUBDIRS = {
    "configs": "configs",
    "docs": "docs",
    "diagrams": "docs/diagrams",
    "scripts": "scripts",
    "tests": "tests",
}

#: Where each symbolic key lives, relative to the data root.
_DATA_SUBDIRS = {
    "data": "",
    "raw": "data/raw",
    "interim": "data/interim",
    "processed": "data/processed",
    "manifests": "data/manifests",
    "artifacts": "artifacts",
    "runs": "artifacts/runs",
    "exports": "artifacts/exports",
    "tracks": "artifacts/tracks",
    "deploy": "artifacts/deploy",
}

_SUBDIRS = {**_CODE_SUBDIRS, **_DATA_SUBDIRS}


def _has_layout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "configs").is_dir()


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if _has_layout(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _has_layout(candidate):
            return candidate

    # Last resort: src/anti_uav/utils/paths.py -> project root is 4 levels up.
    return here.parents[3]


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Absolute path to the repository root (the one holding ``configs/``)."""
    override = os.environ.get(_ENV_ROOT)
    if override:
        candidate = Path(override).expanduser().resolve()
        if _has_layout(candidate) or (candidate / "configs").is_dir():
            return candidate
    return _find_root()


@lru_cache(maxsize=1)
def data_root() -> Path:
    """Base directory for ``data/`` and ``artifacts/``.

    Defaults to :func:`project_root`, but ``ANTI_UAV_DATA_DIR`` moves it. This is
    what lets tests and smoke scripts write frames somewhere disposable without
    losing access to the real configs.
    """
    override = os.environ.get(_ENV_DATA_DIR)
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return project_root()


def subdir(key: str) -> Path:
    """Return an existing project subdirectory by symbolic key."""
    try:
        rel = _SUBDIRS[key]
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(f"unknown path key {key!r}; known: {sorted(_SUBDIRS)}") from exc
    base = project_root() if key in _CODE_SUBDIRS else data_root()
    path = base / rel if rel else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    """``mkdir -p`` returning the path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_root(alias: str) -> Path:
    """``data/raw/<alias>`` - where a dataset's downloaded artefacts land."""
    return ensure_dir(subdir("raw") / alias)


def combo_root(combo_slug: str) -> Path:
    """``data/processed/<combo_slug>`` - materialised YOLO dataset for a combo."""
    return ensure_dir(subdir("processed") / combo_slug)


def relative_to_root(path: str | Path) -> str:
    """POSIX-style path relative to the project root, for logs and manifests."""
    p = Path(path).resolve()
    try:
        return p.relative_to(project_root()).as_posix()
    except ValueError:
        return p.as_posix()
