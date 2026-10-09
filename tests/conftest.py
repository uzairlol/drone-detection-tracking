"""Shared fixtures.

The suite must not touch the developer's real ``data/`` or ``artifacts/``
directories, and must not need a GPU, a dataset or a trained checkpoint. Every
test that touches the filesystem gets a tmp data root instead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _temp_data_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point the whole session at a throwaway data root.

    Set before any anti_uav import, because the path helpers read the env var at
    call time but the config loaders cache, and a session-scoped override keeps
    every module consistent.
    """
    root = tmp_path_factory.mktemp("anti_uav_tests")
    previous = os.environ.get("ANTI_UAV_DATA_DIR")
    os.environ["ANTI_UAV_DATA_DIR"] = str(root)
    yield root
    if previous is None:
        os.environ.pop("ANTI_UAV_DATA_DIR", None)
    else:
        os.environ["ANTI_UAV_DATA_DIR"] = previous


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """An isolated data root for a single test."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def blank_image(tmp_path: Path) -> Path:
    """A real, decodable 640x360 JPEG. Not black - phash and focus_score both
    behave differently on a constant image."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    path = tmp_path / "frame.jpg"
    Image.fromarray(pixels).save(path, quality=92)
    return path
