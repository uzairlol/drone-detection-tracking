"""Deterministic seeding.

Reproducibility matters more than usual here because the headline number is a
comparison across 14 runs. If run A's split or augmentation order depends on
wall-clock entropy, then any mAP delta between two runs is uninterpretable.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (when importable) PyTorch.

    ``PYTHONHASHSEED`` is exported too, but note it only takes effect for
    *child* processes, so this is a best-effort for dataloader workers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Slower, but removes cudnn autotune nondeterminism from the
            # comparison table. Toggle off for a quick throughput probe.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass

    return seed


def rng_for(seed: int, *stream: Any) -> np.random.Generator:
    """A named, reproducible generator.

    Usage: ``rng_for(seed, "split", dataset)``. Two calls with the same seed and
    stream always produce the same stream, so adding a new consumer never
    perturbs an existing one.

    The key goes through :func:`stable_hash`, not ``hash()``. Python salts string
    hashing per process unless ``PYTHONHASHSEED`` is pinned, so using ``hash()``
    here gave every run a different train/val split - which makes the seed a lie
    and silently makes the experiment matrix irreproducible.
    """
    key = "|".join([str(seed), *(str(s) for s in stream)])
    return np.random.default_rng(stable_hash(key))


def stable_hash(text: str) -> int:
    """Process-independent hash (``hash()`` is salted per process)."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
