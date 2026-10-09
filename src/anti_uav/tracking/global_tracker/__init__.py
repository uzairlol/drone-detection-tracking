"""Global (cross-camera) tracking.

* :mod:`manager` - fuses per-camera local tracks into global identities
"""

from __future__ import annotations

from .manager import (
    FusionConfig,
    FusionResult,
    GlobalTrack,
    GlobalTrackManager,
    attach_ground_positions,
    config_from_rules,
)

__all__ = [
    "FusionConfig",
    "FusionResult",
    "GlobalTrack",
    "GlobalTrackManager",
    "attach_ground_positions",
    "config_from_rules",
]
