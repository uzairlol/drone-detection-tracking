"""Tracking: local MOT, global fusion, coordination, metrics.

Layers, from the inside out, matching the system block diagram:

``local``            per-camera tracking - the ``nvtracker`` element
``global_tracker``   cross-camera identity fusion - the Global Track Manager
``coordination``     handoff, recovery, scheduling, PTZ - section 3
``metrics``          MOTA / IDF1 / HOTA / Anti-UAV accuracy
``replayer``         offline harness that ties it all together
``appearance``       ReID embeddings for the association gate
"""

from __future__ import annotations

from .appearance import AppearanceEmbedder, cosine, crop_box
from .kalman import KalmanBank, KalmanTrack2D
from .local import LOCAL_TRACKERS, BaseTracker, build_tracker
from .metrics import (
    EvalReport,
    FrameGT,
    FramePred,
    SequenceResult,
    comparison_table,
    evaluate_report,
    evaluate_sequence,
    tracks_to_histories,
)
from .replayer import (
    ReplayResult,
    available_sequences,
    evaluate_trackers_on_dataset,
    format_reports,
    replay_sequence,
)
from .types import (
    FrameTracks,
    HandoffEvent,
    Track,
    TrackObservation,
    TrackState,
    trackable,
)

__all__ = [
    "LOCAL_TRACKERS",
    "AppearanceEmbedder",
    "BaseTracker",
    "EvalReport",
    "FrameGT",
    "FramePred",
    "FrameTracks",
    "HandoffEvent",
    "KalmanBank",
    "KalmanTrack2D",
    "ReplayResult",
    "SequenceResult",
    "Track",
    "TrackObservation",
    "TrackState",
    "available_sequences",
    "build_tracker",
    "comparison_table",
    "cosine",
    "crop_box",
    "evaluate_report",
    "evaluate_sequence",
    "evaluate_trackers_on_dataset",
    "format_reports",
    "replay_sequence",
    "trackable",
    "tracks_to_histories",
]
