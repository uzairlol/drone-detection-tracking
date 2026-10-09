"""Core tracking types.

Shared vocabulary between the local trackers, the global track manager, the rule
engine and the metrics harness. Keeping these as plain dataclasses rather than
anything array-shaped matters: the global tracking logic is mostly branching and
threshold comparison, and readable branching beats clever vectorisation for code
whose whole job is to be obviously correct at 3 a.m. on a site with no network.

Coordinate frames
-----------------
A :class:`Track` carries its box in **image pixels of the camera that owns it**,
and - when a :class:`~anti_uav.utils.geometry.PinholeCamera` is attached - also a
ground-plane estimate in metres. The global manager works in ground space because
cross-camera association has no meaningful image space; pixel space only makes
sense for the local tracker.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..utils.geometry import BBox, box_centre, box_wh, iou_matrix

#: Below this pixel height no tracker in this repo holds a track reliably.
#: Mirrors rules.tracking.min_trackable_height_px (15 px) in drone_rules.yaml.
MIN_TRACKABLE_HEIGHT_PX = 15.0


class TrackState(StrEnum):
    """Lifecycle of a single track."""

    TENTATIVE = "tentative"  # fewer than persistence.min_hits detections
    CONFIRMED = "confirmed"  # passed the persistence rule
    LOST = "lost"  # missed for max_target_age_frames, still recoverable
    DEAD = "dead"  # exceeded max gap or recovery budget


@dataclass(slots=True)
class TrackObservation:
    """One frame's worth of evidence about a track."""

    frame_index: int
    timestamp_s: float
    box: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "drone"
    camera_id: str = ""
    node: str = ""
    #: The local (per-camera) track this observation belongs to. The global track
    #: manager needs it to record which local track feeds each global identity -
    #: without it, a handoff cannot be attributed to a specific stream.
    track_id: int | None = None
    #: Height of the source frame in pixels. The rule layer needs it to turn a
    #: normalised horizon line from ``coverage_map.yaml`` into pixel coordinates.
    image_height_px: int = 0
    #: Ground-plane centre in metres, when a calibrated camera is available.
    ground_xy: tuple[float, float] | None = None
    #: Ground-plane height above the reference plane, metres.
    ground_z: float | None = None
    #: L2-normalised appearance embedding. ``None`` disables the appearance gate.
    embedding: np.ndarray | None = None
    #: Camera-space velocity, px/frame, from the Kalman prediction step.
    velocity_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def centre(self) -> tuple[float, float]:
        return box_centre(self.box)

    @property
    def height_px(self) -> float:
        return box_wh(self.box)[1]


@dataclass(slots=True)
class Track:
    """One tracked object, local or global.

    ``global_id`` is ``None`` for a local per-camera track and set once the global
    track manager has fused it across cameras. The block diagram's handoff
    coordinator moves the *global* identity between cameras, so the two are kept
    distinct rather than reusing one integer for both.
    """

    track_id: int
    class_id: int = 0
    class_name: str = "drone"
    camera_id: str = ""
    node: str = ""
    state: TrackState = TrackState.TENTATIVE

    #: Height of the source frame in pixels, copied from the first observation. The
    #: rule layer's above-horizon test needs it to convert the normalised
    #: ``horizon_y`` from coverage_map.yaml into pixels.
    image_height_px: int = 0
    #: Latest observation, and the smoothed box the tracker publishes.
    #: NOTE: this is always the last *measured* box. Prediction is applied on demand
    #: via :meth:`predicted_box` and never written back here - overwriting the
    #: measurement with an extrapolation makes velocity compound frame over frame
    #: (predicted box -> velocity measured against prediction -> bigger prediction),
    #: which loses the track entirely within a few frames on a fast target.
    observation: TrackObservation | None = None
    #: The track's published box: the most recent measurement.
    #:
    #: NOT a low-pass filtered version of it. A first-order smoother on a moving
    #: target has a steady-state lag of ``alpha * v / (1 - alpha)`` - at alpha=0.5
    #: and 14 px/frame that is 14 px, which on a 40 px box pushes IoU below 0.5 and
    #: makes a perfectly good tracker score as a miss. Smoothing belongs to the
    #: *display* box; the tracking box is the measurement.
    box: BBox = (0.0, 0.0, 0.0, 0.0)
    #: Heavily smoothed box, for rendering only. Never used for association or IoU.
    smoothed_box: BBox = (0.0, 0.0, 0.0, 0.0)
    #: Image-space velocity, px/frame, from consecutive *measured* boxes.
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    #: Ground-plane state, metres. ``None`` when the camera is uncalibrated.
    ground_xy: tuple[float, float] | None = None
    ground_z: float | None = None
    velocity_m_s: tuple[float, float] = (0.0, 0.0)
    #: 3-sigma uncertainty radius, metres. The block diagram's 3σ covariance.
    uncertainty_m: float = 0.0

    hits: int = 1
    misses: int = 0
    age: int = 1
    first_frame: int = 0
    last_frame: int = 0
    first_timestamp_s: float = 0.0
    last_timestamp_s: float = 0.0

    #: Running confidence mean, so a track that only ever saw weak evidence is
    #: visible as such rather than being carried by the rule layer on one good frame.
    confidence: float = 0.0
    confidence_ema: float = 0.0

    #: Appearance gallery - the most recent embeddings, for re-identification.
    embedding: np.ndarray | None = None
    gallery: list[np.ndarray] = field(default_factory=list)

    #: Populated by the rule engine.
    confirmed: bool = False
    alert: str = ""
    notes: list[str] = field(default_factory=list)

    #: Global identity, assigned by the global track manager.
    global_id: int | None = None
    #: Cameras this track has been observed from. More than one = a handoff happened.
    seen_cameras: set[str] = field(default_factory=set)
    handoffs: int = 0

    #: ``{frame_index: box}`` of every observed position, so MOT scoring can
    #: reconstruct frame-accurate history. Populated by the replayer rather than by
    #: the tracker itself - a live tracker has no use for it and it would grow
    #: without bound over a long sequence.
    history: dict[int, BBox] = field(default_factory=dict)
    #: ``{frame_index: timestamp_s}`` alongside :attr:`history`, so time-window
    #: rules (the hover check) do not have to assume a fixed frame rate.
    history_timestamps: dict[int, float] = field(default_factory=dict)
    #: ``{frame_index: ground height in metres}``, for the rule layer's hover test.
    history_heights: dict[int, float] = field(default_factory=dict)

    def update(
        self,
        observation: TrackObservation,
        *,
        smoothing: float = 0.5,
        camera_displacement: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Fold a matched observation into the track state.

        ``camera_displacement`` is the image-space shift the *camera* underwent
        between the previous and this frame. Subtracting it from the measured
        displacement is what keeps a pan from being learned as target motion: a
        PTZ that moves 200 px in one frame would otherwise hand the tracker a
        200 px/frame velocity, and the next frame's prediction would be off by the
        full pan distance — one camera move would cost the identity.
        """
        if self.observation is not None:
            # Velocity from consecutive MEASURED observations, minus camera motion.
            previous_centre = self.observation.centre
            current_centre = observation.centre
            self.velocity_xy = (
                current_centre[0] - previous_centre[0] - camera_displacement[0],
                current_centre[1] - previous_centre[1] - camera_displacement[1],
            )

        self.observation = observation
        self.box = observation.box
        if observation.image_height_px and not self.image_height_px:
            self.image_height_px = observation.image_height_px
        base = self.smoothed_box if self.hits > 1 else observation.box
        self.smoothed_box = _smooth_box(base, observation.box, smoothing)
        self.confidence = observation.confidence
        # Exponential moving average: a track whose confidence decays over time is
        # one the tracker is about to lose, and the rule layer should see that.
        self.confidence_ema = (
            observation.confidence
            if self.confidence_ema == 0.0
            else (1.0 - 0.3) * self.confidence_ema + 0.3 * observation.confidence
        )
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.last_frame = observation.frame_index
        self.last_timestamp_s = observation.timestamp_s
        self.state = TrackState.CONFIRMED if self.confirmed else TrackState.TENTATIVE

        if observation.ground_xy is not None:
            self.ground_xy = observation.ground_xy
        if observation.ground_z is not None:
            self.ground_z = observation.ground_z
        if observation.camera_id:
            self.seen_cameras.add(observation.camera_id)

        if observation.embedding is not None:
            self.embedding = observation.embedding
            self.gallery.append(observation.embedding)
            # 16 is enough to disambiguate re-entries without holding the whole
            # sequence, which matters because this list lives in memory for every
            # live track.
            if len(self.gallery) > 16:
                del self.gallery[0]

    def predicted_box(self, *, damping: float = 1.0) -> BBox:
        """Box extrapolated one frame forward using the last measured velocity.

        ``damping`` below 1.0 trades lag for stability, which is the right call for
        a 12 px target whose box jitters by more than its own width per frame -
        extrapolating the full measured velocity amplifies that jitter instead of
        cancelling it.
        """
        if self.observation is None:
            return self.box
        vx, vy = self.velocity_xy
        dx, dy = vx * damping, vy * damping
        x1, y1, x2, y2 = self.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def predict(self, *, smoothing: float = 0.5) -> BBox:
        """Backwards-compatible alias for :meth:`predicted_box`."""
        return self.predicted_box()

    def is_alive(self, frame_index: int, *, max_age: int, max_gap: int) -> bool:
        """Whether this track should still be considered trackable.

        ``max_age`` is the tight budget - roughly one PTZ move, from the block
        diagram's ``max_target_age_frames``. ``max_gap`` is the looser one -
        enough frames for the target to reappear after a brief occlusion.
        """
        gap = frame_index - self.last_frame
        return gap <= max(max_age, max_gap) and self.state is not TrackState.DEAD

    def duration_s(self) -> float:
        return max(0.0, self.last_timestamp_s - self.first_timestamp_s)

    def speed_m_s(self) -> float:
        vx, vy = self.velocity_m_s
        return math.hypot(vx, vy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "global_id": self.global_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "camera_id": self.camera_id,
            "node": self.node,
            "state": self.state.value,
            "box": [round(v, 2) for v in self.box],
            "confidence": round(self.confidence, 4),
            "confidence_ema": round(self.confidence_ema, 4),
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "ground_xy": None if self.ground_xy is None else [round(v, 2) for v in self.ground_xy],
            "ground_z": None if self.ground_z is None else round(self.ground_z, 2),
            "velocity_m_s": [round(v, 3) for v in self.velocity_m_s],
            "uncertainty_m": round(self.uncertainty_m, 3),
            "seen_cameras": sorted(self.seen_cameras),
            "handoffs": self.handoffs,
            "confirmed": self.confirmed,
            "alert": self.alert,
        }


def _smooth_box(previous: BBox, current: BBox, alpha: float) -> BBox:
    """Exponential smoothing of a box, for DISPLAY only.

    The centre is smoothed faster than the size: a target oscillating in the wind
    should not have its rendered box resized by the oscillation. Never feed this
    back into association or IoU scoring - see :attr:`Track.box`.
    """
    px, py = box_centre(previous)
    cx, cy = box_centre(current)
    pw, ph = box_wh(previous)
    cw, ch = box_wh(current)

    nx = px + alpha * (cx - px)
    ny = py + alpha * (cy - py)
    nw = pw + 0.25 * (cw - pw)
    nh = ph + 0.25 * (ch - ph)
    return (nx - nw / 2.0, ny - nh / 2.0, nx + nw / 2.0, ny + nh / 2.0)


@dataclass(slots=True)
class FrameTracks:
    """Tracks associated for one frame, before and after the update."""

    frame_index: int
    timestamp_s: float
    camera_id: str = ""
    tracks: list[Track] = field(default_factory=list)
    unmatched_detections: int = 0
    #: Detections that could not be associated and were not spawned into a new track.
    unassociated: list[TrackObservation] = field(default_factory=list)
    #: Confirmed tracks that matched nothing this frame.
    predicted_only: list[int] = field(default_factory=list)

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.state is TrackState.CONFIRMED]

    def by_id(self) -> dict[int, Track]:
        return {t.track_id: t for t in self.tracks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "camera_id": self.camera_id,
            "tracks": [t.to_dict() for t in self.tracks],
            "unassociated": len(self.unassociated),
            "predicted_only": list(self.predicted_only),
        }


class HandoffEvent(StrEnum):
    """Why a handoff or recovery was triggered.

    These are the states the block diagram's Handoff Coordinator and Lost-Target
    Recovery elements move between, named so a log line reads as the state machine
    transition it is.
    """

    NONE = "none"
    #: Predicted FOV exit inside the lead time; looking for a receiver.
    TRIGGERED = "triggered"
    #: A receiver has been commanded and is confirming.
    PENDING = "pending"
    #: Receiver confirmed; global identity transferred.
    COMPLETED = "completed"
    #: No receiver confirmed in time.
    FAILED = "failed"
    #: Track was lost and recovery is running.
    RECOVERY = "recovery"
    RECOVERED = "recovered"
    RECOVERY_FAILED = "recovery_failed"


def iou_matrix_of(boxes_a: Sequence[BBox], boxes_b: Sequence[BBox]) -> np.ndarray:
    """Convenience wrapper so callers need not import numpy helpers directly."""
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    return iou_matrix(
        np.asarray(boxes_a, dtype=float).reshape(-1, 4),
        np.asarray(boxes_b, dtype=float).reshape(-1, 4),
    )


def trackable(observation: TrackObservation, *, min_height_px: float = MIN_TRACKABLE_HEIGHT_PX) -> bool:
    """Whether a detection is large enough to track at all.

    The block diagram's "target pixel size (~15 px min to track)" gate. Below it,
    detection and tracking are different problems: the box jitters by more than
    its own size each frame, so any association is noise.
    """
    return observation.height_px >= min_height_px


def summarise(tracks: Iterable[Track]) -> dict[str, Any]:
    """Aggregate track statistics for the UI and the eval report."""
    items = list(tracks)
    return {
        "count": len(items),
        "confirmed": sum(1 for t in items if t.state is TrackState.CONFIRMED),
        "multi_camera": sum(1 for t in items if len(t.seen_cameras) > 1),
        "mean_hits": (sum(t.hits for t in items) / len(items)) if items else 0.0,
        "mean_uncertainty_m": (
            sum(t.uncertainty_m for t in items) / len(items) if items else 0.0
        ),
    }
