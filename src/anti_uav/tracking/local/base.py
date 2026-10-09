"""Base class for the local (per-camera) trackers.

Local tracking is what the block diagram's ``nvtracker`` element does inside one
worker node: detections in, short-lived per-camera tracks out, no knowledge of
the other 99 cameras. Keeping this layer separable from the global one is what
makes the two independently testable - the local tracker can be scored against
MOT ground truth (see :mod:`anti_uav.tracking.metrics`) without any of the
coordination machinery in the way.

Shared algorithm
----------------
Every tracker here is tracking-by-detection with the same three steps:

1. **predict** - advance each track one frame.
2. **associate** - solve a linear assignment between predicted tracks and
   detections, using whatever cost the tracker defines.
3. **update / birth / death** - matched pairs update, unmatched detections may
   spawn tracks, unmatched tracks age out.

They differ in three ways, and only these three:

* the cost function (IoU only / IoU plus Kalman distance / motion + appearance);
* whether low-confidence detections are used at all (ByteTrack's insight);
* how the cost threshold behaves with respect to detection confidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ...config.schema import TrackerName
from ...utils.geometry import BBox, iou_matrix, linear_assignment
from ...utils.logging import get_logger
from ..types import (
    MIN_TRACKABLE_HEIGHT_PX,
    FrameTracks,
    Track,
    TrackObservation,
    TrackState,
)

log = get_logger(__name__)

_EPS = 1e-6


class BaseTracker(ABC):
    """Common machinery for the local trackers."""

    #: Registry key.
    name: str = "base"
    #: Whether this tracker accepts low-confidence detections.
    uses_low_confidence: bool = False

    def __init__(
        self,
        *,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        match_threshold: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
        min_trackable_height_px: float = MIN_TRACKABLE_HEIGHT_PX,
        class_id: int = 0,
        **_ignored: Any,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_trackable_height_px = min_trackable_height_px
        self.class_id = class_id
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self.frame_index = 0
        self.last_timestamp_s = 0.0
        #: Homography for the frame currently being processed, or None. Set by
        #: update(); read by the cost path through association_boxes.
        self.global_motion: np.ndarray | None = None
        #: Image centre the camera displacement is measured at. Set from the first
        #: frame's size when known; a sensible default otherwise.
        self._motion_centre: tuple[float, float] = (640.0, 360.0)

    # -- subclass contract -------------------------------------------------- #

    @abstractmethod
    def _cost(self, tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
        """``(len(tracks), len(observations))`` association cost, lower is better."""

    def association_boxes(self, tracks: Sequence[Track]) -> np.ndarray:
        """Predicted track boxes with any camera motion this frame removed.

        Order matters and is the whole point: the **measured** box is projected
        forward through the camera transform first, and only then is the track's
        velocity added on top. Projecting the already-predicted box instead would
        also carry the velocity through the transform, so a pan would cancel the
        target's own motion as well as its own.

        Subclasses build their cost from these rather than calling
        :func:`predicted_boxes` directly, so that a caller passing
        ``global_motion=...`` to :meth:`update` actually sees the effect.
        """
        boxes = predicted_boxes(tracks)
        if self.global_motion is None or boxes.size == 0:
            return boxes
        measured = np.asarray(
            [t.predicted_box(damping=0.0) for t in tracks], dtype=float
        ).reshape(-1, 4)
        carried = project_boxes(measured, self.global_motion)
        return carried + (boxes - measured)

    # -- public API --------------------------------------------------------- #

    def update(
        self,
        observations: Sequence[TrackObservation],
        *,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
        camera_id: str = "",
        global_motion: np.ndarray | None = None,
    ) -> FrameTracks:
        """Advance one frame. ``observations`` are detections from this frame.

        ``global_motion`` is an optional 3x3 homography mapping *previous* image
        coordinates to *current* ones — from optical flow, or from a PTZ's own
        commanded delta. When the whole image shifted, every track looks like it
        moved at once; carrying the track boxes through the same transform removes
        that common-mode term so the motion model sees only what the target did.
        Detections are already in current coordinates and are left alone.
        """
        self.frame_index = frame_index if frame_index is not None else self.frame_index + 1
        self.last_timestamp_s = (
            timestamp_s if timestamp_s is not None else self.last_timestamp_s + 1.0 / 30.0
        )
        self.global_motion = global_motion

        # NB: no extrapolation is written back onto track.observation here. The
        # prediction is derived on demand in predicted_box(); storing it would make
        # the next frame's velocity measurement compare a prediction against a
        # prediction and compound the error.

        detections = [o for o in observations if o.class_id == self.class_id]
        high, low = self._split(detections)

        if observations:
            height = max((o.image_height_px for o in observations), default=0)
            if height:
                self._motion_centre = (640.0, height / 2.0)

        result = FrameTracks(
            frame_index=self.frame_index,
            timestamp_s=self.last_timestamp_s,
            camera_id=camera_id,
        )

        if not self._tracks:
            self._spawn_all(high, camera_id, result)
            self._prune()
            self._collect(list(self._tracks.values()), result)
            self._age(result)
            return result

        live = [t for t in self._tracks.values() if t.is_alive(
            self.frame_index, max_age=self.max_age, max_gap=self.max_age
        )]
        if not live:
            self._tracks.clear()
            self._spawn_all(high, camera_id, result)
            self._prune()
            self._collect(list(self._tracks.values()), result)
            self._age(result)
            return result

        matched, unmatched_track_idx, unmatched_det_idx = self._associate(live, high)

        for track_idx, det_idx, _cost in matched:
            self._match(live[track_idx], high[det_idx])

        for track_idx in unmatched_track_idx:
            track = live[track_idx]
            track.misses += 1
            if track.misses > self.max_age:
                track.state = TrackState.DEAD
            elif track.state is TrackState.CONFIRMED:
                track.state = TrackState.LOST
            result.predicted_only.append(track.track_id)

        # Stage 2 (optional): leftover tracks meet leftover low-confidence
        # detections. Runs BEFORE spawn so that only HIGH-confidence detections
        # can ever create a track, on every tracker.
        self._second_pass(live, high, low, result, camera_id)

        for det_idx in unmatched_det_idx:
            self._maybe_spawn(high[det_idx], camera_id, result)

        self._prune()
        self._collect(live, result)
        self._age(result)
        return result

    # -- steps -------------------------------------------------------------- #

    def _split(
        self, detections: Sequence[TrackObservation]
    ) -> tuple[list[TrackObservation], list[TrackObservation]]:
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]
        return high, low

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[TrackObservation],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """One linear-assignment pass on the high-confidence detections."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = self._cost(tracks, detections)
        pairs = linear_assignment(cost, max_cost=self.match_threshold)

        matched = [(t, d, c) for t, d, c in pairs]
        matched_track = {t for t, _d, _c in matched}
        matched_det = {d for _t, d, _c in matched}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
        return matched, unmatched_tracks, unmatched_dets

    def _second_pass(
        self,
        live: Sequence[Track],
        high: Sequence[TrackObservation],
        low: Sequence[TrackObservation],
        result: FrameTracks,
        camera_id: str,
    ) -> None:
        """Optional recovery pass over low-confidence detections (ByteTrack).

        Default is a no-op: SORT has no second pass, and forcing every tracker to
        declare an empty override would obscure which ones actually have the
        concept. ByteTrack and BoT-SORT override this.

        ``low`` is passed in already split by :meth:`update` rather than being
        re-derived here. Re-splitting ``high`` would yield an empty list, because
        every element of ``high`` is already at or above ``high_threshold``.
        """
        del live, high, low, result, camera_id

    def _match(self, track: Track, observation: TrackObservation) -> None:
        """Attach a matched detection. Velocity is derived inside ``Track.update``."""
        track.update(
            observation, smoothing=0.5, camera_displacement=self._camera_displacement()
        )
        if track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED

    def _camera_displacement(self) -> tuple[float, float]:
        """Image-space shift the camera made this frame, from ``global_motion``.

        Read off the transform at the image centre rather than derived from a
        difference of boxes, so a rotation or zoom contributes a sensible
        translation rather than dividing by zero.
        """
        if self.global_motion is None:
            return (0.0, 0.0)
        matrix = np.asarray(self.global_motion, dtype=float)
        if matrix.shape != (3, 3):
            return (0.0, 0.0)
        centre = self._motion_centre
        point = np.array([centre[0], centre[1], 1.0])
        projected = matrix @ point
        with np.errstate(divide="ignore", invalid="ignore"):
            mapped = projected[:2] / projected[2]
        if not np.all(np.isfinite(mapped)):
            return (0.0, 0.0)
        return (float(mapped[0] - centre[0]), float(mapped[1] - centre[1]))

    def _spawn_all(
        self,
        detections: Sequence[TrackObservation],
        camera_id: str,
        result: FrameTracks,
    ) -> None:
        for detection in detections:
            self._spawn(detection, camera_id)

    def _maybe_spawn(
        self,
        detection: TrackObservation,
        camera_id: str,
        result: FrameTracks,
    ) -> None:
        self._spawn(detection, camera_id)

    def _spawn(self, observation: TrackObservation, camera_id: str) -> Track:
        track_id = self._next_id
        self._next_id += 1
        track = Track(
            track_id=track_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            camera_id=camera_id or observation.camera_id,
            node=observation.node,
            state=TrackState.TENTATIVE,
            observation=observation,
            box=observation.box,
            image_height_px=observation.image_height_px,
            ground_xy=observation.ground_xy,
            ground_z=observation.ground_z,
            hits=1,
            misses=0,
            age=1,
            first_frame=self.frame_index,
            last_frame=self.frame_index,
            first_timestamp_s=self.last_timestamp_s,
            last_timestamp_s=self.last_timestamp_s,
            confidence=observation.confidence,
            confidence_ema=observation.confidence,
            embedding=observation.embedding,
        )
        if observation.embedding is not None:
            track.gallery.append(observation.embedding)
        self._tracks[track_id] = track
        return track

    def _prune(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.state is TrackState.DEAD]
        for track_id in dead:
            del self._tracks[track_id]

    def _collect(self, live: Sequence[Track], result: FrameTracks) -> None:
        """Publish this frame's tracks.

        Everything alive is published, including tracks that matched nothing and
        are being carried on prediction alone. That is standard MOT tracker
        behaviour and it is what makes a miss count as a miss rather than being
        quietly hidden - suppressing unmatched tracks would inflate every metric.
        """
        alive = {t.track_id for t in live}
        for track in self._tracks.values():
            if track.state is TrackState.DEAD:
                continue
            if track.hits >= self.min_hits or track.track_id in alive:
                result.tracks.append(track)
        result.tracks.sort(key=lambda t: t.track_id)

    def _age(self, result: FrameTracks) -> None:
        for track in result.tracks:
            track.age = max(1, self.frame_index - track.first_frame + 1)

    # -- introspection ------------------------------------------------------ #

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    @property
    def active_tracks(self) -> list[Track]:
        return [
            t for t in self._tracks.values()
            if t.state in {TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST}
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.frame_index = 0
        self.last_timestamp_s = 0.0

    def __len__(self) -> int:
        return len(self.active_tracks)


def _replace_box(observation: TrackObservation | None, box: BBox) -> TrackObservation | None:
    """Return a copy of an observation with a different box.

    The predicted box must not overwrite the last *measured* box - the rule layer
    reads ``track.observation`` to evaluate the current detection, not the
    extrapolation.
    """
    if observation is None:
        return None
    updated = TrackObservation(
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        box=box,
            confidence=observation.confidence,
            class_id=observation.class_id,
            class_name=observation.class_name,
            camera_id=observation.camera_id,
            node=observation.node,
            ground_xy=observation.ground_xy,
            ground_z=observation.ground_z,
            embedding=observation.embedding,
            velocity_xy=observation.velocity_xy,
            track_id=observation.track_id,
            image_height_px=observation.image_height_px,
        )
    return updated


def box_arrays(tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> tuple[np.ndarray, np.ndarray]:
    track_boxes = np.asarray([t.box for t in tracks], dtype=float).reshape(-1, 4)
    obs_boxes = np.asarray([o.box for o in observations], dtype=float).reshape(-1, 4)
    return track_boxes, obs_boxes


def predicted_boxes(tracks: Sequence[Track], *, damping: float = 1.0) -> np.ndarray:
    """``(N, 4)`` of predicted track boxes. What motion-gated association compares against."""
    if not tracks:
        return np.empty((0, 4), dtype=float)
    return np.asarray([t.predicted_box(damping=damping) for t in tracks], dtype=float).reshape(-1, 4)


def project_boxes(boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Project an ``(N, 4)`` array of xyxy boxes through a 3x3 homography.

    Used to carry predicted track boxes across a camera move. All four corners go
    through the transform, which is divided out by the homogeneous coordinate so a
    translation and a mild perspective change both land where they should.

    The result takes the projected top-left and bottom-right corners as the new
    box. For the translation and small-rotation cases this project actually sees —
    a PTZ settling after a move, a mast swaying — that is exact; under a large
    rotation the axis-aligned box is an approximation, which is the same
    approximation the BoT-SORT paper's camera-motion correction makes.
    """
    matrix = np.asarray(transform, dtype=float)
    if boxes.size == 0:
        return np.asarray(boxes, dtype=float).reshape(0, 4)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack(
        [
            np.stack([x1, y1], axis=-1),
            np.stack([x2, y1], axis=-1),
            np.stack([x2, y2], axis=-1),
            np.stack([x1, y2], axis=-1),
        ],
        axis=1,
    ).reshape(-1, 2)

    homogeneous = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous @ matrix.T
        projected = projected[:, :2] / projected[:, 2:3]
    projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    projected = projected.reshape(-1, 4, 2)
    return np.concatenate([projected[:, 0], projected[:, 2]], axis=1)


def iou_cost(
    tracks: Sequence[Track],
    observations: Sequence[TrackObservation],
    track_boxes: np.ndarray | None = None,
) -> np.ndarray:
    """``1 - IoU`` cost matrix, comparing **predicted** track boxes to detections.

    Motion-gated association: the gate asks "would the track be here if it kept
    moving at its measured velocity", which is what lets a target that has moved
    more than its own box width still match.

    ``track_boxes`` supplies already-adjusted boxes (for example after camera
    motion compensation), overriding the prediction.
    """
    if not tracks or not observations:
        return np.zeros((len(tracks), len(observations)), dtype=float)
    obs_boxes = np.asarray([o.box for o in observations], dtype=float).reshape(-1, 4)
    boxes = predicted_boxes(tracks) if track_boxes is None else np.asarray(track_boxes, float)
    return 1.0 - iou_matrix(boxes, obs_boxes)


def build_tracker(
    name: TrackerName | str,
    *,
    high_threshold: float = 0.6,
    low_threshold: float = 0.1,
    match_threshold: float = 0.8,
    max_age: int = 30,
    min_hits: int = 3,
    class_id: int = 0,
    **kwargs: Any,
) -> BaseTracker:
    """Factory used by the replayer, the API and the CLI."""
    from .botsort import BotSortTracker
    from .bytetrack import ByteTrackTracker
    from .sort import SortTracker

    registry = {
        "sort": SortTracker,
        "bytetrack": ByteTrackTracker,
        "botsort": BotSortTracker,
    }
    key = name.value if isinstance(name, TrackerName) else str(name).lower()
    if key == "nvdcf":
        raise ValueError(
            "nvdcf is a DeepStream-only tracker and cannot run on this platform. "
            "It is provided as a config artifact at "
            "src/anti_uav/deploy/deepstream/nvtracker_nvdcf.cfg. For local evaluation "
            "use botsort, which is the closest Python equivalent (motion + appearance)."
        )
    cls = registry.get(key)
    if cls is None:
        raise KeyError(f"unknown tracker {name!r}; available: {sorted(registry)}")

    return cls(
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        match_threshold=match_threshold,
        max_age=max_age,
        min_hits=min_hits,
        class_id=class_id,
        **kwargs,
    )
