"""ByteTrack.

ByteTrack's contribution to this project is a single idea, and it happens to be
exactly the one this data needs: **a detection that failed the confidence
threshold is usually still a real target, and it is disproportionately a hard one.**

In a sky-and-clutter scene, the frames where a drone is about to be lost - passing
behind a pylon, motion-blurred, or simply very small - are the frames where its
detection confidence dips. A tracker that throws those detections away loses the
target precisely when it needs help most. ByteTrack keeps them in a second
association round instead.

That matters more here than in MOT17 because of the class mix: the drone-vs-bird
datasets contain many genuine birds, and the frames where a bird is half-occluded
also look like low-confidence frames. ByteTrack's second pass is therefore a
double-edged sword on this data - it rescues drones, and it also gives birds a
second chance to be tracked. That trade-off is measurable, which is why all three
trackers are evaluated rather than just the best one.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", ECCV 2022.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...utils.geometry import BBox, linear_assignment
from ...utils.logging import get_logger
from ..types import FrameTracks, Track, TrackObservation, TrackState
from .base import BaseTracker, iou_cost

log = get_logger(__name__)


class ByteTrackTracker(BaseTracker):
    """Two-pass association: high confidence, then low confidence."""

    name = "bytetrack"
    uses_low_confidence = True

    def __init__(self, *args: Any, low_match_scale: float = 0.5, **kwargs: Any) -> None:
        """
        ``low_match_scale`` tightens the IoU gate for the second pass. Low
        confidence detections are noisier, so they should have to be a *better*
        geometric match to be accepted, not merely a similar one. 0.5 means the
        second pass requires roughly twice the IoU of the first.
        """
        super().__init__(*args, **kwargs)
        self.low_match_scale = low_match_scale

    def _split(
        self, detections: Sequence[TrackObservation]
    ) -> tuple[list[TrackObservation], list[TrackObservation]]:
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]
        return high, low

    def _cost(self, tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
        """ByteTrack's first pass is plain motion-gated IoU. No appearance term."""
        return iou_cost(tracks, observations, self.association_boxes(tracks))

    def _second_pass(
        self,
        live: Sequence[Track],
        high: Sequence[TrackObservation],
        low: Sequence[TrackObservation],
        result: FrameTracks,
        camera_id: str,
    ) -> None:
        """Associate leftover tracks with leftover *low* confidence detections.

        Only tracks that already matched something, or that are confirmed, take
        part. A brand-new tentative track should not be rescued by a weak
        detection - otherwise noise gets promoted to a track.
        """
        del high
        if not live or not low:
            return

        candidates = [t for t in live if t.hits > 1 or t.state is TrackState.CONFIRMED]
        if not candidates:
            return

        cost = iou_cost(candidates, low, self.association_boxes(candidates))
        threshold = self.match_threshold * self.low_match_scale
        pairs = linear_assignment(cost, max_cost=threshold)

        for track_idx, det_idx, _cost in pairs:
            track = candidates[track_idx]
            track.update(
                low[det_idx], smoothing=0.5, camera_displacement=self._camera_displacement()
            )
            if track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED
            # A rescued track is still a match: remove it from the missed list.
            if track.track_id in result.predicted_only:
                result.predicted_only.remove(track.track_id)
            track.misses = 0

    def update(
        self,
        observations: Sequence[TrackObservation],
        *,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
        camera_id: str = "",
        global_motion: np.ndarray | None = None,
    ) -> FrameTracks:
        """Same three steps as the base, with the low-confidence pass spliced in
        between association and spawn so a weak detection never creates a track on
        its own."""
        self.frame_index = frame_index if frame_index is not None else self.frame_index + 1
        self.last_timestamp_s = (
            timestamp_s if timestamp_s is not None else self.last_timestamp_s + 1.0 / 30.0
        )
        self.global_motion = global_motion

        # NB: no extrapolation written back onto track.observation. See
        # anti_uav.tracking.types.Track.predicted_box for why that compounds.

        detections = [o for o in observations if o.class_id == self.class_id]
        high, low = self._split(detections)

        result = FrameTracks(
            frame_index=self.frame_index,
            timestamp_s=self.last_timestamp_s,
            camera_id=camera_id,
        )

        live = [
            t for t in self._tracks.values()
            if t.is_alive(self.frame_index, max_age=self.max_age, max_gap=self.max_age)
        ]
        if not live:
            self._tracks.clear()
            for detection in high:
                self._spawn(detection, camera_id)
            self._prune()
            self._collect(list(self._tracks.values()), result)
            self._age(result)
            return result

        matched, unmatched_tracks, unmatched_dets = self._associate(live, high)

        for track_idx, det_idx, _cost in matched:
            self._match(live[track_idx], high[det_idx])

        for track_idx in unmatched_tracks:
            track = live[track_idx]
            track.misses += 1
            if track.misses > self.max_age:
                track.state = TrackState.DEAD
            elif track.state is TrackState.CONFIRMED:
                track.state = TrackState.LOST
            result.predicted_only.append(track.track_id)

        # Stage 2: leftover tracks meet leftover low-confidence detections.
        self._second_pass(live, high, low, result, camera_id)

        # Stage 3: genuinely new targets. Only HIGH-confidence detections may
        # create a track.
        for det_idx in unmatched_dets:
            detection = high[det_idx]
            self._spawn(detection, camera_id)

        self._prune()
        self._collect(live, result)
        self._age(result)
        return result

    def describe(self) -> str:
        return (
            f"ByteTrack(tracks={len(self.active_tracks)}, hi={self.high_threshold}, "
            f"lo={self.low_threshold}, low_match_scale={self.low_match_scale}, "
            f"max_age={self.max_age})"
        )


def _predicted(track: Track) -> TrackObservation | None:
    """Deprecated shim. Prediction is now derived on demand - see ``Track.predicted_box``."""
    from .base import _replace_box

    return _replace_box(track.observation, track.predicted_box())


def unmatched_summary(result: FrameTracks) -> dict[str, int]:
    """Diagnostics for a ByteTrack run: how often each stage did the work."""
    return {
        "tracks": len(result.tracks),
        "predicted_only": len(result.predicted_only),
        "unassociated": len(result.unassociated),
    }


def boxes_from_iou_pairs(
    pairs: Sequence[tuple[int, int, float]],
    tracks: Sequence[Track],
    observations: Sequence[TrackObservation],
) -> list[tuple[BBox, BBox]]:
    """Resolve index pairs to box pairs. Used by the tests."""
    return [(tracks[t].box, observations[d].box) for t, d, _c in pairs]


def cost_matrix(tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
    return iou_cost(tracks, observations)
