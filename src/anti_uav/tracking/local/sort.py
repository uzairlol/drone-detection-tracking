"""SORT: Simple Online and Realtime Tracking.

The baseline everything else is measured against. IoU-only association, a constant
-velocity filter in pixel space, no notion of detection confidence beyond a single
threshold.

Worth keeping for three reasons:

* it is the reference point in the MOT literature, so a number from this tracker
  is comparable to published work;
* it is fast enough to be obviously correct, which makes it the right thing to
  debug against when a fancier tracker misbehaves;
* its failures are informative. On this project's data SORT's dominant error mode
  is a drone crossing a bird and swapping identity, because IoU cannot tell two
  similar-looking objects apart. That is precisely the gap ByteTrack and BoT-SORT
  close, so having SORT in the comparison makes their contribution legible
  instead of just "higher IDF1".

The block diagram's edge budget also matters here: this is the tracker a Jetson
Thor can afford to run alongside 13 batched streams, so it is the honest floor for
"what can we deploy".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...utils.geometry import iou_matrix
from ..types import Track, TrackObservation
from .base import BaseTracker


class SortTracker(BaseTracker):
    """IoU-gated association with a constant-velocity predictor."""

    name = "sort"
    uses_low_confidence = False

    def _cost(self, tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
        """IoU between the *predicted* track box and each detection.

        This is SORT's defining term: association is gated on where the target
        would be if it kept its current velocity, not on where it was last seen.
        ``association_boxes`` is what carries any camera-motion correction.
        """
        if not tracks or not observations:
            return np.zeros((len(tracks), len(observations)), dtype=float)

        track_boxes = self.association_boxes(tracks).reshape(-1, 4)
        obs_boxes = np.asarray([o.box for o in observations], dtype=float).reshape(-1, 4)
        return 1.0 - iou_matrix(track_boxes, obs_boxes)

    def describe(self) -> str:
        return (
            f"SORT(tracks={len(self.active_tracks)}, hi={self.high_threshold}, "
            f"iou_thr={1.0 - self.match_threshold:.2f}, max_age={self.max_age})"
        )


class IoUOnlyTracker(SortTracker):
    """Association on the last *measured* box rather than the predicted one.

    Kept because it isolates the contribution of motion prediction: comparing it
    against :class:`SortTracker` on the same data shows how much the constant
    -velocity term is actually buying for targets as small as these. On a 15 px
    target moving fast, the answer is often "less than expected".
    """

    def _cost(self, tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
        if not tracks or not observations:
            return np.zeros((len(tracks), len(observations)), dtype=float)
        track_boxes = np.asarray(
            [t.observation.box if t.observation else t.box for t in tracks], dtype=float
        ).reshape(-1, 4)
        obs_boxes = np.asarray([o.box for o in observations], dtype=float).reshape(-1, 4)
        return 1.0 - iou_matrix(track_boxes, obs_boxes)


def predict_next(
    box: tuple[float, float, float, float],
    previous: tuple[float, float, float, float] | None,
    *,
    damping: float = 1.0,
) -> tuple[float, float, float, float]:
    """Constant-velocity extrapolation of a box.

    ``damping`` scales the velocity term. Values below 1.0 trade lag for
    stability, which is the right call for a 12 px target whose bounding box
    jitters by more than its own width between frames - extrapolating the full
    measured velocity amplifies that jitter instead of cancelling it.
    """
    if previous is None:
        return box
    vx = (box[0] - previous[0]) * damping
    vy = (box[1] - previous[1]) * damping
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (box[0] + vx, box[1] + vy, box[0] + vx + width, box[1] + vy + height)


def iou_thresholds_for(sizes_px: Sequence[float]) -> np.ndarray:
    """A per-observation IoU threshold scaled by target size.

    A 0.3 IoU threshold that is generous for a 200 px target is meaningless for a
    15 px one, where a two-pixel position error already costs a third of the box.
    This is a common and silent reason a tracker tuned on one dataset underperforms
    on another, so the size dependence is explicit rather than hidden in a constant.
    """
    thresholds = []
    for size in sizes_px:
        if size >= 64:
            thresholds.append(0.30)
        elif size >= 32:
            thresholds.append(0.25)
        elif size >= 16:
            thresholds.append(0.20)
        else:
            thresholds.append(0.15)
    return np.asarray(thresholds, dtype=float)
