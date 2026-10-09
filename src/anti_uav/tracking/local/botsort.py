"""BoT-SORT: appearance-aware association.

The block diagram's default local tracker. Its addition over ByteTrack is a
second association gate that fuses motion with **appearance**, using a ReID
embedding per detection.

Why appearance is not optional on this data
--------------------------------------------
The failure mode IoU-only association cannot fix is a **crossing**: a drone and a
bird pass through the same image region in opposite directions, their boxes
overlap for several frames, and the tracker either swaps their identities or
drops one. Drone-vs-Bird was built specifically to make that confusion hard, and
with a median target of 28 px the boxes overlap heavily during the crossing.

It also fixes ID switches after an occlusion: a drone that reappears two metres
from where it was lost has near-zero IoU with its predicted position but a very
high cosine similarity to its gallery. Motion gating alone loses it; appearance
recovers it.

The cost here is that a ReID model has to run per detection, which is real GPU
time on a Jetson that is already running 13 batched streams. That trade-off is why
:mod:`anti_uav.deploy.deepstream` ships an NvDCF config (motion + shape features,
no ReID) *alongside* the BoT-SORT reference, rather than pretending one config
covers every budget.

Reference: Cao et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
arXiv 2206.14651.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...utils.geometry import linear_assignment
from ...utils.logging import get_logger
from ..types import FrameTracks, Track, TrackObservation
from .base import BaseTracker, box_arrays, iou_cost, predicted_boxes, project_boxes

log = get_logger(__name__)

_EPS = 1e-6


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, safe against zero vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < _EPS or norm_b < _EPS:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


def cosine_similarity_matrix(
    track_galleries: Sequence[np.ndarray | None],
    detections: Sequence[np.ndarray | None],
) -> np.ndarray:
    """``(n_tracks, n_detections)`` cosine similarity, ``1.0`` where unknown.

    A track or detection with no embedding gets a neutral score rather than zero.
    Treating "no appearance available" as "appearance disagrees" would make the
    gate actively harmful whenever the ReID model is unavailable - which is the
    normal case on a CPU dev box.
    """
    matrix = np.ones((len(track_galleries), len(detections)), dtype=float)
    for i, gallery in enumerate(track_galleries):
        if gallery is None:
            continue
        gallery_norm = gallery / max(float(np.linalg.norm(gallery)), _EPS)
        for j, embedding in enumerate(detections):
            if embedding is None:
                continue
            norm = embedding / max(float(np.linalg.norm(embedding)), _EPS)
            matrix[i, j] = float(np.dot(gallery_norm, norm))
    return matrix


def fuse_costs(
    motion_cost: np.ndarray,
    appearance_cost: np.ndarray,
    *,
    lambda_appearance: float = 0.25,
    appearance_available: bool = False,
) -> np.ndarray:
    """Blend the two gates.

    ``lambda_appearance`` is deliberately small (0.25). Appearance is a *tie-breaker*
    here, not a primary signal: on a 28 px target the ReID embedding is itself
    derived from very few pixels, and trusting it heavily would trade a reliable
    geometric match for a noisy appearance one.

    When no embeddings exist at all, the fused cost collapses to the motion cost
    exactly, so the tracker degrades to ByteTrack rather than to noise.
    """
    if not appearance_available:
        return motion_cost
    lambda_appearance = float(np.clip(lambda_appearance, 0.0, 1.0))
    return (1.0 - lambda_appearance) * motion_cost + lambda_appearance * appearance_cost


class BotSortTracker(BaseTracker):
    """Motion + appearance association, with a camera-motion compensation hook."""

    name = "botsort"
    uses_low_confidence = True

    def __init__(
        self,
        *args: Any,
        lambda_appearance: float = 0.25,
        reid_threshold: float = 0.55,
        low_match_scale: float = 0.5,
        with_reid: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        ``reid_threshold`` is the cosine floor below which an association is
        refused outright. Mirrors ``rules.tracking.reid_similarity_threshold``.

        ``with_reid=False`` runs the identical algorithm with the appearance term
        disabled. That is not a debug flag: it is the ablation that tells you
        whether the ReID model is earning its GPU time on *your* data.
        """
        super().__init__(*args, **kwargs)
        self.lambda_appearance = lambda_appearance
        self.reid_threshold = reid_threshold
        self.low_match_scale = low_match_scale
        self.with_reid = with_reid
        self.reid_matches = 0
        self.motion_only_matches = 0

    # -- cost --------------------------------------------------------------- #

    def _cost(self, tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> np.ndarray:
        if not tracks or not observations:
            return np.zeros((len(tracks), len(observations)), dtype=float)

        motion = iou_cost(tracks, observations, self.association_boxes(tracks))
        if not self.with_reid:
            return motion

        available = any(t.embedding is not None for t in tracks) and any(
            o.embedding is not None for o in observations
        )
        if not available:
            return motion

        similarity = cosine_similarity_matrix(
            [t.embedding for t in tracks], [o.embedding for o in observations]
        )
        # Cosine similarity in [-1, 1] -> a cost in [0, 2], then clamped to the
        # same scale as the IoU cost so the blend is meaningful.
        appearance_cost = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)
        return fuse_costs(
            motion,
            appearance_cost,
            lambda_appearance=self.lambda_appearance,
            appearance_available=True,
        )

    def _split(
        self, detections: Sequence[TrackObservation]
    ) -> tuple[list[TrackObservation], list[TrackObservation]]:
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]
        return high, low

    # -- association -------------------------------------------------------- #

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[TrackObservation],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = self._cost(tracks, detections)
        pairs = linear_assignment(cost, max_cost=self.match_threshold)

        # Second gate: refuse a geometric match whose appearance actively
        # contradicts it. Applied after the assignment so it costs nothing extra -
        # the solver already found the optimum, and this only removes pairs.
        if self.with_reid:
            accepted: list[tuple[int, int, float]] = []
            for track_idx, det_idx, pair_cost in pairs:
                track_embedding = tracks[track_idx].embedding
                det_embedding = detections[det_idx].embedding
                if track_embedding is None or det_embedding is None:
                    self.motion_only_matches += 1
                    accepted.append((track_idx, det_idx, pair_cost))
                    continue

                similarity = _cosine(track_embedding, det_embedding)
                if similarity >= self.reid_threshold:
                    self.reid_matches += 1
                    accepted.append((track_idx, det_idx, pair_cost))
                else:
                    # Appearance contradicts. The detection is left unmatched, so
                    # it starts its own track rather than stealing an identity -
                    # which is the whole point of the gate.
                    log.debug(
                        "reid gate rejected association",
                        extra={
                            "track": tracks[track_idx].track_id,
                            "similarity": round(similarity, 3),
                            "threshold": self.reid_threshold,
                        },
                    )
            pairs = accepted

        matched_track = {t for t, _d, _c in pairs}
        matched_det = {d for _t, d, _c in pairs}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
        return pairs, unmatched_tracks, unmatched_dets

    def _second_pass(
        self,
        live: Sequence[Track],
        high: Sequence[TrackObservation],
        low: Sequence[TrackObservation],
        result: FrameTracks,
        camera_id: str,
    ) -> None:
        """Low-confidence second pass, inheriting ByteTrack's second round."""
        from .bytetrack import ByteTrackTracker

        helper = ByteTrackTracker.__new__(ByteTrackTracker)
        helper.__dict__.update(self.__dict__)
        helper.low_match_scale = self.low_match_scale
        ByteTrackTracker._second_pass(helper, live, high, low, result, camera_id)

    # -- camera motion ------------------------------------------------------ #

    def compensate_camera_motion(
        self,
        transform: np.ndarray,
        tracks: Sequence[Track],
        observations: Sequence[TrackObservation],
    ) -> tuple[np.ndarray, list[TrackObservation]]:
        """Remove the common-mode image motion from one frame of association.

        A fixed overview camera on a windy mast, or a PTZ that just completed a
        move, translates the entire image between frames. Every track then appears
        to move at once, which a per-track motion model reads as a real manoeuvre
        and predicts badly — a 20 px target on a PTZ that just panned 200 px loses
        its identity in a single frame.

        ``transform`` is a 3x3 homography mapping *previous* image coordinates to
        *current* ones. The **tracks** are carried forward through it, because a
        track's box is expressed in the previous frame and has to be brought into
        the current one before IoU is meaningful. The detections are already in
        current coordinates and are returned unchanged.

        Returns the adjusted track boxes and the observations, ready for
        :meth:`_cost`. Passing the same transform to ``update(global_motion=...)``
        does this automatically; this method is for callers that want the boxes
        without running a full frame.
        """
        if transform is None:
            return predicted_boxes(tracks), list(observations)
        boxes = project_boxes(predicted_boxes(tracks), transform)
        return boxes, list(observations)

    def describe(self) -> str:
        return (
            f"BoT-SORT(tracks={len(self.active_tracks)}, hi={self.high_threshold}, "
            f"lo={self.low_threshold}, lambda_app={self.lambda_appearance}, "
            f"reid_thr={self.reid_threshold}, reid={'on' if self.with_reid else 'off'}, "
            f"matches: reid={self.reid_matches} motion={self.motion_only_matches})"
        )


def ablation_pair(**kwargs: Any) -> list[BaseTracker]:
    """The BoT-SORT with-reid / without-reid pair used by the ablation table."""
    return [
        BotSortTracker(with_reid=False, name="botsort_no_reid", **kwargs),  # type: ignore[arg-type]
        BotSortTracker(with_reid=True, name="botsort", **kwargs),
    ]


def box_grid(tracks: Sequence[Track], observations: Sequence[TrackObservation]) -> tuple[np.ndarray, np.ndarray]:
    return box_arrays(tracks, observations)
