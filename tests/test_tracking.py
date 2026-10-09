"""Trackers, association, and the metric definitions.

A tracker test that only asserts "it ran" is worthless. These use a scripted
scenario with an exactly known answer: one target crossing the frame, and
identity switches that are deliberately injected and then counted.
"""

from __future__ import annotations

import types
from typing import ClassVar

import numpy as np
import pytest

from anti_uav.data import frameindex
from anti_uav.data.frameindex import FrameRecord
from anti_uav.tracking import replayer
from anti_uav.tracking.local import build_tracker
from anti_uav.tracking.metrics import FrameGT, FramePred, evaluate_sequence
from anti_uav.tracking.types import TrackObservation, TrackState

TRACKERS = ["sort", "bytetrack", "botsort"]
FPS = 25.0

#: Scenario for the replay detection cache: one target drifting right, plus one
#: weak distractor appearing partway through.
CACHE_FRAMES = 12
CACHE_STEP = 6.0


def observation(
    frame_index: int,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
    class_id: int = 0,
) -> TrackObservation:
    return TrackObservation(
        frame_index=frame_index,
        timestamp_s=frame_index / FPS,
        box=box,
        confidence=confidence,
        class_id=class_id,
        class_name="drone" if class_id == 0 else "bird",
        image_height_px=1080,
    )


def moving_boxes(n: int, *, step: float = 12.0, jitter: float = 0.0) -> list[tuple[float, float, float, float]]:
    """A single target drifting right at ``step`` px/frame."""
    rng = np.random.default_rng(11)
    out = []
    for f in range(n):
        x = 200.0 + step * f
        y = 400.0 + float(rng.normal(0, jitter))
        out.append((x, y, x + 24.0, y + 18.0))
    return out


class TestSingleTarget:
    @pytest.mark.parametrize("name", TRACKERS)
    def test_one_target_yields_one_confirmed_track(self, name: str) -> None:
        tracker = build_tracker(name)
        boxes = moving_boxes(20)
        result = None
        for f, box in enumerate(boxes):
            result = tracker.update([observation(f, box)], frame_index=f, timestamp_s=f / FPS)
        assert result is not None
        assert len(result.tracks) == 1
        assert result.tracks[0].state is TrackState.CONFIRMED

    @pytest.mark.parametrize("name", TRACKERS)
    def test_track_id_is_stable(self, name: str) -> None:
        tracker = build_tracker(name)
        ids = set()
        for f, box in enumerate(moving_boxes(20)):
            frame = tracker.update([observation(f, box)], frame_index=f, timestamp_s=f / FPS)
            ids.update(t.track_id for t in frame.tracks)
        assert len(ids) == 1, f"{name} invented {len(ids)} identities: {ids}"

    @pytest.mark.parametrize("name", TRACKERS)
    def test_velocity_tracks_the_motion(self, name: str) -> None:
        tracker = build_tracker(name)
        step = 12.0
        for f, box in enumerate(moving_boxes(15)):
            tracker.update([observation(f, box)], frame_index=f, timestamp_s=f / FPS)
        assert any(abs(t.velocity_xy[0] - step) < 3.0 for t in tracker.tracks)


class TestBirdSuppression:
    @pytest.mark.parametrize("name", ["bytetrack", "botsort"])
    def test_class_one_does_not_hold_a_drone_track(self, name: str) -> None:
        """A bird seen alone must never become a confirmed drone track."""
        tracker = build_tracker(name, class_id=0)
        confirmed = False
        for f, box in enumerate(moving_boxes(25)):
            frame = tracker.update(
                [observation(f, box, class_id=1)], frame_index=f, timestamp_s=f / FPS
            )
            confirmed |= any(t.state is TrackState.CONFIRMED for t in frame.tracks)
        assert confirmed is False


class TestLowConfidenceSecondPass:
    """ByteTrack's second pass must actually run.

    It used to be unreachable: ``update`` discarded the low-confidence split and
    handed ``high`` to ``_second_pass``, which re-split it. Every element of
    ``high`` is already at or above ``high_threshold``, so the re-derived low list
    was always empty and the pass returned immediately. ByteTrack was silently
    single-pass, and BoT-SORT inherited the same dead path.
    """

    #: A tracker configured so 0.90 is "high" and 0.20 is "low".
    CONF: ClassVar[dict[str, float]] = {
        "high_threshold": 0.60,
        "low_threshold": 0.10,
        "match_threshold": 0.75,
    }

    def _confirm_one(self, name: str, frames: int = 4) -> object:
        """Drive a single confirmed track drifting right, return the tracker."""
        tracker = build_tracker(name, **self.CONF)
        for f, box in enumerate(moving_boxes(frames, step=2.0)):
            tracker.update([observation(f, box)], frame_index=f, timestamp_s=f / FPS)
        assert any(t.state is TrackState.CONFIRMED for t in tracker.tracks)
        return tracker

    @pytest.mark.parametrize("name", ["bytetrack", "botsort"])
    def test_a_low_confidence_detection_rescues_a_missed_track(self, name: str) -> None:
        tracker = self._confirm_one(name)
        before = next(iter(tracker.tracks))
        hits_before = before.hits

        # Same motion, but the detection has fallen below the high threshold.
        nxt = len(moving_boxes(5, step=2.0))
        frame = tracker.update(
            [observation(nxt - 1, moving_boxes(nxt, step=2.0)[-1], confidence=0.20)],
            frame_index=nxt - 1,
            timestamp_s=(nxt - 1) / FPS,
        )

        track = next(iter(tracker.tracks))
        assert track.misses == 0, f"{name} let the track age out instead of rescuing it"
        assert track.hits == hits_before + 1, f"{name} did not absorb the weak detection"
        assert before.track_id not in frame.predicted_only, (
            f"{name} still reported a rescued track as prediction-only"
        )

    @pytest.mark.parametrize("name", ["bytetrack", "botsort"])
    def test_a_weak_detection_alone_never_creates_a_track(self, name: str) -> None:
        """The second pass rescues tracks. It must not promote noise."""
        frames = 4
        tracker = self._confirm_one(name, frames)
        count = len(tracker.tracks)

        # The next frame, so the confirmed track is still alive. Far away from it
        # and weak, so there is no geometric match for the second pass to use.
        tracker.update(
            [observation(frames, (900.0, 900.0, 924.0, 918.0), confidence=0.20)],
            frame_index=frames,
            timestamp_s=frames / FPS,
        )
        assert len(tracker.tracks) == count, f"{name} spawned a track from a weak detection"


class TestReplayDetectionCache:
    """``track-eval`` must not re-run inference once per tracker.

    Detection used to happen inside the per-tracker loop, so a three-tracker
    sweep paid for three identical inference passes and any inter-run
    nondeterminism showed up as a tracker difference. Detection is now computed
    once per sequence into frozen rows and replayed per tracker.
    """

    class _CountingDetector:
        """A single target drifting right, plus one weak distractor."""

        def __init__(self) -> None:
            self.calls = 0

        def predict_image(self, image: np.ndarray) -> object:
            i = self.calls
            self.calls += 1
            dets = []
            if i >= 1:
                x = 100.0 + CACHE_STEP * i
                dets.append(self._det(x, 0.9))
            if i >= 6:
                x = 100.0 + CACHE_STEP * i
                dets.append(self._det(x + 400.0, 0.2))
            return types.SimpleNamespace(detections=dets, height=64, width=64)

        @classmethod
        def _det(cls, x: float, confidence: float) -> object:
            return types.SimpleNamespace(
                box=(x, 10.0, x + 20.0, 30.0),
                confidence=confidence,
                class_id=0,
                class_name="drone",
            )

    @pytest.fixture
    def dataset(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> dict[int, FrameGT]:
        """A fake converted dataset: real files on disk, stubbed index and decoder."""
        for i in range(CACHE_FRAMES):
            path = tmp_path / "frames" / "seq" / "rgb" / f"{i:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-a-real-jpeg")

        records = [
            FrameRecord(
                sequence_id="seq",
                frame_index=i,
                modality="rgb",
                dataset="fake",
                variant="v",
                image=f"frames/seq/rgb/{i:06d}.jpg",
                label="",
                width=64,
                height=64,
            )
            for i in range(CACHE_FRAMES)
        ]
        # detect_sequence imports load_index from the source module inside the
        # function body, so the patch has to land there rather than on replayer.
        monkeypatch.setattr(
            frameindex, "load_index", lambda dataset, variant: records, raising=False
        )
        monkeypatch.setattr(
            replayer, "interim_root", lambda dataset, variant: tmp_path, raising=False
        )
        monkeypatch.setattr(
            replayer,
            "read_image",
            lambda path: np.zeros((64, 64, 3), dtype=np.uint8),
            raising=False,
        )

        gt = {
            i: FrameGT(
                frame_index=i,
                objects=[(1, (100.0 + CACHE_STEP * i, 10.0, 120.0 + CACHE_STEP * i, 30.0), 1.0)],
            )
            for i in range(1, CACHE_FRAMES)
        }
        gt[0] = FrameGT(frame_index=0, objects=[])
        return gt

    def test_detection_runs_once_per_sequence(self, dataset: dict[int, FrameGT]) -> None:
        detector = self._CountingDetector()
        detections = replayer.detect_sequence(
            detector, "fake", "v", "seq", with_embeddings=False
        )
        assert detections.ok
        assert detections.n_frames == CACHE_FRAMES
        assert detector.calls == CACHE_FRAMES

    @pytest.mark.parametrize("name", TRACKERS)
    def test_replaying_from_cache_runs_no_further_inference(
        self, name: str, dataset: dict[int, FrameGT]
    ) -> None:
        detector = self._CountingDetector()
        detections = replayer.detect_sequence(
            detector, "fake", "v", "seq", with_embeddings=False
        )
        before = detector.calls
        result = replayer.replay_from_detections(
            detections, dataset, tracker_name=name, gt_source="synthetic", has_ids=True
        )
        assert result.ok
        assert detector.calls == before, "replay_from_detections re-ran the detector"

    @pytest.mark.parametrize("name", TRACKERS)
    def test_cached_scores_match_the_direct_path(
        self, name: str, dataset: dict[int, FrameGT]
    ) -> None:
        detector = self._CountingDetector()
        cached = replayer.replay_from_detections(
            replayer.detect_sequence(detector, "fake", "v", "seq", with_embeddings=False),
            dataset,
            tracker_name=name,
            gt_source="synthetic",
            has_ids=True,
        )

        # replay_sequence loads its own ground truth, so point it at the same one.
        original = replayer.load_ground_truth_for
        try:
            replayer.load_ground_truth_for = lambda *a, **k: (dataset, "synthetic", True)
            direct = replayer.replay_sequence(
                self._CountingDetector(),
                "fake",
                "v",
                "seq",
                tracker_name=name,
                with_embeddings=False,
            )
        finally:
            replayer.load_ground_truth_for = original

        assert direct.ok
        a, b = cached.metrics, direct.metrics
        assert a is not None and b is not None
        assert a.mota == pytest.approx(b.mota, abs=1e-12)
        assert a.idf1 == pytest.approx(b.idf1, abs=1e-12)
        assert a.hota == pytest.approx(b.hota, abs=1e-12)
        assert a.id_switches == b.id_switches
        assert a.gt_boxes == b.gt_boxes
        assert a.pred_boxes == b.pred_boxes

    def test_missing_ground_truth_errors_instead_of_scoring_zero(
        self, dataset: dict[int, FrameGT], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scored against nothing, every prediction is a false positive and MOTA
        reads 0 - which looks like a broken tracker rather than a broken run."""
        monkeypatch.setattr(
            replayer, "load_ground_truth_for", lambda *a, **k: ({}, "none", False)
        )
        result = replayer.replay_sequence(
            self._CountingDetector(), "fake", "v", "seq", with_embeddings=False
        )
        assert not result.ok
        assert "no ground truth" in result.error

    def test_detections_are_immutable(self, dataset: dict[int, FrameGT]) -> None:
        detections = replayer.detect_sequence(
            self._CountingDetector(), "fake", "v", "seq", with_embeddings=False
        )
        populated = next(f for f in detections.frames if f)
        with pytest.raises(AttributeError):
            populated[0].confidence = 0.123  # type: ignore[misc]

    def test_observations_are_rebuilt_per_tracker(self, dataset: dict[int, FrameGT]) -> None:
        """TrackObservation is mutable and a track holds a reference to it, so the
        cache must hand out fresh objects rather than shared ones."""
        detections = replayer.detect_sequence(
            self._CountingDetector(), "fake", "v", "seq", with_embeddings=False
        )
        populated = next(f for f in detections.frames if f)
        first = populated[0].as_observation(0, 0.0)
        second = populated[0].as_observation(0, 0.0)
        assert first is not second
        first.confidence = 0.0
        assert second.confidence != 0.0


class TestCameraMotionCompensation:
    """A PTZ pan must not be learned as target motion.

    A camera that moves the whole image hands the tracker a velocity that is
    actually the camera's, and a prediction that is wrong by the full pan distance
    on the very next frame. Both effects are what turn one camera move into a lost
    identity.
    """

    PAN_AT = 6
    PAN_PX = 200.0
    STEP = 4.0
    FRAMES = 16

    @staticmethod
    def _translation(dx: float) -> np.ndarray:
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def _run(self, name: str, *, compensate: bool) -> tuple[int, set[int]]:
        tracker = build_tracker(
            name,
            high_threshold=0.5,
            low_threshold=0.1,
            match_threshold=0.75,
            max_age=24,
            min_hits=3,
        )
        seen: set[int] = set()
        for i in range(self.FRAMES):
            x = 400.0 + self.STEP * i
            shifted = x + (self.PAN_PX if i >= self.PAN_AT else 0.0)
            transform = (
                (self._translation(self.PAN_PX) if i == self.PAN_AT else self._translation(0.0))
                if compensate
                else None
            )
            frame = tracker.update(
                [observation(i, (shifted, 300.0, shifted + 20.0, 316.0))],
                frame_index=i,
                timestamp_s=i / FPS,
                global_motion=transform,
            )
            seen.update(t.track_id for t in frame.tracks)
        return len(seen), seen

    @pytest.mark.parametrize("name", TRACKERS)
    def test_a_pan_shatters_a_track_without_compensation(self, name: str) -> None:
        """Baseline: without compensation this scenario loses the identity."""
        count, _ids = self._run(name, compensate=False)
        assert count > 1, f"{name} absorbed the pan by luck; the test is not exercising it"

    @pytest.mark.parametrize("name", TRACKERS)
    def test_compensation_keeps_one_identity_across_the_pan(self, name: str) -> None:
        count, ids = self._run(name, compensate=True)
        assert count == 1, f"{name} invented {count} identities across the pan: {sorted(ids)}"

    @pytest.mark.parametrize("name", TRACKERS)
    def test_velocity_excludes_the_camera_move(self, name: str) -> None:
        """Velocity must stay the target's own motion, not the pan."""
        tracker = build_tracker(
            name,
            high_threshold=0.5,
            low_threshold=0.1,
            match_threshold=0.75,
            max_age=24,
            min_hits=3,
        )
        for i in range(self.FRAMES):
            x = 400.0 + self.STEP * i
            shifted = x + (self.PAN_PX if i >= self.PAN_AT else 0.0)
            tracker.update(
                [observation(i, (shifted, 300.0, shifted + 20.0, 316.0))],
                frame_index=i,
                timestamp_s=i / FPS,
                global_motion=self._translation(self.PAN_PX if i == self.PAN_AT else 0.0),
            )
        track = next(iter(tracker.tracks))
        assert track.velocity_xy[0] == pytest.approx(self.STEP, abs=0.5), track.velocity_xy

    def test_project_boxes_applies_the_homography(self) -> None:
        from anti_uav.tracking.local.base import project_boxes

        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        moved = project_boxes(boxes, self._translation(5.0))
        assert moved[0].tolist() == pytest.approx([15.0, 20.0, 35.0, 40.0])

    def test_project_boxes_identity_is_a_no_op(self) -> None:
        from anti_uav.tracking.local.base import project_boxes

        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        assert np.allclose(project_boxes(boxes, np.eye(3)), boxes)

    def test_project_boxes_handles_no_tracks(self) -> None:
        from anti_uav.tracking.local.base import project_boxes

        assert project_boxes(np.empty((0, 4)), self._translation(5.0)).shape == (0, 4)

    def test_measure_at_the_image_centre_not_a_box_corner(self) -> None:
        """A pure translation must not depend on where the target happens to be."""
        tracker = build_tracker("botsort", min_hits=1)
        tracker._motion_centre = (640.0, 360.0)
        tracker.global_motion = self._translation(37.0)
        assert tracker._camera_displacement() == pytest.approx((37.0, 0.0))
        tracker.global_motion = self._translation(-12.0)
        assert tracker._camera_displacement() == pytest.approx((-12.0, 0.0))

    def test_degenerate_transform_does_not_produce_nan(self) -> None:
        tracker = build_tracker("botsort")
        tracker.global_motion = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        dx, dy = tracker._camera_displacement()
        assert np.isfinite(dx) and np.isfinite(dy)


class TestMetrics:
    def test_perfect_prediction_scores_one(self) -> None:
        boxes = moving_boxes(30)
        gt = {
            f: FrameGT(frame_index=f, objects=[(1, b, 1.0)])
            for f, b in enumerate(boxes)
        }
        pred = {
            f: FramePred(f, objects=[(1, b)])
            for f, b in enumerate(boxes)
        }
        result = evaluate_sequence("perfect", gt, pred)
        assert result.mota == pytest.approx(1.0, abs=1e-6)
        assert result.id_switches == 0
        assert result.idf1 == pytest.approx(1.0, abs=1e-6)

    def test_identity_switch_is_counted(self) -> None:
        """Two ids across one sequence is one switch, not two tracks' worth."""
        boxes = moving_boxes(30)
        gt = {f: FrameGT(frame_index=f, objects=[(1, b, 1.0)]) for f, b in enumerate(boxes)}
        pred = {
            f: FramePred(f, objects=[(1 if f < 15 else 2, b)])
            for f, b in enumerate(boxes)
        }
        result = evaluate_sequence("switch", gt, pred)
        assert result.id_switches == 1

    def test_missed_detections_reduce_mota(self) -> None:
        boxes = moving_boxes(30)
        gt = {f: FrameGT(frame_index=f, objects=[(1, b, 1.0)]) for f, b in enumerate(boxes)}
        pred = {
            f: FramePred(f, objects=[(1, b)]) for f, b in enumerate(boxes) if f % 2 == 0
        }
        result = evaluate_sequence("gaps", gt, pred)
        assert 0.0 < result.mota < 1.0

    def test_false_positives_reduce_mota(self) -> None:
        boxes = moving_boxes(20)
        gt = {f: FrameGT(frame_index=f, objects=[(1, b, 1.0)]) for f, b in enumerate(boxes)}
        pred = {
            f: FramePred(
                f,
                objects=[(1, b)] + ([(2, (10.0, 10.0, 40.0, 40.0))] if f < 5 else []),
            )
            for f, b in enumerate(boxes)
        }
        result = evaluate_sequence("fp", gt, pred)
        assert result.mota < 1.0

    def test_empty_prediction_is_zero_not_an_error(self) -> None:
        gt = {0: FrameGT(frame_index=0, objects=[(1, (10.0, 10.0, 30.0, 30.0), 1.0)])}
        result = evaluate_sequence("empty", gt, {0: FramePred(0, objects=[])})
        assert result.mota == 0.0

    def test_hota_is_bounded(self) -> None:
        boxes = moving_boxes(15)
        gt = {f: FrameGT(frame_index=f, objects=[(1, b, 1.0)]) for f, b in enumerate(boxes)}
        pred = {f: FramePred(f, objects=[(1, b)]) for f, b in enumerate(boxes)}
        result = evaluate_sequence("hota", gt, pred)
        assert 0.0 <= result.hota <= 1.0


class TestKalman:
    """The metric-space filter behind the global tracker.

    It works in metres, not pixels, so it is exercised on a straight-line
    constant-velocity track where the forecast has a known answer.
    """

    def test_predict_leads_a_constant_velocity_target(self) -> None:
        from anti_uav.tracking.kalman import KalmanTrack2D

        kf = KalmanTrack2D()
        dt = 0.04  # 25 fps
        speed = 12.0  # m/s
        # 12 m/s at 25 fps is a 0.48 m step, far larger than the filter's initial
        # velocity variance, so it needs a couple of seconds to lock on. Run long
        # enough that we are testing convergence, not warm-up.
        steps = 75
        for step in range(steps):
            kf.predict(dt)
            kf.update((speed * dt * step, 0.0))

        assert kf.speed_m_s > 0.8 * speed, f"converged to {kf.speed_m_s:.2f}, want ~{speed}"
        x, y = kf.predict(dt)
        assert y == pytest.approx(0.0, abs=0.5), "a due-east track must not drift north"
        assert x > speed * dt * steps * 0.8

    def test_converges_to_the_measured_position(self) -> None:
        from anti_uav.tracking.kalman import KalmanTrack2D

        kf = KalmanTrack2D()
        kf.initialise(position=(500.0, 250.0), velocity=(3.0, -1.0))
        for step in range(40):
            kf.predict(0.04)
            kf.update((500.0 + 3.0 * 0.04 * step, 250.0 - 0.04 * step))
        x, y = kf.position
        assert abs(x - (500.0 + 3.0 * 0.04 * 39)) < 1.0
        assert abs(y - (250.0 - 0.04 * 39)) < 1.0

    def test_uncertainty_grows_without_measurements(self) -> None:
        from anti_uav.tracking.kalman import KalmanTrack2D

        kf = KalmanTrack2D()
        kf.initialise(position=(100.0, 100.0))
        kf.predict(0.04)
        kf.update((100.0, 100.0))
        before = kf.uncertainty_m
        for _ in range(5):
            kf.predict(0.04)
        assert kf.uncertainty_m > before
