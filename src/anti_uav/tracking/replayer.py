"""Offline replay: run detection + tracking over a dataset and score the result.

This is the harness that answers "which tracker is better on *this* data". It
runs the same three components in the same order the edge pipeline does:

    frames -> Detector.predict -> Tracker.update -> (optionally) GlobalTrackManager

with ground truth from the converted dataset's MOT files or the YOLO labels. The
important property is that it shares the association code path with the metrics
harness and the API, so a number produced here describes the tracker that would
actually run.

Ground-truth sources, in order of preference
--------------------------------------------
1. ``data/interim/<dataset>/<variant>/mot/<seq>/<modality>.txt`` - the publisher's
   own MOT file, passed through unchanged by the converter. MM-UAV's toolkit reads
   the same file, so scores are directly comparable to the published baseline.
2. the YOLO ``.txt`` labels - no track ids, so identity metrics are not
   available and only detection-side metrics can be computed. Detected and
   reported rather than silently skipping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.loader import load_registry, load_rules
from ..config.schema import TrackerName
from ..data.frameindex import interim_root
from ..utils.imaging import read_image
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import subdir
from .local import build_tracker
from .metrics import (
    EvalReport,
    FrameGT,
    FramePred,
    SequenceResult,
    evaluate_sequence,
)
from .types import FrameTracks, TrackObservation, TrackState

log = get_logger(__name__)

#: Detections below this confidence are still handed to the tracker.
DEFAULT_FLOOR = 0.05


@dataclass(slots=True)
class ReplayResult:
    """Everything one replay produced."""

    sequence: str
    tracker: str
    frames: int = 0
    detections: int = 0
    tracks_created: int = 0
    tracks_confirmed: int = 0
    metrics: SequenceResult | None = None
    per_frame: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    #: Where the MOT rows were written, when the caller asked for a dump. The
    #: CLI reports this back so the operator knows which file to open.
    dump_path: str | None = None

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass(frozen=True, slots=True)
class DetectedBox:
    """One detection, as read off the frame.

    Frozen and free of any tracker state, so the same instance can be handed to
    every tracker in a comparison without one of them being able to affect the
    others. :meth:`as_observation` builds a fresh :class:`TrackObservation` per
    tracker for exactly that reason — ``TrackObservation`` is mutable and a
    tracker stores a reference to it.
    """

    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    image_height_px: int
    embedding: tuple[float, ...] | None = None

    def as_observation(self, frame_index: int, timestamp_s: float) -> TrackObservation:
        return TrackObservation(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            box=self.box,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            image_height_px=self.image_height_px,
            embedding=None if self.embedding is None else np.asarray(self.embedding, dtype=np.float32),
        )


@dataclass(slots=True)
class SequenceDetections:
    """Every frame of one sequence, detected once.

    This exists so a multi-tracker comparison pays for inference once per
    sequence instead of once per tracker. That is roughly a 3x saving on the
    default three-tracker sweep, and it is also strictly more correct: any
    difference between the tracker rows is then attributable to the tracker
    alone, rather than to detector nondeterminism.
    """

    dataset: str
    variant: str
    sequence_id: str
    modality: str
    #: One entry per replayed frame, each already filtered by the confidence floor.
    frames: list[tuple[DetectedBox, ...]] = field(default_factory=list)
    #: Per-frame detection counts *before* the floor, for the replay dump.
    raw_counts: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_detections(self) -> int:
        return sum(len(f) for f in self.frames)


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #


def load_mot_ground_truth(
    dataset: str,
    variant: str,
    sequence_id: str,
    modality: str = "rgb",
) -> tuple[dict[int, FrameGT], bool]:
    """Load a converted MOT file. Returns ``(ground_truth, has_track_ids)``."""
    path = interim_root(dataset, variant) / "mot" / sequence_id / f"{modality}.txt"
    if not path.is_file():
        return {}, False

    gt: dict[int, FrameGT] = {}
    has_ids = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = [p for p in line.strip().replace(" ", ",").split(",") if p]
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1, y1, x2, y2 = (float(v) for v in parts[2:6])
            except ValueError:
                continue
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            # parts[6] (score) and parts[7] (object class) are deliberately not
            # read: GT scores are all 1 by definition, and a GT class of `0`
            # (Anti-UAV) versus `1` (MOT) is a dataset convention, not a signal.
            gt.setdefault(frame, FrameGT(frame_index=frame)).objects.append(
                (track_id, (x1, y1, x2, y2), visibility)
            )
            has_ids = has_ids or track_id > 1

    return gt, has_ids


def load_yolo_ground_truth(
    dataset: str,
    variant: str,
    records: Sequence[Any],
    unified_labels: Sequence[str] = ("drone", "bird"),
) -> dict[int, FrameGT]:
    """Derive per-frame ground truth from YOLO labels, with a single id per frame.

    Without publisher track ids every detection is treated as id 1. That is enough
    to measure MOTA/HOTA against single-target sequences (Anti-UAV's case) and
    useless for identity metrics on multi-object sequences, so the caller is told.
    """
    from ..data.harmonize import read_yolo_label

    root = interim_root(dataset, variant)
    drone_index = unified_labels.index("drone") if "drone" in unified_labels else 0

    gt: dict[int, FrameGT] = {}
    for record in records:
        label = root / record.label
        if not label.is_file():
            continue
        boxes = read_yolo_label(
            label,
            width=record.width,
            height=record.height,
            strict=False,
        )
        frame_gt = FrameGT(frame_index=record.frame_index)
        for class_id, cx, cy, bw, bh in boxes:
            if class_id != drone_index:
                continue
            x1 = (cx - bw / 2.0) * record.width
            y1 = (cy - bh / 2.0) * record.height
            x2 = (cx + bw / 2.0) * record.width
            y2 = (cy + bh / 2.0) * record.height
            frame_gt.objects.append((1, (x1, y1, x2, y2), 1.0))
        gt[record.frame_index] = frame_gt
    return gt


def load_ground_truth_for(
    dataset: str,
    variant: str,
    sequence_id: str,
    modality: str = "rgb",
    *,
    records: Sequence[Any] = (),
) -> tuple[dict[int, FrameGT], str, bool]:
    """``(ground_truth, source, has_track_ids)`` - prefers the publisher's MOT file."""
    gt, has_ids = load_mot_ground_truth(dataset, variant, sequence_id, modality)
    if gt:
        return gt, "mot", has_ids

    if records:
        return (
            load_yolo_ground_truth(dataset, variant, records),
            "yolo (no track ids)",
            False,
        )

    return {}, "none", False


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def detect_sequence(
    detector: Any,
    dataset: str,
    variant: str,
    sequence_id: str,
    *,
    modality: str = "rgb",
    conf_floor: float = DEFAULT_FLOOR,
    max_frames: int | None = None,
    with_embeddings: bool = True,
    reid_checkpoint: str | Path | None = None,
    fps: float = 30.0,
) -> SequenceDetections:
    """Run the detector over one sequence exactly once.

    Returns immutable :class:`DetectedBox` rows, already filtered by
    ``conf_floor``. Replaying one sequence with several trackers from this is
    equivalent to calling :func:`replay_sequence` per tracker, and costs one
    inference pass rather than N.
    """
    from ..data.frameindex import load_index

    detections = SequenceDetections(
        dataset=dataset, variant=variant, sequence_id=sequence_id, modality=modality
    )

    records = [
        r for r in load_index(dataset, variant)
        if r.sequence_id == sequence_id and r.modality == modality
    ]
    if not records:
        detections.error = (
            f"no frames for {dataset}/{variant}/{sequence_id}/{modality}. "
            f"Check that the converter produced this sequence."
        )
        return detections
    records.sort(key=lambda r: r.frame_index)

    root = interim_root(dataset, variant)
    embedder = _maybe_embedder(with_embeddings, reid_checkpoint)

    frame_index = 0
    for record in records:
        if max_frames is not None and frame_index >= max_frames:
            break

        image_path = root / record.image
        if not image_path.is_file():
            continue

        image = read_image(image_path)
        frame_result = detector.predict_image(image)

        # Recorded before the confidence floor, because the per-frame dump
        # reports what the detector saw, not what survived filtering.
        detections.raw_counts.append(len(frame_result.detections))

        observations = [
            TrackObservation(
                frame_index=frame_index,
                timestamp_s=frame_index / max(fps, 1e-6),
                box=d.box,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                image_height_px=frame_result.height or image.shape[0],
            )
            for d in frame_result.detections
            if d.confidence >= conf_floor
        ]

        if embedder is not None and observations:
            _attach_embeddings(embedder, image, observations)

        detections.frames.append(
            tuple(
                DetectedBox(
                    box=o.box,
                    confidence=o.confidence,
                    class_id=o.class_id,
                    class_name=o.class_name,
                    image_height_px=o.image_height_px,
                    embedding=(
                        None if o.embedding is None else tuple(float(v) for v in o.embedding)
                    ),
                )
                for o in observations
            )
        )
        frame_index += 1

    return detections


def replay_from_detections(
    detections: SequenceDetections,
    ground_truth: dict[int, FrameGT],
    *,
    tracker_name: TrackerName | str = TrackerName.BOTSORT,
    gt_source: str = "none",
    has_ids: bool = True,
    conf_floor: float = DEFAULT_FLOOR,
    iou_threshold: float = 0.5,
    record_history: bool = True,
    dump_per_frame: bool = False,
    fps: float = 30.0,
    **tracker_kwargs: Any,
) -> ReplayResult:
    """Associate and score a sequence from detections that were already computed."""
    result = ReplayResult(
        sequence=f"{detections.dataset}/{detections.variant}/{detections.sequence_id}",
        tracker=str(tracker_name),
    )

    if detections.error:
        result.error = detections.error
        return result

    tracker = build_tracker(tracker_name, **tracker_kwargs)
    predictions: dict[int, FramePred] = {}

    for frame_index, boxes in enumerate(detections.frames):
        # Fresh observations per tracker: TrackObservation is mutable and a track
        # holds a reference to the one it was matched with.
        observations = [
            box.as_observation(frame_index, frame_index / max(fps, 1e-6)) for box in boxes
        ]

        tracks: FrameTracks = tracker.update(
            observations, frame_index=frame_index, timestamp_s=frame_index / max(fps, 1e-6)
        )

        if record_history:
            for track in tracks.tracks:
                track.history[frame_index] = track.box

        predictions[frame_index] = FramePred(
            frame_index=frame_index,
            objects=[(t.track_id, t.box) for t in tracks.tracks if t.state is not TrackState.DEAD],
        )

        if dump_per_frame:
            raw = detections.raw_counts[frame_index] if frame_index < len(detections.raw_counts) else 0
            result.per_frame.append(
                {
                    "frame": frame_index,
                    "detections": raw,
                    "tracks": [
                        {
                            "id": t.track_id,
                            "state": t.state.value,
                            "box": [round(v, 2) for v in t.box],
                            "conf": round(t.confidence, 3),
                            "hits": t.hits,
                        }
                        for t in tracks.tracks
                    ],
                }
            )

        result.frames += 1
        result.detections += len(observations)

    for track in tracker.tracks:
        result.tracks_created += 1
        if track.state is TrackState.CONFIRMED:
            result.tracks_confirmed += 1

    result.metrics = evaluate_sequence(
        detections.sequence_id,
        _resample_ground_truth(ground_truth, len(predictions)),
        predictions,
        iou_threshold=iou_threshold,
        visibility_aware=detections.dataset == "antiuav",
    )
    result.metrics.notes.append(f"ground truth source: {gt_source}")
    if not has_ids:
        result.warnings.append(
            "Ground truth has no track ids (single-target id=1). MOTA and HOTA are "
            "meaningful; IDF1 and IDSW are not, because every identity looks like the "
            "same one."
        )
    if result.detections == 0:
        result.warnings.append(
            "The detector produced no detections above the confidence floor "
            f"({conf_floor}). Metrics are 0 by definition - check the checkpoint and "
            "the confidence threshold before concluding the tracker is at fault."
        )
    return result


def replay_sequence(
    detector: Any,
    dataset: str,
    variant: str,
    sequence_id: str,
    *,
    modality: str = "rgb",
    tracker_name: TrackerName | str = TrackerName.BOTSORT,
    conf_floor: float = DEFAULT_FLOOR,
    iou_threshold: float = 0.5,
    max_frames: int | None = None,
    with_embeddings: bool = True,
    reid_checkpoint: str | Path | None = None,
    record_history: bool = True,
    dump_per_frame: bool = False,
    fps: float = 30.0,
    **tracker_kwargs: Any,
) -> ReplayResult:
    """Run one sequence end to end and score it.

    ``detector`` is any object exposing ``predict_image`` - normally a
    :class:`~anti_uav.detection.predictor.Detector`. Passing a stub makes this
    testable without a GPU, which is how the whole tracking stack is exercised on
    the CPU dev box.

    One detection pass, one tracker. :func:`evaluate_trackers_on_dataset` drives
    :func:`detect_sequence` and :func:`replay_from_detections` separately so that
    comparing trackers does not repeat inference per tracker.
    """
    detections = detect_sequence(
        detector,
        dataset,
        variant,
        sequence_id,
        modality=modality,
        conf_floor=conf_floor,
        max_frames=max_frames,
        with_embeddings=with_embeddings,
        reid_checkpoint=reid_checkpoint,
        fps=fps,
    )
    if not detections.ok:
        failed = ReplayResult(
            sequence=f"{dataset}/{variant}/{sequence_id}", tracker=str(tracker_name)
        )
        failed.error = detections.error
        return failed

    from ..data.frameindex import load_index

    records = [
        r for r in load_index(dataset, variant)
        if r.sequence_id == sequence_id and r.modality == modality
    ]
    ground_truth, gt_source, has_ids = load_ground_truth_for(
        dataset, variant, sequence_id, modality, records=records
    )
    if not ground_truth:
        failed = ReplayResult(
            sequence=f"{dataset}/{variant}/{sequence_id}", tracker=str(tracker_name)
        )
        failed.error = (
            f"no ground truth for {sequence_id}. Converted datasets keep the "
            f"publisher's MOT file under data/interim/{dataset}/{variant}/mot/; "
            f"falling back to the YOLO labels requires at least one converted frame."
        )
        return failed

    return replay_from_detections(
        detections,
        ground_truth,
        tracker_name=tracker_name,
        gt_source=gt_source,
        has_ids=has_ids,
        conf_floor=conf_floor,
        iou_threshold=iou_threshold,
        record_history=record_history,
        dump_per_frame=dump_per_frame,
        fps=fps,
        **tracker_kwargs,
    )


def _resample_ground_truth(gt: dict[int, FrameGT], n_frames: int) -> dict[int, FrameGT]:
    """Align ground-truth frame indices to the replayed frame indices.

    The converter writes MOT rows keyed on the *publisher's* frame numbering,
    which survives stride-based extraction but not a `max_frames` truncation, so
    the two are aligned by index here. Explicit and lossy, rather than silently
    offsetting the ground truth against the predictions by one frame.
    """
    keys = sorted(gt)
    if not keys:
        return {}
    if keys == list(range(len(keys))) and len(keys) <= n_frames:
        return gt
    return {
        index: gt[key]
        for index, key in enumerate(keys)
        if index < n_frames
    }


def _maybe_embedder(enabled: bool, checkpoint: str | Path | None = None) -> Any:
    """The appearance embedder for this run, or ``None``.

    A missing or unloadable ReID model is a normal state, not an error: the
    tracker falls back to motion-only association and says so in its log.
    """
    if not enabled:
        return None
    try:
        from .appearance import AppearanceEmbedder

        return AppearanceEmbedder.load_default(checkpoint)
    except Exception as exc:
        log.info(
            "no appearance embedder available; running motion-only association",
            extra={"reason": str(exc)},
        )
        return None


def _attach_embeddings(embedder: Any, image: np.ndarray, observations: Sequence[TrackObservation]) -> None:
    """Crop each detection and embed it. Failures leave the embedding unset.

    Leaving it unset is the right failure mode: the tracker's appearance gate
    treats a missing embedding as neutral rather than as disagreement.
    """
    try:
        boxes = np.asarray([o.box for o in observations], dtype=float).reshape(-1, 4)
        crops = embedder.crops(image, boxes)
        for observation, crop in zip(observations, crops, strict=False):
            observation.embedding = embedder.embed(crop)
    except Exception as exc:
        log.debug("embedding failed; continuing motion-only", extra={"reason": str(exc)})


# --------------------------------------------------------------------------- #
# multi-sequence evaluation
# --------------------------------------------------------------------------- #


def evaluate_trackers_on_dataset(
    detector: Any,
    dataset: str,
    variant: str,
    *,
    sequences: Sequence[str] | None = None,
    limit: int | None = 20,
    modality: str = "rgb",
    trackers: Sequence[TrackerName | str] = (TrackerName.SORT, TrackerName.BYTETRACK, TrackerName.BOTSORT),
    iou_threshold: float = 0.5,
    conf_floor: float = DEFAULT_FLOOR,
    save: bool = True,
    reid_checkpoint: str | Path | None = None,
    **tracker_kwargs: Any,
) -> dict[str, EvalReport]:
    """Score every tracker on the same sequences with the same detections.

    Detections are computed **once per sequence** by :func:`detect_sequence` and
    reused across every tracker. That is both much faster on a three-tracker sweep
    and strictly more correct: any difference between the tracker rows is then
    attributable to the tracker alone, not to detector nondeterminism.
    """
    from ..data.frameindex import load_index

    records = load_index(dataset, variant)
    available = sorted({r.sequence_id for r in records if r.modality == modality})
    chosen = list(sequences) if sequences else available[:limit]

    if not chosen:
        log.warning("no sequences to evaluate", extra={"dataset": dataset, "variant": variant})
        return {}

    rules = load_rules()
    common = {
        "high_threshold": rules.confidence.initiate,
        "low_threshold": max(0.01, rules.confidence.maintain / 4.0),
        "match_threshold": rules.tracking.max_association_cost,
        "max_age": rules.tracking.max_target_age_frames * 6,
        "min_hits": rules.persistence.min_hits,
        **tracker_kwargs,
    }

    reports: dict[str, EvalReport] = {}
    per_sequence: dict[str, list[SequenceResult]] = {}

    for name in trackers:
        per_sequence[str(name)] = []

    n_tracker_failures = 0
    for sequence_id in chosen:
        log.info("replaying", extra={"dataset": dataset, "sequence": sequence_id})

        detections = detect_sequence(
            detector,
            dataset,
            variant,
            sequence_id,
            modality=modality,
            conf_floor=conf_floor,
            reid_checkpoint=reid_checkpoint,
        )
        if not detections.ok:
            log.warning(
                "detection failed",
                extra={"dataset": dataset, "sequence": sequence_id, "error": detections.error},
            )
            continue

        sequence_records = [r for r in records if r.sequence_id == sequence_id]
        ground_truth, gt_source, has_ids = load_ground_truth_for(
            dataset, variant, sequence_id, modality, records=sequence_records
        )
        if not ground_truth:
            # Scored against nothing, every prediction would be a false positive
            # and MOTA would read 0. That is a broken row, not a bad tracker.
            log.warning(
                "no ground truth; skipping sequence",
                extra={"dataset": dataset, "variant": variant, "sequence": sequence_id},
            )
            continue

        for name in trackers:
            result = replay_from_detections(
                detections,
                ground_truth,
                tracker_name=name,
                gt_source=gt_source,
                has_ids=has_ids,
                conf_floor=conf_floor,
                iou_threshold=iou_threshold,
                **common,
            )
            if not result.ok or result.metrics is None:
                n_tracker_failures += 1
                log.warning(
                    "replay failed",
                    extra={"sequence": sequence_id, "tracker": str(name), "error": result.error},
                )
                continue
            per_sequence[str(name)].append(result.metrics)

    if n_tracker_failures:
        log.warning(
            "some tracker/sequence combinations failed to replay",
            extra={"failures": n_tracker_failures},
        )

    from .metrics import evaluate_report

    for name, results in per_sequence.items():
        if results:
            reports[name] = evaluate_report(
                results, tracker=name, dataset=f"{dataset}/{variant}", iou_threshold=iou_threshold
            )

    if save and reports:
        write_json(
            subdir("tracks") / f"eval_{dataset}_{variant}.json",
            {name: report.to_dict() for name, report in reports.items()},
        )

    return reports


def format_reports(reports: dict[str, EvalReport]) -> str:
    """Ranked tracker table for one dataset."""
    if not reports:
        return "no tracker results - check that the dataset was converted and split"
    from .metrics import comparison_table

    return comparison_table(list(reports.values()))


def dump_predictions(
    result: ReplayResult,
    path: str | Path,
) -> Path:
    """Write per-frame track output in the MOT layout for external toolkits.

    Lets MM-UAV's own evaluation toolkit score our tracker output with their code,
    which is the cleanest way to check our numbers against theirs.

    Columns: ``frame,id,x1,y1,x2,y2,conf,class,visibility``, with 1-based frame
    numbers — the same layout :func:`anti_uav.data.convert.base.write_mot_gt`
    writes, so our output and our converted ground truth are mutually readable
    by ``load_mot_ground_truth``.

    Note this is an **xyxy** layout. MOTChallenge and TrackEval conventionally use
    ``bb_left,bb_top,bb_width,bb_height``, so a loader that assumes the MOT17
    column order will read these rows as garbage. MM-UAV's own toolkit uses xyxy,
    which is why it is not converted here.
    """
    target = Path(path)
    lines: list[str] = []
    for frame in result.per_frame:
        for track in frame["tracks"]:
            x1, y1, x2, y2 = track["box"]
            lines.append(
                f"{frame['frame'] + 1},{track['id']},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
                f"{track['conf']:.4f},1,1.0"
            )
    body = "\n".join(lines) + ("\n" if lines else "")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def available_sequences(dataset: str, variant: str, modality: str = "rgb") -> list[str]:
    from ..data.frameindex import load_index

    return sorted({r.sequence_id for r in load_index(dataset, variant) if r.modality == modality})


def default_variant(dataset: str) -> str:
    registry = load_registry()
    spec = registry.datasets.get(dataset)
    return spec.default_variant or "full" if spec else "full"


def iter_datasets() -> Iterable[str]:
    return load_registry().aliases
