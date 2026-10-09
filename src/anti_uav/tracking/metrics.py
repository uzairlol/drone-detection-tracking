"""Tracking metrics.

Implemented natively rather than pulled from ``motmetrics`` or ``TrackEval``, for
three reasons that matter here:

1. **Anti-UAV's own metric.** The challenge defines a tracking accuracy that
   accounts for per-frame visibility flags - a frame where the target is genuinely
   absent must not be scored as a miss. No off-the-shelf MOT metric models that,
   so writing the ~30 lines is cheaper than fighting a framework to do it.
2. **Two different IoU conventions in the wild.** MOT uses 0.5 (or 0.7 for AMOT);
   Anti-UAV's published numbers and the aerial-detection literature generally use
   0.5 too, but MM-UAV's toolkit reports HOTA across a range. Getting the
   convention wrong silently shifts every number.
3. **No install friction.** ``motmetrics`` has a long history of dependency
   conflicts; this project already has numpy and scipy.

Metrics implemented
-------------------
``MOTA``   Missed/False-positive/ID-switch rate. Harshly punishes fragmentation,
            which is the right pressure for a system that pages a human.
``IDF1``   Identity association quality. The complement of MOTA: tolerant of
            fragmentation, intolerant of identity switches.
``HOTA``   Geometric + association average over IoU thresholds 0.5:0.95. The most
            informative single number for this task, because it does not reward
            either error being traded for the other.
``IDSW``   Raw identity switch count. Reported alongside the others because a
            system can have a good IDF1 and still swap identities visibly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..utils.geometry import iou_matrix
from .types import Track

_EPS = 1e-9

#: HOTA integrates over this IoU range, as in the HOTA paper.
HOTA_IOU_RANGE = np.arange(0.5, 0.96, 0.05)


@dataclass(slots=True)
class SequenceResult:
    """Per-sequence metric values. These are what get averaged over a dataset."""

    sequence: str
    frames: int = 0
    gt_boxes: int = 0
    pred_boxes: int = 0
    matched: int = 0
    misses: int = 0
    false_positives: int = 0
    id_switches: int = 0
    mota: float = 0.0
    idf1: float = 0.0
    hota: float = 0.0
    hota_components: dict[str, float] = field(default_factory=dict)
    fragmentations: int = 0
    #: Mean number of frames a predicted identity spent matched to something.
    #: Low values with a high ``matched`` count mean the tracker is fragmenting
    #: one target into many short identities.
    track_length_avg: float = 0.0
    #: Anti-UAV's visibility-aware accuracy, when the ground truth has flags.
    anti_uav_accuracy: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "frames": self.frames,
            "gt_boxes": self.gt_boxes,
            "pred_boxes": self.pred_boxes,
            "matched": self.matched,
            "misses": self.misses,
            "false_positives": self.false_positives,
            "id_switches": self.id_switches,
            "mota": round(self.mota, 5),
            "idf1": round(self.idf1, 5),
            "hota": round(self.hota, 5),
            "hota_components": {k: round(v, 5) for k, v in self.hota_components.items()},
            "fragmentations": self.fragmentations,
            "track_length_avg": round(self.track_length_avg, 5),
            "anti_uav_accuracy": (
                None if self.anti_uav_accuracy is None else round(self.anti_uav_accuracy, 5)
            ),
        }


@dataclass(slots=True)
class EvalReport:
    """Aggregated metrics over many sequences."""

    sequences: list[SequenceResult] = field(default_factory=list)
    tracker: str = ""
    dataset: str = ""
    iou_threshold: float = 0.5

    @property
    def mota(self) -> float:
        """Corpus-level MOTA.

        Computed from summed counts, **not** averaged per sequence. Averaging
        per-sequence MOTA over-weights short sequences, and MM-UAV's sequences
        vary by an order of magnitude in length.
        """
        total = sum(s.gt_boxes for s in self.sequences)
        if total == 0:
            return 0.0
        return 1.0 - (
            sum(s.misses + s.false_positives + s.id_switches for s in self.sequences) / total
        )

    @property
    def idf1(self) -> float:
        """Corpus-level IDF1 - an F1 over the whole id-matching problem."""
        gt_total = sum(s.gt_boxes for s in self.sequences)
        pred_total = sum(s.pred_boxes for s in self.sequences)
        matched = sum(s.matched for s in self.sequences)
        if gt_total == 0 or pred_total == 0:
            return 0.0
        # idtp / idfn / idfp are the standard MOTChallenge decomposition of IDF1
        # into a precision and a recall over *identities*.
        idtp = matched
        idfn = gt_total - matched
        idfp = pred_total - matched
        denominator = 2.0 * idtp + idfp + idfn
        return (2.0 * idtp / denominator) if denominator > _EPS else 0.0

    @property
    def hota(self) -> float:
        return float(np.mean([s.hota for s in self.sequences])) if self.sequences else 0.0

    @property
    def id_switches(self) -> int:
        return sum(s.id_switches for s in self.sequences)

    @property
    def fragmentations(self) -> int:
        return sum(s.fragmentations for s in self.sequences)

    @property
    def gt_boxes(self) -> int:
        return sum(s.gt_boxes for s in self.sequences)

    @property
    def precision(self) -> float:
        matched = sum(s.matched for s in self.sequences)
        predicted = sum(s.pred_boxes for s in self.sequences)
        return matched / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        matched = sum(s.matched for s in self.sequences)
        return matched / self.gt_boxes if self.gt_boxes else 0.0

    @property
    def anti_uav_accuracy(self) -> float | None:
        values = [s.anti_uav_accuracy for s in self.sequences if s.anti_uav_accuracy is not None]
        return float(np.mean(values)) if values else None

    def to_dict(self) -> dict:
        return {
            "tracker": self.tracker,
            "dataset": self.dataset,
            "iou_threshold": self.iou_threshold,
            "sequences": len(self.sequences),
            "gt_boxes": self.gt_boxes,
            "mota": round(self.mota, 5),
            "idf1": round(self.idf1, 5),
            "hota": round(self.hota, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "anti_uav_accuracy": (
                None if self.anti_uav_accuracy is None else round(self.anti_uav_accuracy, 5)
            ),
            "per_sequence": [s.to_dict() for s in self.sequences],
        }

    def summary(self) -> str:
        anti = self.anti_uav_accuracy
        lines = [
            f"tracker        : {self.tracker}",
            f"dataset        : {self.dataset}",
            f"IoU threshold  : {self.iou_threshold}",
            f"sequences      : {len(self.sequences)}   gt boxes: {self.gt_boxes}",
            "",
            f"  MOTA      {self.mota:>8.4f}",
            f"  IDF1      {self.idf1:>8.4f}",
            f"  HOTA      {self.hota:>8.4f}",
            f"  precision {self.precision:>8.4f}",
            f"  recall    {self.recall:>8.4f}",
            f"  IDSW      {self.id_switches:>8d}   fragmentations: {self.fragmentations}",
        ]
        if anti is not None:
            lines.append(f"  Anti-UAV accuracy {anti:>8.4f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# per-sequence evaluation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FrameGT:
    """Ground truth for one frame."""

    frame_index: int
    #: ``[(track_id, box, visibility), ...]``, visibility in [0, 1].
    objects: list[tuple[int, tuple[float, float, float, float], float]] = field(
        default_factory=list
    )

    @property
    def visible(self) -> list[tuple[int, tuple[float, float, float, float]]]:
        return [(tid, box) for tid, box, vis in self.objects if vis > 0.0]


@dataclass(slots=True)
class FramePred:
    """Predictions for one frame."""

    frame_index: int
    #: ``[(track_id, box), ...]``
    objects: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)


def evaluate_sequence(
    sequence: str,
    gt_frames: dict[int, FrameGT],
    pred_frames: dict[int, FramePred],
    *,
    iou_threshold: float = 0.5,
    visibility_aware: bool = False,
) -> SequenceResult:
    """Score one sequence with greedy IoU matching.

    Greedy rather than Hungarian: MOTChallenge's CLEAR-MOT definition *is* greedy
    (highest-IoU match first), and matching that definition is what makes a number
    from here comparable to a published one. ByteTrack/HOTA papers that use an
    optimal assignment will score slightly higher, and it is worth knowing which
    you are looking at.
    """
    result = SequenceResult(sequence=sequence)

    # Track id remapping: predictions get their own index space.
    all_frames = sorted(set(gt_frames) | set(pred_frames))
    result.frames = len(all_frames)

    matched_count = 0
    gt_count = 0
    pred_count = 0
    misses = 0
    false_positives = 0
    id_switches = 0
    fragmentations = 0

    #: Ground-truth id -> the prediction id it matched on the previous frame.
    previous_match: dict[int, int | None] = {}
    #: Prediction id -> frames on which it matched something. Lets a run report
    #: how long its identities actually survived, which is what distinguishes a
    #: tracker that follows one target from one that shreds it into many.
    pred_lengths: dict[int, int] = {}
    iou_accumulator = HotaAccumulator()
    anti_uav_score: list[float] = []
    anti_uav_present_frames = 0

    for frame_index in all_frames:
        gt = gt_frames.get(frame_index, FrameGT(frame_index=frame_index))
        pred = pred_frames.get(frame_index, FramePred(frame_index=frame_index))

        gt_objects = gt.visible
        pred_objects = list(pred.objects)

        if visibility_aware:
            anti_uav_score.append(_anti_uav_frame(gt, pred_objects))
            if gt_objects:
                anti_uav_present_frames += 1

        gt_count += len(gt_objects)
        pred_count += len(pred_objects)

        pairs, unmatched_gt, _unmatched_pred = _greedy_match(
            gt_objects, pred_objects, iou_threshold
        )
        iou_accumulator.update(gt_objects, pred_objects, pairs, iou_threshold)

        # An ID switch is a ground-truth id that was matched to prediction P on
        # the previous frame and to a different prediction Q on this one.
        for gt_idx, pred_idx in pairs:
            gt_id = gt_objects[gt_idx][0]
            pred_id = pred_objects[pred_idx][0]
            was = previous_match.get(gt_id)
            if was is not None and was != pred_id:
                id_switches += 1
            previous_match[gt_id] = pred_id
            pred_lengths[pred_id] = pred_lengths.get(pred_id, 0) + 1
            matched_count += 1

        for gt_idx in unmatched_gt:
            misses += 1
            gt_id = gt_objects[gt_idx][0]
            if previous_match.get(gt_id) is not None:
                # Tracked, then lost. A fragmentation rather than a plain miss -
                # the two must be separable when tuning the recovery budget.
                fragmentations += 1
            previous_match[gt_id] = None

        false_positives += len(_unmatched_pred)

    result.gt_boxes = gt_count
    result.pred_boxes = pred_count
    result.matched = matched_count
    result.misses = misses
    result.false_positives = false_positives
    result.id_switches = id_switches
    result.fragmentations = fragmentations
    if pred_lengths:
        result.track_length_avg = sum(pred_lengths.values()) / len(pred_lengths)
    result.mota = (
        1.0 - (misses + false_positives + id_switches) / gt_count if gt_count else 0.0
    )
    result.idf1 = _sequence_idf1(matched_count, gt_count, pred_count)
    result.hota, result.hota_components = iou_accumulator.finalise()
    if anti_uav_score and anti_uav_present_frames:
        result.anti_uav_accuracy = float(np.mean(anti_uav_score))
    if gt_count == 0:
        result.notes.append("no visible ground truth in this sequence; metrics are 0 by definition")
    return result


def _greedy_match(
    gt_objects: Sequence[tuple[int, tuple[float, float, float, float]]],
    pred_objects: Sequence[tuple[int, tuple[float, float, float, float]]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Highest-IoU-first matching, the CLEAR-MOT definition.

    Returns ``(pairs, unmatched_gt_indices, unmatched_pred_indices)`` where pairs
    index into the *original* lists.
    """
    if not gt_objects or not pred_objects:
        return [], list(range(len(gt_objects))), list(range(len(pred_objects)))

    gt_boxes = np.asarray([b for _id, b in gt_objects], dtype=float).reshape(-1, 4)
    pred_boxes = np.asarray([b for _id, b in pred_objects], dtype=float).reshape(-1, 4)
    iou = iou_matrix(gt_boxes, pred_boxes)

    pairs: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # Iterate over all (gt, pred) pairs in descending IoU. Using a flat sort
    # rather than per-row argmax is what makes this greedy-overall rather than
    # greedy-per-row, which is the definition that matches published numbers.
    flat = np.argsort(-iou, axis=None, kind="stable")
    for flat_index in flat:
        gi, pi = divmod(int(flat_index), iou.shape[1])
        if gi in used_gt or pi in used_pred:
            continue
        if iou[gi, pi] < iou_threshold:
            break
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi))

    return (
        pairs,
        [i for i in range(len(gt_objects)) if i not in used_gt],
        [i for i in range(len(pred_objects)) if i not in used_pred],
    )


def _sequence_idf1(matched: int, gt_total: int, pred_total: int) -> float:
    if gt_total == 0 or pred_total == 0:
        return 0.0
    idtp = matched
    idfn = gt_total - matched
    idfp = pred_total - matched
    denominator = 2.0 * idtp + idfp + idfn
    return (2.0 * idtp / denominator) if denominator > _EPS else 0.0


def _anti_uav_frame(
    gt: FrameGT,
    predictions: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> float:
    """Anti-UAV's per-frame accuracy.

    From the challenge definition: for each frame, the IoU between the predicted
    box and its ground-truth box, where the predicted visibility flag ``p`` is 0
    when the tracker claims the box is empty and 1 otherwise.

    **The average covers every frame in the union of the ground-truth and
    prediction frame keys — including the frames where the target is absent.**
    That is deliberate and it is what the challenge intends, even though it is
    easy to misread: a frame with no visible target scores 1.0 for correctly
    abstaining, so a tracker that fires on empty sky is penalised rather than
    rewarded. Averaging only over present frames would score a tracker that
    hallucinates in every empty frame as perfect.

    The per-frame cases:

    * **no visible ground truth** (``v == 0``, or no annotation at all) -
      predicting *nothing* scores 1.0, predicting a box scores 0.0. This is the
      case Anti-UAV's visibility flags exist for, and it is the single biggest
      source of divergence between "our tracker scores 0.7" and the challenge
      leaderboard.
    * **target present** - the best IoU against the visible ground truth, or 0.0
      if nothing was predicted.
    * **partial occlusion** (``0 < v < 1``) - still scored, against the
      annotation the converter already clipped to the visible region.
    """
    if not gt.objects:
        # Target absent: the best possible answer is to predict nothing.
        return 1.0 if not predictions else 0.0

    visible = gt.visible
    if not visible:
        return 1.0 if not predictions else 0.0

    if not predictions:
        return 0.0

    gt_boxes = np.asarray([b for _id, b in visible], dtype=float).reshape(-1, 4)
    pred_boxes = np.asarray([b for _id, b in predictions], dtype=float).reshape(-1, 4)
    iou = iou_matrix(gt_boxes, pred_boxes)
    if not iou.size:
        return 0.0
    return float(max(0.0, float(iou.max())))


class HotaAccumulator:
    """Streaming HOTA.

    HOTA is the geometric mean of a detection score and an association score,
    averaged over IoU thresholds 0.5:0.95. Accumulating incrementally means a
    2.8M-frame MM-UAV sequence does not need its whole IoU history in memory.

    Formula per IoU threshold alpha::

        DetA(alpha)  = TP / (TP + FN + FP)
        AssA(alpha) = TP / (TP + FN + IDSW)
        HOTA(alpha) = sqrt(DetA * AssA)

    and the reported HOTA is the mean over alpha. Note that a tracker cannot
    improve this by trading fragmentation for identity switches - which is
    precisely why it is the metric to rank on.
    """

    def __init__(self, iou_range: np.ndarray = HOTA_IOU_RANGE) -> None:
        self.iou_range = np.asarray(iou_range, dtype=float)
        self.tp = np.zeros_like(self.iou_range)
        self.fp = np.zeros_like(self.iou_range)
        self.fn = np.zeros_like(self.iou_range)
        #: Identity switches per IoU threshold.
        self.idswitch = np.zeros_like(self.iou_range)
        self.frames_with_gt = 0
        self.frames_with_pred = 0
        #: ``{threshold_index: {gt_id: pred_id}}`` from the previous frame.
        self._last_mapping: list[dict[int, int]] = [{} for _ in self.iou_range]

    def update(
        self,
        gt_objects: Sequence[tuple[int, tuple[float, float, float, float]]],
        pred_objects: Sequence[tuple[int, tuple[float, float, float, float]]],
        pairs: Sequence[tuple[int, int]],
        primary_iou: float,
    ) -> None:
        """Accumulate one frame.

        Matching is recomputed per IoU threshold rather than reusing the single
        ``primary_iou`` match, which is what makes HOTA's averaging meaningful: a
        tracker that only just clears 0.5 should not be credited with 0.95 matches.

        ``pairs`` and ``primary_iou`` are accepted for interface symmetry with the
        per-threshold path but unused - recomputing here is deliberate.
        """
        del pairs, primary_iou

        if gt_objects:
            self.frames_with_gt += 1
        if pred_objects:
            self.frames_with_pred += 1

        if not gt_objects and not pred_objects:
            return

        if not gt_objects:
            for index in range(len(self.iou_range)):
                self.fp[index] += len(pred_objects)
            return

        if not pred_objects:
            for index in range(len(self.iou_range)):
                self.fn[index] += len(gt_objects)
                self._last_mapping[index] = {}
            return

        gt_ids = [tid for tid, _b in gt_objects]
        pred_ids = [tid for tid, _b in pred_objects]

        for index, threshold in enumerate(self.iou_range):
            frame_pairs, missed, spurious = _greedy_match(
                gt_objects, pred_objects, float(threshold)
            )
            tp = len(frame_pairs)
            self.tp[index] += tp
            self.fn[index] += len(missed)
            self.fp[index] += len(spurious)

            mapping = {gt_ids[g]: pred_ids[p] for g, p in frame_pairs}
            previous = self._last_mapping[index]
            switches = sum(
                1
                for gt_id, pred_id in mapping.items()
                if gt_id in previous and previous[gt_id] != pred_id
            )
            self.idswitch[index] += switches
            self._last_mapping[index] = mapping

    def finalise(self) -> tuple[float, dict[str, float]]:
        """Mean HOTA plus its detection/association components."""
        if self.frames_with_gt == 0:
            return 0.0, {"det_a": 0.0, "ass_a": 0.0}

        det_a = np.zeros_like(self.iou_range)
        ass_a = np.zeros_like(self.iou_range)

        for index in range(len(self.iou_range)):
            tp = self.tp[index]
            fn = self.fn[index]
            fp = self.fp[index]
            switches = self.idswitch[index]

            det_denominator = tp + fn + fp
            ass_denominator = tp + fn + switches
            det_a[index] = tp / det_denominator if det_denominator > _EPS else 0.0
            ass_a[index] = tp / ass_denominator if ass_denominator > _EPS else 0.0

        hota_per_threshold = np.sqrt(np.maximum(det_a, 0.0) * np.maximum(ass_a, 0.0))
        return (
            float(np.mean(hota_per_threshold)),
            {
                "det_a": float(np.mean(det_a)),
                "ass_a": float(np.mean(ass_a)),
            },
        )


def evaluate_report(
    results: Sequence[SequenceResult],
    *,
    tracker: str = "",
    dataset: str = "",
    iou_threshold: float = 0.5,
) -> EvalReport:
    return EvalReport(
        sequences=list(results),
        tracker=tracker,
        dataset=dataset,
        iou_threshold=iou_threshold,
    )


def mean_or_zero(values: Iterable[float]) -> float:
    collected = [v for v in values if not math.isnan(v)]
    return float(np.mean(collected)) if collected else 0.0


def tracks_to_frames(
    tracks: Iterable[Track],
    *,
    min_hits: int = 1,
    include_lost: bool = False,
) -> dict[int, FramePred]:
    """Reconstruct per-frame predictions from finished tracks.

    A ``Track`` stores only its latest box, so frame-accurate history must come
    from ``Track.history``, which :mod:`anti_uav.tracking.replayer` records as it
    runs. Tracks without history contribute a single frame at ``last_frame``.
    """
    return tracks_to_histories(tracks, min_hits=min_hits, include_lost=include_lost)


def tracks_to_histories(
    tracks: Iterable[Track],
    *,
    min_hits: int = 1,
    include_lost: bool = False,
) -> dict[int, FramePred]:
    """``{frame_index: FramePred}`` for every frame any track was observed in.

    Uses ``Track.history`` when the replayer recorded it (the normal case), so the
    reconstruction is frame-accurate. Falls back to the latest box for tracks
    without history.
    """
    allowed = {"confirmed", "tentative"} | ({"lost"} if include_lost else set())
    frames: dict[int, FramePred] = {}

    for track in tracks:
        if track.hits < min_hits or track.state.value not in allowed:
            continue

        history = getattr(track, "history", None)
        if history:
            for frame_index, box in history.items():
                frames.setdefault(frame_index, FramePred(frame_index)).objects.append(
                    (track.track_id, box)
                )
        else:
            frames.setdefault(track.last_frame, FramePred(track.last_frame)).objects.append(
                (track.track_id, track.box)
            )

    return frames


def mot_gt_from_records(
    records: Sequence[tuple[int, int, tuple[float, float, float, float], float]],
    visibility_aware: bool = False,
) -> dict[int, FrameGT]:
    """``[(frame, track_id, box, visibility)]`` -> per-frame ground truth."""
    out: dict[int, FrameGT] = {}
    for frame_index, track_id, box, visibility in records:
        if visibility_aware and visibility <= 0.0:
            out.setdefault(frame_index, FrameGT(frame_index)).objects.append((track_id, box, 0.0))
            continue
        out.setdefault(frame_index, FrameGT(frame_index)).objects.append(
            (track_id, box, visibility)
        )
    return out


def comparison_table(reports: Sequence[EvalReport]) -> str:
    """The tracker comparison table for the write-up."""
    if not reports:
        return "no results"
    header = (
        f"{'tracker':<20}{'dataset':<12}{'MOTA':>8}{'IDF1':>8}{'HOTA':>8}"
        f"{'prec':>8}{'recall':>8}{'IDSW':>7}{'frag':>7}{'seqs':>6}"
    )
    lines = [header, "-" * len(header)]
    for report in sorted(reports, key=lambda r: -r.hota):
        lines.append(
            f"{report.tracker:<20}{report.dataset:<12}"
            f"{report.mota:>8.4f}{report.idf1:>8.4f}{report.hota:>8.4f}"
            f"{report.precision:>8.4f}{report.recall:>8.4f}"
            f"{report.id_switches:>7}{report.fragmentations:>7}"
            f"{len(report.sequences):>6}"
        )
    lines.append("")
    lines.append("HOTA is the column to rank on. MOTA punishes fragmentation harder")
    lines.append("than IDF1 does, so the two can disagree about which tracker is better;")
    lines.append("HOTA does not reward trading one error for the other.")
    return "\n".join(lines)
