"""Dataset statistics.

The point of this module is to answer "is this dataset what I think it is?"
*before* training, using numbers that would betray a broken conversion.

Two of these are load-bearing for this project specifically:

``box_size_histogram``
    If most MM-UAV boxes land in the ``<8 px`` bucket, either tiling is inactive
    or the detector has no chance. That is a 10-second check that saves a day.

``bird_coverage``
    Anti-UAV and MM-UAV are drone-only. Any run containing them without
    ``dvb``/``mavvid`` has never been shown a bird, so its precision number is
    unfalsifiable. Surfacing that on every run is the difference between a
    usable result table and a misleading one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.imaging import list_images
from ..utils.io import write_json
from ..utils.logging import get_logger
from .frameindex import FrameRecord, load_index, load_report
from .harmonize import box_size_histogram, read_yolo_label

log = get_logger(__name__)

SIZE_BINS = (0, 8, 16, 32, 64, 128, 256, float("inf"))


@dataclass(slots=True)
class DatasetStats:
    dataset: str
    variant: str
    frames: int = 0
    labelled_frames: int = 0
    empty_frames: int = 0
    boxes: int = 0
    sequences: int = 0
    modalities: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    #: Sequences containing at least one bird box.
    bird_sequences: int = 0
    drone_sequences: int = 0
    size_histogram: dict[str, int] = field(default_factory=dict)
    frame_sizes: dict[str, int] = field(default_factory=dict)
    target_absent_frames: int = 0
    tracks: int = 0
    boxes_per_frame_mean: float = 0.0
    #: ``sqrt(w*h)`` percentiles, in pixels.
    size_percentiles: dict[str, float] = field(default_factory=dict)
    image_bytes: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)
    dropped_boxes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def provides_bird_negatives(self) -> bool:
        return self.bird_sequences > 0

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "variant": self.variant,
            "frames": self.frames,
            "labelled_frames": self.labelled_frames,
            "empty_frames": self.empty_frames,
            "boxes": self.boxes,
            "sequences": self.sequences,
            "modalities": self.modalities,
            "class_counts": self.class_counts,
            "bird_sequences": self.bird_sequences,
            "drone_sequences": self.drone_sequences,
            "provides_bird_negatives": self.provides_bird_negatives,
            "size_histogram": self.size_histogram,
            "frame_sizes": self.frame_sizes,
            "target_absent_frames": self.target_absent_frames,
            "tracks": self.tracks,
            "boxes_per_frame_mean": round(self.boxes_per_frame_mean, 3),
            "size_percentiles": {k: round(v, 2) for k, v in self.size_percentiles.items()},
            "image_bytes": self.image_bytes,
            "unknown_labels": self.unknown_labels,
            "dropped_boxes": self.dropped_boxes,
            "warnings": self.warnings,
        }


def compute(
    records: Sequence[FrameRecord],
    *,
    dataset: str,
    variant: str,
    unified_labels: Sequence[str] = ("drone", "bird"),
    sample_for_percentiles: int = 20_000,
    seed: int = 0,
) -> DatasetStats:
    """Aggregate stats over a converted dataset's index."""
    stats = DatasetStats(dataset=dataset, variant=variant)
    if not records:
        stats.warn("index is empty - run `anti-uav convert` first")
        return stats

    sizes: list[float] = []
    sequence_classes: dict[str, set[int]] = {}
    tracks: set[tuple[str, str, int]] = set()

    for record in records:
        stats.frames += 1
        stats.modalities[record.modality] = stats.modalities.get(record.modality, 0) + 1

        if record.box_count:
            stats.labelled_frames += 1
        else:
            stats.empty_frames += 1

        if not record.target_present:
            stats.target_absent_frames += 1

        stats.boxes += record.box_count
        for class_id in record.class_ids:
            name = unified_labels[class_id] if class_id < len(unified_labels) else f"_{class_id}"
            stats.class_counts[name] = stats.class_counts.get(name, 0) + 1
            sequence_classes.setdefault(record.group_key, set()).add(class_id)

        if record.track_id is not None:
            tracks.add((record.group_key, record.modality, record.track_id))

        size_key = f"{record.width}x{record.height}"
        stats.frame_sizes[size_key] = stats.frame_sizes.get(size_key, 0) + 1

    stats.sequences = len(sequence_classes)
    stats.bird_sequences = sum(1 for classes in sequence_classes.values() if 1 in classes)
    stats.drone_sequences = sum(1 for classes in sequence_classes.values() if 0 in classes)
    stats.tracks = len(tracks)
    stats.boxes_per_frame_mean = stats.boxes / max(stats.frames, 1)

    # Box sizes need pixel dimensions per frame, so read labels. Sampling keeps
    # this bounded on a 400k-frame dataset.
    root = _interim_root(dataset, variant)
    sample = _sample(records, sample_for_percentiles, seed)
    histogram: dict[str, int] = {
        (
            f"{SIZE_BINS[i]:g}-{SIZE_BINS[i + 1]:g}px"
            if np.isfinite(SIZE_BINS[i + 1])
            else f"{SIZE_BINS[i]:g}+px"
        ): 0
        for i in range(len(SIZE_BINS) - 1)
    }

    for record in sample:
        if not record.width or not record.height:
            continue
        label_path = root / record.label
        boxes = read_yolo_label(
            label_path,
            width=record.width,
            height=record.height,
            stats=None,
        )
        for key, count in box_size_histogram(
            boxes, width=record.width, height=record.height, bins=SIZE_BINS
        ).items():
            histogram[key] = histogram.get(key, 0) + count
        sizes.extend(_box_sides(boxes, record.width, record.height))

    stats.size_histogram = histogram
    if sizes:
        array = np.asarray(sizes, dtype=float)
        stats.size_percentiles = {
            "p05": float(np.percentile(array, 5)),
            "p25": float(np.percentile(array, 25)),
            "median": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "p95": float(np.percentile(array, 95)),
        }

    report = load_report(dataset, variant)
    if report is not None:
        stats.unknown_labels = dict(report.unknown_labels)
        stats.dropped_boxes = report.boxes_dropped

    images = list_images(root / "frames")
    stats.image_bytes = sum(p.stat().st_size for p in images)

    _add_warnings(stats, sample)
    return stats


def _interim_root(dataset: str, variant: str) -> Path:
    from .frameindex import interim_root

    return interim_root(dataset, variant)


def _sample(records: Sequence[FrameRecord], limit: int, seed: int) -> list[FrameRecord]:
    if len(records) <= limit:
        return list(records)
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(records), size=limit, replace=False)
    return [records[int(i)] for i in sorted(indices)]


def _box_sides(
    boxes: Sequence[tuple[int, float, float, float, float]], width: int, height: int
) -> list[float]:
    return [
        float(np.sqrt(max(bw * width, 0.0) * max(bh * height, 0.0)))
        for _cls, _cx, _cy, bw, bh in boxes
    ]


def _add_warnings(stats: DatasetStats, sample: Sequence[FrameRecord]) -> None:
    """Turn the numbers into the sentences you actually want to read."""
    if stats.frames == 0:
        return

    if stats.dropped_boxes > 0:
        ratio = stats.dropped_boxes / max(stats.boxes + stats.dropped_boxes, 1)
        if ratio > 0.05:
            stats.warn(
                f"{stats.dropped_boxes:,} boxes were dropped during conversion "
                f"({ratio:.1%}). Above ~5% this usually means the annotation coordinate "
                f"convention was misread rather than the boxes being genuinely degenerate."
            )

    if stats.unknown_labels:
        stats.warn(
            f"Unmapped source labels were discarded: {stats.unknown_labels}. Add them to "
            f"classes.mapping in configs/datasets/registry.yaml."
        )

    median = stats.size_percentiles.get("median", 0.0)
    if median and median < 16:
        # The target shrinks further when the frame is letterboxed into a square
        # input, because the binding constraint is the frame's long edge.
        typical_edge = max((int(s.split("x")[0]) for s in stats.frame_sizes), default=1280)
        effective = median * min(1.0, 640.0 / max(typical_edge, 1))
        stats.warn(
            f"Median target is {median:.1f} px, which becomes roughly {effective:.1f} px "
            f"after letterboxing a {typical_edge} px-wide frame to imgsz=640. Below ~8 px no "
            f"detector in this project reads the object reliably. Enable tiling for this "
            f"dataset - see tiling_threshold_px in configs/matrix.yaml."
        )

    if not stats.provides_bird_negatives:
        stats.warn(
            "No bird annotations. This dataset cannot teach the model to reject birds, so "
            "precision measured on it is unfalsifiable. Always evaluate on dvb or mavvid "
            "before trusting a number from here."
        )

    if stats.empty_frames / max(stats.frames, 1) < 0.05:
        stats.warn(
            f"Only {stats.empty_frames / max(stats.frames, 1):.1%} of frames have zero boxes. "
            f"A detector trained mostly on positive frames rarely learns a usable "
            f"confidence floor, and the rule layer's confidence.initiate threshold has "
            f"nothing to calibrate against."
        )

    sizes = stats.frame_sizes
    if len(sizes) > 12:
        stats.warn(
            f"{len(sizes)} distinct frame resolutions. The detector will spend capacity on "
            f"resizing rather than on detection; consider capping ingest.max_dimension to "
            f"the most common size."
        )

    if stats.sequences < 10:
        stats.warn(
            f"Only {stats.sequences} sequences. Validation metrics will be extremely noisy "
            f"and a train/val split may not be meaningful."
        )


def format_stats(stats: DatasetStats) -> str:
    """Compact table for the CLI."""
    lines = [
        f"dataset        : {stats.dataset}/{stats.variant}",
        f"frames         : {stats.frames:,}  "
        f"(labelled {stats.labelled_frames:,}, empty {stats.empty_frames:,})",
        f"sequences      : {stats.sequences:,}   tracks: {stats.tracks:,}",
        f"boxes          : {stats.boxes:,}  ({stats.boxes_per_frame_mean:.2f} per frame)",
        f"modalities     : {stats.modalities}",
        f"classes        : {stats.class_counts}",
        f"bird sequences : {stats.bird_sequences:,}  "
        f"(bird negatives: {'yes' if stats.provides_bird_negatives else 'NO'})",
    ]
    if stats.size_percentiles:
        pct = stats.size_percentiles
        lines.append(
            f"target size px : median {pct['median']:.1f}  p05 {pct['p05']:.1f}  "
            f"p95 {pct['p95']:.1f}"
        )
    if stats.size_histogram:
        lines.append("")
        lines.append("box size distribution (sampled):")
        total = max(sum(stats.size_histogram.values()), 1)
        for key, value in stats.size_histogram.items():
            bar = "#" * int(30 * value / total)
            lines.append(f"  {key:<12} {value:>9,}  {bar}")

    if stats.frame_sizes and len(stats.frame_sizes) <= 6:
        lines.append("")
        lines.append("frame resolutions: " + ", ".join(f"{k} x{v:,}" for k, v in stats.frame_sizes.items()))
    elif stats.frame_sizes:
        top = sorted(stats.frame_sizes.items(), key=lambda kv: -kv[1])[:5]
        lines.append(
            f"frame resolutions: {len(stats.frame_sizes)} distinct, top: "
            + ", ".join(f"{k} x{v:,}" for k, v in top)
        )

    if stats.warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  ! {w}" for w in stats.warnings)
    return "\n".join(lines)


def save(stats: DatasetStats, dataset: str, variant: str) -> Path:
    from ..utils.paths import subdir

    return write_json(subdir("manifests") / f"stats_{dataset}_{variant}.json", stats.to_dict())


def for_dataset(dataset: str, variant: str = "full", **kwargs) -> DatasetStats:
    """Convenience: load an index and compute stats in one call."""
    records = load_index(dataset, variant)
    return compute(records, dataset=dataset, variant=variant, **kwargs)


def combo_summary(
    stats_by_dataset: dict[str, DatasetStats],
    unified_labels: Sequence[str] = ("drone", "bird"),
) -> str:
    """Cross-dataset table printed before every training run.

    The ``bird negatives`` column is the one to read. A combo whose sources are
    all ``NO`` will produce a precision number that means nothing.
    """
    header = (
        f"{'dataset':<10}{'frames':>10}{'seqs':>7}{'boxes':>11}"
        f"{'drone':>10}{'bird':>9}{'med px':>8}{'bird neg':>10}"
    )
    lines = [header, "-" * len(header)]
    for name in sorted(stats_by_dataset):
        s = stats_by_dataset[name]
        median = s.size_percentiles.get("median", 0.0)
        lines.append(
            f"{name:<10}{s.frames:>10,}{s.sequences:>7}{s.boxes:>11,}"
            f"{s.class_counts.get('drone', 0):>10,}"
            f"{s.class_counts.get('bird', 0):>9,}"
            f"{median:>8.1f}"
            f"{('yes' if s.provides_bird_negatives else 'NO'):>10}"
        )
    total_frames = sum(s.frames for s in stats_by_dataset.values())
    bird_positive = [n for n, s in stats_by_dataset.items() if s.provides_bird_negatives]
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':<10}{total_frames:>10,}")

    if not bird_positive:
        lines.append("")
        lines.append(
            "  ! NO SOURCE IN THIS COMBO CONTAINS BIRD ANNOTATIONS. Any precision "
            "reported\n"
            "    for this combo is unfalsifiable - evaluate on dvb or mavvid before "
            "drawing\n    conclusions."
        )
    else:
        lines.append("")
        lines.append(f"  bird negatives supplied by: {', '.join(bird_positive)}")
    del unified_labels
    return "\n".join(lines)