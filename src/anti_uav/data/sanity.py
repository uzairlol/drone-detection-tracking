"""Post-build invariants.

This module is the reason the pipeline can be trusted. Every check here
corresponds to a way the project could silently produce meaningless numbers, and
each one is fatal or loudly warned rather than logged in passing.

The three that have actually bitten projects like this:

1. **Split leakage** - the same sequence on both sides of the train/val boundary.
   Inflates mAP by double digits.
2. **A source with no bird negatives in a combo whose headline metric is
   precision.** Unfalsifiable precision.
3. **Targets too small to see.** MM-UAV at 12 px in a 640 letterbox is 6 px.
   Training on that produces a confidently wrong model.

Run ``anti-uav sanity`` between build and train. A non-zero exit code means do
not train yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..utils.logging import get_logger
from .frameindex import FrameRecord, load_index

log = get_logger(__name__)


class Severity(StrEnum):
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


@dataclass(slots=True)
class Check:
    name: str
    severity: Severity
    message: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.severity is not Severity.FAIL


@dataclass(slots=True)
class SanityReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, severity: Severity, message: str, detail: str = "") -> None:
        self.checks.append(Check(name, severity, message, detail))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity is Severity.WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def exit_code(self) -> int:
        return 1 if self.failures else 0


def run_all(
    records: Sequence[FrameRecord],
    *,
    combo: str,
    unified_labels: Sequence[str] = ("drone", "bird"),
    expect_datasets: Sequence[str] = (),
    min_val_sequences: int = 5,
    min_target_px: float = 8.0,
) -> SanityReport:
    """Every invariant check, in the order you want to read the failures."""
    report = SanityReport()

    _check_non_empty(report, records, combo)
    _check_sequences_populated(report, records, combo)
    _check_split_coverage(report, records, combo, min_val_sequences)
    _check_no_split_leakage(report, records)
    _check_expected_datasets(report, records, expect_datasets)
    _check_bird_negatives(report, records, unified_labels, combo)
    _check_class_index_range(report, records, len(unified_labels))
    _check_target_size(report, records, min_target_px, combo)
    _check_negative_fraction(report, records)
    _check_label_files_exist(report, records)
    _check_label_geometry(report, records, unified_labels)

    return report


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #


def _check_non_empty(report: SanityReport, records: Sequence[FrameRecord], combo: str) -> None:
    if not records:
        report.add(
            "index",
            Severity.FAIL,
            f"{combo}: the frame index is empty.",
            "Run: anti-uav ingest && anti-uav convert && anti-uav build",
        )
    else:
        report.add("index", Severity.OK, f"{combo}: {len(records):,} frames in the index.")


def _check_sequences_populated(
    report: SanityReport, records: Sequence[FrameRecord], combo: str
) -> None:
    groups = {r.group_key for r in records}
    per_group: dict[str, int] = {}
    for record in records:
        per_group[record.group_key] = per_group.get(record.group_key, 0) + 1

    singles = sum(1 for count in per_group.values() if count == 1)
    ratio = singles / max(len(groups), 1)

    if len(groups) < 10:
        report.add(
            "sequence_count",
            Severity.FAIL,
            f"{combo}: only {len(groups)} sequences. Too few to split or to trust a metric.",
            "Check that the converter resolved sequence ids from the source layout.",
        )
    elif ratio > 0.5:
        report.add(
            "sequence_count",
            Severity.WARN,
            f"{combo}: {ratio:.0%} of sequences contain a single frame.",
            "Sequence grouping is probably wrong (MAV-VID's flat layout is the usual "
            "cause). Single-frame sequences cannot leak, but they also give the tracker "
            "nothing to follow.",
        )
    else:
        report.add(
            "sequence_count",
            Severity.OK,
            f"{combo}: {len(groups):,} sequences, {len(records) / max(len(groups), 1):.0f} "
            f"frames each.",
        )


def _check_split_coverage(
    report: SanityReport, records: Sequence[FrameRecord], combo: str, minimum: int
) -> None:
    unsplit = sum(1 for r in records if not r.split)
    if unsplit:
        report.add(
            "splits_assigned",
            Severity.FAIL,
            f"{combo}: {unsplit:,} frames have no split assigned.",
            "Run: anti-uav splits --group-by sequence",
        )

    val_groups = {r.group_key for r in records if r.split == "val"}
    train_groups = {r.group_key for r in records if r.split == "train"}

    if not val_groups:
        report.add(
            "val_split",
            Severity.FAIL,
            f"{combo}: the validation split is empty.",
        )
    elif len(val_groups) < minimum:
        report.add(
            "val_split",
            Severity.WARN,
            f"{combo}: only {len(val_groups)} validation sequences.",
            f"Metrics from fewer than ~20 sequences move by more than a point between "
            f"runs. Expect the run-to-run delta to exceed any model difference you are "
            f"trying to measure.",
        )
    else:
        report.add(
            "val_split",
            Severity.OK,
            f"{combo}: {len(val_groups):,} val sequences, {len(train_groups):,} train sequences.",
        )

    val_frames = sum(1 for r in records if r.split == "val")
    ratio = val_frames / max(len(records), 1)
    if ratio and not 0.05 <= ratio <= 0.4:
        report.add(
            "val_ratio",
            Severity.WARN,
            f"{combo}: validation is {ratio:.0%} of frames.",
            "Outside 5-40% the estimate is either too noisy or too slow to train against.",
        )


def _check_no_split_leakage(report: SanityReport, records: Sequence[FrameRecord]) -> None:
    """A sequence must appear in exactly one split. This is the big one."""
    by_group: dict[str, set[str]] = {}
    for record in records:
        if record.split:
            by_group.setdefault(record.group_key, set()).add(record.split)

    leaking = {k: sorted(v) for k, v in by_group.items() if len(v) > 1}
    if leaking:
        sample = ", ".join(f"{k}->{v}" for k, v in list(leaking.items())[:5])
        report.add(
            "split_leakage",
            Severity.FAIL,
            f"{len(leaking)} sequence(s) appear in more than one split.",
            f"{sample}. Adjacent frames of a video are near-identical, so validation "
            f"metrics from this split are inflated and meaningless. Re-run with "
            f"`--group-by sequence`.",
        )
    else:
        report.add(
            "split_leakage",
            Severity.OK,
            f"No sequence appears in more than one split ({len(by_group)} checked).",
        )


def _check_expected_datasets(
    report: SanityReport, records: Sequence[FrameRecord], expected: Sequence[str]
) -> None:
    present = sorted({r.dataset for r in records})
    missing = sorted(set(expected) - set(present))
    if missing:
        report.add(
            "expected_sources",
            Severity.FAIL,
            f"Combo declares {sorted(expected)} but the index only contains {present}.",
            f"Missing: {missing}. Either the download failed or `anti-uav build` skipped "
            f"an unavailable source - check the build log before training on a partial combo.",
        )
    elif expected:
        report.add(
            "expected_sources", Severity.OK, f"All declared sources present: {present}."
        )


def _check_bird_negatives(
    report: SanityReport,
    records: Sequence[FrameRecord],
    unified_labels: Sequence[str],
    combo: str,
) -> None:
    bird_index = unified_labels.index("bird") if "bird" in unified_labels else None
    if bird_index is None:
        return

    sources_with_birds = {
        r.dataset for r in records if bird_index in r.class_ids
    }
    if not sources_with_birds:
        report.add(
            "bird_negatives",
            Severity.WARN,
            f"{combo}: no source contributes bird annotations.",
            "Precision measured on this combo is unfalsifiable. Evaluate on dvb or "
            "mavvid before concluding anything about false positives.",
        )
        return

    val_bird = any(
        bird_index in r.class_ids and r.split == "val" for r in records
    )
    if not val_bird:
        report.add(
            "bird_negatives",
            Severity.WARN,
            f"{combo}: bird annotations exist ({', '.join(sorted(sources_with_birds))}) but "
            f"none are in the validation split.",
            "Bird recall cannot be measured. This is usually a split-stratification "
            "problem - the val split needs sequences that contain birds.",
        )
    else:
        report.add(
            "bird_negatives",
            Severity.OK,
            f"Bird negatives available from {', '.join(sorted(sources_with_birds))}, "
            f"including in val.",
        )


def _check_class_index_range(
    report: SanityReport, records: Sequence[FrameRecord], n_classes: int
) -> None:
    bad: dict[int, int] = {}
    for record in records:
        for class_id in record.class_ids:
            if not 0 <= class_id < n_classes:
                bad[class_id] = bad.get(class_id, 0) + 1
    if bad:
        report.add(
            "class_index_range",
            Severity.FAIL,
            f"Class ids outside [0, {n_classes}): {bad}.",
            "The class mapping and the label files disagree. Usually means one dataset was "
            "converted with a different unified_labels ordering.",
        )
    else:
        report.add("class_index_range", Severity.OK, f"All class ids within [0, {n_classes}).")


def _check_target_size(
    report: SanityReport, records: Sequence[FrameRecord], minimum_px: float, combo: str
) -> None:
    """Sample the labels on disk and measure the real target scale.

    Read from disk rather than trusted from the index, because the index does not
    store box geometry - this doubles as a check that the label files actually
    contain what the index claims.
    """
    import math
    import statistics

    from .harmonize import read_yolo_label

    sample = records[:: max(1, len(records) // 3000)]
    sizes: list[float] = []
    for record in sample:
        if not record.width or not record.height:
            continue
        path = _label_path(record)
        if path is None or not path.is_file():
            continue
        boxes = read_yolo_label(
            path, width=record.width, height=record.height, strict=False
        )
        sizes.extend(
            math.sqrt(max(bw * record.width, 0.0) * max(bh * record.height, 0.0))
            for _cls, _cx, _cy, bw, bh in boxes
        )

    if not sizes:
        report.add(
            "target_size",
            Severity.INFO,
            "No boxes available to measure target scale.",
        )
        return

    tiny = sum(1 for s in sizes if s < minimum_px)
    ratio = tiny / len(sizes)
    median = statistics.median(sizes)
    if ratio > 0.5:
        report.add(
            "target_size",
            Severity.WARN,
            f"{combo}: {ratio:.0%} of {len(sizes):,} sampled targets are under "
            f"{minimum_px:.0f} px (median {median:.1f} px).",
            "At that scale a 640 px letterbox shrinks the object to a few pixels and no "
            "detector reads it reliably. Enable tiling for this source "
            "(tiling_threshold_px in configs/matrix.yaml) or accept that this source "
            "contributes almost nothing.",
        )
    else:
        report.add(
            "target_size",
            Severity.OK,
            f"Median target {median:.1f} px, {ratio:.0%} below {minimum_px:.0f} px.",
        )


def _check_negative_fraction(report: SanityReport, records: Sequence[FrameRecord]) -> None:
    negatives = sum(1 for r in records if r.box_count == 0)
    ratio = negatives / max(len(records), 1)
    if ratio < 0.05:
        report.add(
            "negative_frames",
            Severity.WARN,
            f"Only {ratio:.1%} of frames have zero annotations.",
            "A detector trained with almost no negatives learns no usable confidence "
            "floor, and rules.confidence.initiate in drone_rules.yaml has nothing to "
            "calibrate against.",
        )
    else:
        report.add("negative_frames", Severity.OK, f"{ratio:.1%} of frames are negatives.")


def _check_label_files_exist(report: SanityReport, records: Sequence[FrameRecord]) -> None:
    missing = 0
    for record in records:
        path = _label_path(record)
        if path is None or not path.is_file():
            missing += 1
    if missing:
        report.add(
            "label_files",
            Severity.FAIL,
            f"{missing:,} of {len(records):,} frames have no label file on disk.",
            "A missing label file and an empty label file mean different things: empty is "
            "a legitimate negative, missing is data loss. Re-run `anti-uav build`.",
        )
    else:
        report.add("label_files", Severity.OK, "Every indexed frame has a label file.")


def _check_label_geometry(
    report: SanityReport, records: Sequence[FrameRecord], unified_labels: Sequence[str]
) -> None:
    """Sample the labels on disk and confirm the normalised geometry is sane."""
    sample = records[:: max(1, len(records) // 2000)]
    out_of_range = 0
    degenerate = 0
    checked = 0

    for record in sample:
        path = _label_path(record)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 5:
                    degenerate += 1
                    continue
                try:
                    values = [float(v) for v in parts[1:5]]
                except ValueError:
                    degenerate += 1
                    continue
                checked += 1
                cx, cy, bw, bh = values
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    out_of_range += 1
                if bw <= 0.0 or bh <= 0.0:
                    degenerate += 1

    if checked == 0:
        report.add(
            "label_geometry", Severity.INFO, "No label rows were available to sample."
        )
        return

    if out_of_range:
        ratio = out_of_range / checked
        report.add(
            "label_geometry",
            Severity.FAIL,
            f"{out_of_range:,} of {checked:,} sampled rows have a centre outside [0, 1].",
            f"{ratio:.1%} of labels are malformed. YOLO normalises to [0, 1]; if the "
            f"converter wrote pixel coordinates, every box is wrong.",
        )
    elif degenerate:
        report.add(
            "label_geometry",
            Severity.WARN,
            f"{degenerate:,} of {checked:,} sampled rows are malformed.",
        )
    else:
        report.add(
            "label_geometry",
            Severity.OK,
            f"{checked:,} sampled label rows are normalised and non-degenerate.",
        )


def _label_path(record: FrameRecord) -> Path | None:
    from .frameindex import interim_root

    return interim_root(record.dataset, record.variant) / record.label


def format_report(report: SanityReport) -> str:
    icons = {
        Severity.OK: "PASS",
        Severity.INFO: "INFO",
        Severity.WARN: "WARN",
        Severity.FAIL: "FAIL",
    }
    lines: list[str] = []
    for check in report.checks:
        lines.append(f"[{icons[check.severity]}] {check.name}: {check.message}")
        if check.detail and check.severity is not Severity.OK:
            for line in _wrap(check.detail, indent=8):
                lines.append(line)

    lines.append("")
    if report.ok:
        lines.append(
            f"sanity: {len(report.checks) - len(report.warnings)} checks passed, "
            f"{len(report.warnings)} warnings. Safe to train."
        )
    else:
        lines.append(
            f"sanity: {len(report.failures)} FAILURE(S), {len(report.warnings)} warnings. "
            f"DO NOT TRAIN until the failures are fixed."
        )
    return "\n".join(lines)


def _wrap(text: str, *, indent: int, width: int = 88) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if len(" ".join(current + [word])) > width - indent:
            lines.append(" " * indent + " ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" " * indent + " ".join(current))
    return lines


def check_combo(combo: str, datasets: Sequence[str], **kwargs) -> SanityReport:
    """Load each source's index and run every check across the union."""
    records: list[FrameRecord] = []
    for alias in datasets:
        records.extend(load_index(alias, kwargs.get("variants", {}).get(alias, "full")))
    return run_all(records, combo=combo, expect_datasets=datasets, **{
        k: v for k, v in kwargs.items() if k not in {"variants"}
    })