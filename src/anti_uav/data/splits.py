"""Leakage-safe train/val splitting.

**The problem this exists to solve.** Every dataset in this project is video.
Consecutive frames of a drone hovering on camera are near-identical - a per-frame
hash would call them the same image. So a naive random frame split puts frame 412
and frame 413 of the same sequence on opposite sides of the boundary, the
validation set contains a near-copy of most of the training set, and mAP comes
back inflated by 10-20 points. The number looks great and means nothing.

**The rule.** A ``group_key`` (sequence + modality) is the atomic unit. All frames
sharing one go to the same split. Not "usually" - always, enforced here and
re-checked by :mod:`anti_uav.data.sanity`.

**Stratification.** Groups are also stratified by ``(dataset, modality)`` so each
source contributes roughly its true share to each split. Without this, a combo
containing a 40k-frame dataset and a 10k-frame one can end up with the small
source almost entirely in train and its val set reduced to a handful of
sequences - which makes a cross-source comparison meaningless.

**The trap in the obvious fix.** Balancing frame *counts* is the wrong target.
Balancing *sequence counts* is right, because sequences are the independent
units. A 2000-frame sequence and a 60-frame sequence count the same, even though
one of them dominates the frame count. The objective below therefore weights
groups equally, and reports the resulting frame distribution so the imbalance is
visible rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.schema import SplitStrategy
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import combo_root
from ..utils.seed import rng_for
from .frameindex import FrameRecord

log = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")

#: A sequence shorter than this is not worth a whole val split, but it is still
#: too valuable to discard, so it is merged into the smallest valid group.
MIN_GROUP_FRAMES = 8


@dataclass(slots=True)
class GroupStats:
    frames: int
    boxes: int
    has_bird: bool
    datasets: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SplitReport:
    """Everything needed to judge whether a split is trustworthy."""

    combo: str
    strategy: str
    seed: int
    ratios: dict[str, float]
    total_groups: int = 0
    total_frames: int = 0
    groups_per_split: dict[str, int] = field(default_factory=dict)
    frames_per_split: dict[str, int] = field(default_factory=dict)
    boxes_per_split: dict[str, int] = field(default_factory=dict)
    #: ``{split: {dataset: frames}}`` - the table to eyeball before training.
    frames_per_split_per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    sequences_per_split_per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    bird_positive_groups: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Set when the requested ratios could not be honoured.
    degraded: bool = False

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def _group_key(record: FrameRecord) -> str:
    return record.group_key


def build_groups(records: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    """Bucket records by their atomic split unit."""
    groups: dict[str, list[FrameRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)
    return groups


def _stats_for(group: Sequence[FrameRecord]) -> GroupStats:
    return GroupStats(
        frames=len(group),
        boxes=sum(r.box_count for r in group),
        has_bird=any(1 in r.class_ids for r in group),
        datasets={r.dataset for r in group},
    )


def _stratum_of(group: Sequence[FrameRecord]) -> tuple[str, str]:
    """Stratification key: (dataset, modality)."""
    first = group[0]
    return (first.dataset, first.modality)


def apply_pattern_strategy(
    records: list[FrameRecord],
    pattern: str,
    *,
    holdout: str = "val",
) -> tuple[list[FrameRecord], SplitReport]:
    """Assign splits from a regex over the sequence id.

    Use this when the publisher's own split should be honoured: their grouping
    regex tells you which sequence belongs to train, and honouring it is the only
    way your number is comparable to their published one.
    """
    compiled = re.compile(pattern)
    out: list[FrameRecord] = []
    for record in records:
        match = compiled.match(record.sequence_id)
        record.split = holdout if match else "train"
        out.append(record)
    return out, SplitReport(
        combo="<pattern>",
        strategy="pattern",
        seed=0,
        ratios={holdout: 0.0},
        total_frames=len(out),
    )


def apply_source_strategy(
    records: list[FrameRecord],
    held_out: Sequence[str],
) -> tuple[list[FrameRecord], SplitReport]:
    """Hold out entire datasets. This is the cross-dataset evaluation setup.

    ``held_out`` datasets get ``test``; everything else becomes train, with a
    stratified val carved out of it.
    """
    held = set(held_out)
    train_pool = [r for r in records if r.dataset not in held]
    test_pool = [r for r in records if r.dataset in held]

    assigned, report = _stratified_assign(train_pool, val_fraction=0.15, seed=0)
    for record in test_pool:
        record.split = "test"
    report.strategy = "source"
    report.ratios = {"val": 0.15, "test": len(test_pool) / max(len(records), 1)}
    report.warn(
        f"Cross-dataset holdout: {sorted(held)} are the test set. Numbers from this run "
        f"measure generalisation to unseen footage, not detection quality, and are "
        f"NOT comparable to the per-combo validation tables."
    )
    return [*assigned, *test_pool], report


def _stratified_assign(
    records: list[FrameRecord],
    *,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
    min_val_groups: int = 2,
) -> tuple[list[FrameRecord], SplitReport]:
    """The real implementation behind every strategy that needs balancing."""
    groups = build_groups(records)
    report = SplitReport(
        combo="<inline>",
        strategy="sequence",
        seed=seed,
        ratios={"val": val_fraction, "test": test_fraction},
        total_groups=len(groups),
        total_frames=len(records),
    )

    if not groups:
        return [], report

    strata: dict[tuple[str, str], list[str]] = {}
    for key, group in groups.items():
        strata.setdefault(_stratum_of(group), []).append(key)

    assignment: dict[str, str] = {}

    for stratum, keys in sorted(strata.items()):
        keys = sorted(keys)
        # Weight groups EQUALLY, not by frame count - sequences are the
        # independent units, so a 2000-frame sequence must not get 30x the vote
        # of a 60-frame one when deciding where the val boundary falls.
        order = rng_for(seed, "split", stratum).permutation(len(keys))
        shuffled = [keys[i] for i in order]

        large = [k for k in shuffled if len(groups[k]) >= MIN_GROUP_FRAMES]
        small = [k for k in shuffled if len(groups[k]) < MIN_GROUP_FRAMES]
        if not large:
            # Every sequence in this stratum is short. The "short sequences ride
            # along with train" rule below would then leave val empty and the
            # build would fail, so promote them all and split normally. This is
            # the right call for a genuinely small dataset.
            large, small = shuffled, []

        n_val = int(round(len(large) * val_fraction))
        n_test = int(round(len(large) * test_fraction))

        # Never leave a stratum with an empty val split; a 2-group stratum is the
        # smallest that yields a usable per-class number.
        if len(large) >= 3:
            n_val = max(n_val, min(min_val_groups, len(large) // 2))
        else:
            n_val = 0

        # Test takes from the tail of what val leaves, so they do not overlap.
        if n_val + n_test > len(large):
            n_test = max(0, len(large) - n_val)

        for key in large[:n_val]:
            assignment[key] = "val"
        for key in large[n_val : n_val + n_test]:
            assignment[key] = "test"
        for key in large[n_val + n_test :]:
            assignment[key] = "train"
        # Too-short sequences ride along with train. They still have labels, they
        # just do not have enough motion context to justify their own split.
        for key in small:
            assignment[key] = "train"

        if n_val == 0 and len(large) >= 1:
            report.warn(
                f"Stratum {stratum[0]}/{stratum[1]} has only {len(large)} sequence(s) of "
                f">= {MIN_GROUP_FRAMES} frames, so it contributed no validation frames. Any "
                f"metric for this source is train-set only and will read high."
            )

    out: list[FrameRecord] = []
    for record in records:
        record.split = assignment.get(_group_key(record), "train")
        out.append(record)

    summarise(out, groups, report)
    return out, report


def summarise(
    records: list[FrameRecord],
    groups: dict[str, list[FrameRecord]],
    report: SplitReport,
) -> SplitReport:
    """Fill the per-split tables and emit the warnings worth reading."""
    for name in SPLIT_NAMES:
        subset = [r for r in records if r.split == name]
        subset_keys = {_group_key(r) for r in subset}

        report.groups_per_split[name] = len(subset_keys)
        report.frames_per_split[name] = len(subset)
        report.boxes_per_split[name] = sum(r.box_count for r in subset)

        frames_by_source: dict[str, int] = {}
        keys_by_source: dict[str, set[str]] = {}
        for record in subset:
            frames_by_source[record.dataset] = frames_by_source.get(record.dataset, 0) + 1
            keys_by_source.setdefault(record.dataset, set()).add(_group_key(record))
        report.frames_per_split_per_source[name] = frames_by_source
        report.sequences_per_split_per_source[name] = {
            dataset: len(keys) for dataset, keys in keys_by_source.items()
        }

        report.bird_positive_groups[name] = sum(
            1 for key in subset_keys if _stats_for(groups[key]).has_bird
        )

    val_groups = report.groups_per_split.get("val", 0)
    if val_groups < 5:
        report.warn(
            f"The validation split has only {val_groups} sequences. Metrics from fewer "
            f"than ~20 sequences have a confidence interval too wide to compare two runs. "
            f"Add data or accept the noise."
        )

    bird_val = report.bird_positive_groups.get("val", 0)
    if bird_val == 0:
        report.warn(
            "No validation sequence contains bird annotations. Bird recall is undefined "
            "for this split, which means a model can look perfect while having learned "
            "nothing about false positives - the exact failure this project exists to fix."
        )

    return report


def split_records(
    records: list[FrameRecord],
    *,
    strategy: SplitStrategy = SplitStrategy.SEQUENCE,
    val_fraction: float = 0.15,
    test_fraction: float = 0.0,
    seed: int = 0,
    held_out_sources: Sequence[str] = (),
    pattern: str | None = None,
    combo: str = "<inline>",
) -> tuple[list[FrameRecord], SplitReport]:
    """Assign a split to every record. Returns the records plus a report.

    ``SEQUENCE`` is the default and the one to use for training. ``FILE`` is
    available for completeness but must not be used on video-derived data.
    """
    if strategy is SplitStrategy.SOURCE:
        return apply_source_strategy(records, held_out_sources or ("mmuav",))

    if strategy is SplitStrategy.PATTERN and pattern:
        return apply_pattern_strategy(records, pattern)

    if strategy is SplitStrategy.FILE:
        log.warning(
            "FILE strategy selected on video-derived data. Adjacent frames will land in "
            "different splits and validation metrics will be inflated. This is only correct "
            "for genuinely static, independently captured image sets.",
            extra={"combo": combo},
        )
        ordered = sorted(records, key=lambda r: (r.dataset, r.sequence_id, r.frame_index))
        cutoff = int(len(ordered) * (1.0 - val_fraction))
        for index, record in enumerate(ordered):
            record.split = "train" if index < cutoff else "val"
        groups = build_groups(ordered)
        report = SplitReport(
            combo=combo,
            strategy="file",
            seed=seed,
            ratios={"val": val_fraction},
            total_groups=len(groups),
            total_frames=len(ordered),
        )
        report.warn(
            "Frame-level split on video data. Reported metrics are upper bounds, not "
            "estimates."
        )
        # `summarise` fills the group's stats into the report and returns it;
        # the caller wants the records back, and `ordered` is already carrying
        # the split assignments.
        return ordered, summarise(ordered, groups, report)

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    assigned, report = _stratified_assign(
        records,
        val_fraction=val_fraction,
        seed=seed,
        test_fraction=test_fraction,
    )
    report.combo = combo
    report.strategy = "sequence"
    return assigned, report


def persist(records: list[FrameRecord], report: SplitReport, combo_slug: str) -> dict[str, str]:
    """Write the split back into the index and save the report.

    Two artifacts: ``index.jsonl`` gains a ``split`` field on every record, and
    ``splits.json`` records the exact assignment table plus the report. The
    second one is what makes a val number reproducible months later - you can
    recover which frames were in val without re-running the splitter.
    """
    root = combo_root(combo_slug)
    from .frameindex import write_index

    # Group records by their source dataset+variant so each interim tree keeps
    # its own index, but with splits applied.
    by_source: dict[tuple[str, str], list[FrameRecord]] = {}
    for record in records:
        by_source.setdefault((record.dataset, record.variant), []).append(record)

    for (dataset, variant), subset in by_source.items():
        write_index(dataset, variant, subset)

    assignment = {
        record.group_key: record.split
        for record in sorted(records, key=lambda r: (r.dataset, r.group_key))
    }
    report_path = write_json(
        root / "splits.json",
        {
            "combo": combo_slug,
            "strategy": report.strategy,
            "seed": report.seed,
            "ratios": report.ratios,
            "groups_per_split": report.groups_per_split,
            "frames_per_split": report.frames_per_split,
            "boxes_per_split": report.boxes_per_split,
            "frames_per_split_per_source": report.frames_per_split_per_source,
            "sequences_per_split_per_source": report.sequences_per_split_per_source,
            "bird_positive_groups": report.bird_positive_groups,
            "warnings": report.warnings,
            "assignment": assignment,
        },
    )

    _write_split_txt(root, records)
    log.info(
        "splits persisted",
        extra={"combo": combo_slug, "report": str(report_path)},
    )
    return {"report": str(report_path)}


def _write_split_txt(root: Path, records: list[FrameRecord]) -> Path:
    """``<split>/<dataset>_<sequence>_<modality>.txt`` lists, the ultralytics format.

    Pointing a ``data.yaml`` at text files of image paths avoids materialising a
    symlinked image tree, which matters when one combo references 400k frames.
    """
    buckets: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for record in records:
        buckets.setdefault(record.split or "train", []).append(record.image)

    for name, paths in buckets.items():
        if not paths:
            continue
        target = root / f"{name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(sorted(set(paths))) + "\n", encoding="utf-8")
    return root / "splits"


def format_report(report: SplitReport) -> str:
    """Human-readable table for the CLI."""
    lines = [
        f"split strategy : {report.strategy}",
        f"groups (seq/mod): {report.total_groups}",
        f"frames          : {report.total_frames}",
        "",
        f"{'split':<6}{'sequences':>10}{'frames':>10}{'boxes':>10}  {'with bird':>10}",
    ]
    for name in SPLIT_NAMES:
        lines.append(
            f"{name:<6}"
            f"{report.groups_per_split.get(name, 0):>10}"
            f"{report.frames_per_split.get(name, 0):>10}"
            f"{report.boxes_per_split.get(name, 0):>10}"
            f"  {report.bird_positive_groups.get(name, 0):>10}"
        )

    if report.frames_per_split_per_source:
        lines.append("")
        lines.append("frames per source")
        datasets = sorted({d for table in report.frames_per_split_per_source.values() for d in table})
        header = f"{'split':<6}" + "".join(f"{d:>16}" for d in datasets)
        lines.append(header)
        for name in SPLIT_NAMES:
            table = report.frames_per_split_per_source.get(name, {})
            lines.append(f"{name:<6}" + "".join(f"{table.get(d, 0):>16,}" for d in datasets))

    if report.warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  ! {w}" for w in report.warnings)
    return "\n".join(lines)


def val_fraction_needed(records: list[FrameRecord], target_val_sequences: int = 20) -> float:
    """Suggest a val fraction that yields a usable number of val sequences.

    20 val sequences is roughly where per-class mAP stops moving by more than a
    point between runs, which is the threshold for being able to rank two models.
    """
    groups = build_groups(records)
    if not groups:
        return 0.15
    total = len(groups)
    return float(np.clip(target_val_sequences / total, 0.05, 0.5))