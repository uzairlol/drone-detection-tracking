"""Near-duplicate detection across the corpus.

Two independent reasons this matters here, one about honesty and one about
accuracy.

**Accuracy.** Two datasets shot at the same site, or the same dataset re-extracted
at a different stride, will share frames. When the *same* frame ends up in train
and in val, the val score is fiction. The sequence-level splitter in
:mod:`anti_uav.data.splits` prevents this within a dataset. This module catches
it *between* datasets, and within a dataset when the sequence grouping itself was
wrong (MAV-VID's flat layout is exactly that risk).

**Honesty.** A detector evaluated on frames it has effectively memorised reports a
number that will collapse the first time it sees a new camera. That is a
deployment failure, discovered after the training run you were proud of.

The comparison is perceptual (DCT pHash, 64-bit) rather than exact, because
re-extraction and JPEG recompression change bytes without changing content. Frames
within one video are *also* near-duplicates of each other, so the output is
reported as clusters and you choose the threshold - this module finds and reports,
it does not silently delete your data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.imaging import hamming, phash, read_image
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import subdir
from .frameindex import FrameRecord

log = get_logger(__name__)

#: Below this Hamming distance two frames are "the same frame".
DEFAULT_THRESHOLD = 4
#: Above this they are definitely different. The band between is reported as
#: "review" rather than guessed at.
REVIEW_THRESHOLD = 10


@dataclass(slots=True)
class DuplicateCluster:
    """Frames judged to be the same content."""

    hashes: list[str] = field(default_factory=list)
    members: list[tuple[str, str, str]] = field(default_factory=list)  # (dataset, seq, image)
    datasets: set[str] = field(default_factory=set)
    splits: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def crosses_datasets(self) -> bool:
        return len(self.datasets) > 1

    @property
    def crosses_splits(self) -> bool:
        return len(self.splits) > 1

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "datasets": sorted(self.datasets),
            "splits": sorted(self.splits),
            "crosses_datasets": self.crosses_datasets,
            "crosses_splits": self.crosses_splits,
            "members": [
                {"dataset": d, "sequence": s, "image": i} for d, s, i in self.members
            ],
        }


@dataclass(slots=True)
class DuplicateReport:
    total_frames: int
    hashed_frames: int
    clusters: int
    duplicate_frames: int
    cross_dataset_clusters: int
    cross_split_clusters: int
    threshold: int
    #: Sample clusters flagged for manual review (the grey band).
    review_candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duplicate_fraction(self) -> float:
        return self.duplicate_frames / max(self.hashed_frames, 1)

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "hashed_frames": self.hashed_frames,
            "clusters": self.clusters,
            "duplicate_frames": self.duplicate_frames,
            "duplicate_fraction": round(self.duplicate_fraction, 5),
            "cross_dataset_clusters": self.cross_dataset_clusters,
            "cross_split_clusters": self.cross_split_clusters,
            "threshold": self.threshold,
            "review_candidates": self.review_candidates,
            "warnings": self.warnings,
        }


def _hash_records(
    records: Sequence[FrameRecord],
    root_lookup,
    *,
    limit: int | None,
    seed: int,
) -> list[tuple[str, str, str, str, str]]:
    """``[(hash, dataset, sequence, image, split)]`` for the records we could hash."""
    out: list[tuple[str, str, str, str, str]] = []
    candidates = records
    if limit and len(records) > limit:
        generator = np.random.default_rng(seed)
        indices = generator.choice(len(records), size=limit, replace=False)
        candidates = [records[int(i)] for i in sorted(indices)]

    for record in candidates:
        path = root_lookup(record)
        if path is None or not path.is_file():
            continue
        try:
            digest = phash(read_image(path))
        except Exception as exc:  # noqa: BLE001
            log.debug("phash failed", extra={"file": record.image, "error": str(exc)})
            continue
        out.append((digest, record.dataset, record.sequence_id, record.image, record.split))

    return out


def find_duplicates(
    records: Sequence[FrameRecord],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    limit: int | None = 200_000,
    seed: int = 0,
    progress_every: int = 5_000,
) -> tuple[list[DuplicateCluster], DuplicateReport]:
    """Cluster frames by perceptual hash.

    Complexity is ``O(n * clusters)``, which is fine for the ~500k-frame ceiling
    this project targets. Beyond that, switch to LSH bucketing on the pHash bits
    - noted here rather than over-engineered for a scale we will not reach.
    """
    from .frameindex import interim_root

    root_cache: dict[tuple[str, str], Path] = {}

    def root_lookup(record: FrameRecord) -> Path | None:
        key = (record.dataset, record.variant)
        if key not in root_cache:
            root_cache[key] = interim_root(*key)
        return root_cache[key] / record.image

    hashes = _hash_records(records, root_lookup, limit=limit, seed=seed)

    clusters: list[DuplicateCluster] = []
    for digest, dataset, sequence, image, split in hashes:
        placed = False
        for cluster in clusters:
            if any(hamming(digest, known) <= threshold for known in cluster.hashes):
                cluster.hashes.append(digest)
                cluster.members.append((dataset, sequence, image))
                cluster.datasets.add(dataset)
                cluster.splits.add(split)
                placed = True
                break
        if not placed:
            clusters.append(
                DuplicateCluster(
                    hashes=[digest],
                    members=[(dataset, sequence, image)],
                    datasets={dataset},
                    splits={split},
                )
            )

        if len(hashes) % progress_every == 0:
            log.info(
                "dedup progress",
                extra={"hashed": len(hashes), "clusters": len(clusters)},
            )

    # A "cluster" of size 1 is a unique frame, not a duplicate.
    duplicates = [c for c in clusters if c.size > 1]
    duplicate_frames = sum(c.size for c in duplicates)
    cross_dataset = [c for c in duplicates if c.crosses_datasets]
    cross_split = [c for c in duplicates if c.crosses_splits]

    review = [
        c.to_dict()
        for c in duplicates
        if c.crosses_splits and len(c.members) <= 6
    ][:50]

    report = DuplicateReport(
        total_frames=len(records),
        hashed_frames=len(hashes),
        clusters=len(clusters),
        duplicate_frames=duplicate_frames,
        cross_dataset_clusters=len(cross_dataset),
        cross_split_clusters=len(cross_split),
        threshold=threshold,
        review_candidates=review,
    )

    if cross_split:
        report.warnings.append(
            f"{len(cross_split)} duplicate group(s) appear in BOTH train and val. Validation "
            f"metrics from this split are inflated. Fix with: "
            f"`anti-uav splits --strategy source` or raise --dedup-threshold to fold the "
            f"clusters into one sequence."
        )
    if cross_dataset:
        report.warnings.append(
            f"{len(cross_dataset)} duplicate group(s) span two datasets. Expected if the "
            f"same site was filmed twice; report it in your write-up, because it means the "
            f"cross-dataset generalisation number is partly measured on seen footage."
        )
    if report.duplicate_fraction > 0.25:
        report.warnings.append(
            f"{report.duplicate_fraction:.0%} of frames have a near-duplicate somewhere in "
            f"the corpus. Raise the ingest stride to cut redundant frames, or the runs will "
            f"be slower for no gain."
        )

    return duplicates, report


def iter_dataset_records(datasets: Iterable[str], variants: dict[str, str]) -> list[FrameRecord]:
    """Load and concatenate the indexes for several datasets."""
    from .frameindex import load_index

    out: list[FrameRecord] = []
    for alias in datasets:
        variant = variants.get(alias, "full")
        out.extend(load_index(alias, variant))
    return out


def save_report(report: DuplicateReport, name: str = "dedup") -> Path:
    return write_json(subdir("manifests") / f"{name}.json", report.to_dict())


def format_report(report: DuplicateReport, clusters: Sequence[DuplicateCluster]) -> str:
    lines = [
        f"frames hashed   : {report.hashed_frames:,} of {report.total_frames:,}",
        f"threshold       : hamming <= {report.threshold}",
        f"clusters        : {report.clusters:,} unique + {len(clusters):,} duplicated",
        f"duplicate frames: {report.duplicate_frames:,} ({report.duplicate_fraction:.2%})",
        f"cross-dataset   : {report.cross_dataset_clusters:,}",
        f"cross-split     : {report.cross_split_clusters:,}",
    ]

    worst = sorted(
        (c for c in clusters if c.size > 1),
        key=lambda c: -c.size,
    )[:5]
    if worst:
        lines.append("")
        lines.append("largest duplicate groups:")
        for cluster in worst:
            datasets = ", ".join(sorted(cluster.datasets))
            splits = ", ".join(sorted(s for s in cluster.splits if s))
            lines.append(
                f"  {cluster.size:>4} frames  datasets={datasets:<24} splits={splits}"
            )

    if report.warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  ! {w}" for w in report.warnings)
    return "\n".join(lines)