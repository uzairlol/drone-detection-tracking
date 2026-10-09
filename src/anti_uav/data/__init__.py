"""Dataset pipeline: download -> ingest -> convert -> split -> build -> verify.

Stage order, and why:

1. ``download``  fetch raw artefacts, hash them into ``data/manifests``
2. ``convert``   normalise every layout into one interim tree
3. ``splits``    assign whole sequences to train/val - never individual frames
4. ``build``     materialise the combo, applying per-source tiling
5. ``sanity``    verify the invariants. Non-zero exit means do not train.

``ingest`` is folded into ``convert`` for the video datasets, because frame
extraction and annotation parsing have to agree on stride to produce correct
labels; splitting them across commands is how the off-by-one-frame class of bug
gets in.
"""

from __future__ import annotations

from .build import BuildReport, SourceBuild, build, resolve_tiling
from .dedup import DuplicateCluster, DuplicateReport, find_duplicates, iter_dataset_records
from .frameindex import (
    ConvertReport,
    FrameRecord,
    interim_root,
    iter_sequences,
    load_index,
    load_report,
    slugify,
)
from .harmonize import (
    LabelStats,
    UnknownLabelError,
    read_yolo_label,
    resolve_class,
    write_yolo_label,
)
from .sanity import SanityReport, Severity, run_all
from .splits import SplitReport, format_report as format_split_report, split_records
from .stats import DatasetStats, combo_summary, compute, format_stats

__all__ = [
    "BuildReport",
    "ConvertReport",
    "DatasetStats",
    "DuplicateCluster",
    "DuplicateReport",
    "FrameRecord",
    "LabelStats",
    "SanityReport",
    "Severity",
    "SourceBuild",
    "SplitReport",
    "UnknownLabelError",
    "build",
    "combo_summary",
    "compute",
    "find_duplicates",
    "format_split_report",
    "format_stats",
    "interim_root",
    "iter_dataset_records",
    "iter_sequences",
    "load_index",
    "load_report",
    "read_yolo_label",
    "resolve_class",
    "resolve_tiling",
    "run_all",
    "slugify",
    "split_records",
    "write_yolo_label",
]