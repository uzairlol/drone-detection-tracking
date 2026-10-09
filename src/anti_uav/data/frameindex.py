"""The normalised intermediate representation.

Every dataset, whatever its native layout, is converted into exactly one shape::

    data/interim/<dataset>/<variant>/
        frames/<sequence_id>/<modality>/<stem>.jpg
        labels/<sequence_id>/<modality>/<stem>.txt      # YOLO: cls cx cy w h (normalised)
        mot/<sequence_id>/<modality>.txt                # MOT GT, when available
        index.jsonl                                      # one FrameRecord per line
        convert_report.json

Normalising first and grouping second is deliberate. Splitting correctly requires
grouping frames by *sequence*, and a regex over a publisher's original filenames
silently changes meaning when a folder is renamed. The converter's job is to
resolve a stable ``sequence_id`` once, and everything downstream reads that.

``FrameRecord`` is written to ``index.jsonl`` as JSON, so the index survives a
process restart and can be inspected without re-running conversion.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import append_jsonl, read_jsonl
from ..utils.logging import get_logger

log = get_logger(__name__)

INDEX_FILENAME = "index.jsonl"
REPORT_FILENAME = "convert_report.json"

#: Frame filename stem: 6-digit zero-padded, sorts naturally and keeps
#: ``Path.glob`` ordering lexicographic == numeric.
STEM_WIDTH = 6


@dataclass(slots=True)
class FrameRecord:
    """One frame, one modality, one sequence, one label file."""

    #: Stable sequence identifier. The unit that must never straddle a split.
    sequence_id: str
    #: Frame's position within its sequence.
    frame_index: int
    #: ``rgb`` | ``ir`` | ``event``
    modality: str
    #: Source dataset alias.
    dataset: str
    #: Variant, e.g. ``300`` or ``subset``.
    variant: str
    #: Paths are stored relative to the interim root, so the whole tree can move.
    image: str
    label: str
    width: int = 0
    height: int = 0
    #: Unified class indices present in this frame's label file.
    class_ids: list[int] = field(default_factory=list)
    #: Source label names as they appeared in the original annotation.
    source_labels: list[str] = field(default_factory=list)
    box_count: int = 0
    #: MOT identity, when the source provides one.
    track_id: int | None = None
    #: Per-object visibility ratio in [0, 1]. Anti-UAV and MM-UAV only.
    visibility: list[float] = field(default_factory=list)
    #: False for frames the publisher explicitly marks as target-absent. These
    #: are valid negatives and must stay in the dataset.
    target_present: bool = True
    #: Boxes dropped during conversion because they were degenerate or outside
    #: the frame. Non-zero is a data-quality signal worth surfacing.
    dropped_boxes: int = 0
    #: Tiles derived from this frame at build time. Populated by data/build.py.
    tiles: list[str] = field(default_factory=list)
    #: Assigned by data/splits.py: ``train`` | ``val`` | ``test`` | ``""``.
    split: str = ""
    notes: str = ""

    @property
    def group_key(self) -> str:
        """Key that must not straddle a train/val boundary."""
        return f"{self.sequence_id}/{self.modality}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameRecord:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class ConvertReport:
    """What a conversion actually produced, plus everything that went wrong.

    ``unknown_labels`` and ``dropped_boxes`` are the two fields worth reading
    before trusting a run: an unmapped source label means a whole class was
    silently discarded, and a high drop count means the coordinate system was
    probably misinterpreted.
    """

    dataset: str
    variant: str
    started_at: str = ""
    duration_s: float = 0.0
    frames_written: int = 0
    boxes_written: int = 0
    boxes_dropped: int = 0
    sequences: int = 0
    images_seen: int = 0
    images_without_labels: int = 0
    class_histogram: dict[str, int] = field(default_factory=dict)
    unknown_labels: dict[str, int] = field(default_factory=dict)
    sequences_seen: dict[str, int] = field(default_factory=dict)
    frame_size_histogram: dict[str, int] = field(default_factory=dict)
    #: The normalised label mapping actually applied, for the record.
    class_mapping_applied: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Set when the converter found a workable configuration and produced output.
    ok: bool = False

    def note(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def fail(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def interim_root(dataset: str, variant: str = "full") -> Path:
    """``data/interim/<dataset>/<variant>``."""
    from ..utils.paths import subdir

    return subdir("interim") / dataset / variant


def frames_dir(dataset: str, variant: str = "full") -> Path:
    return interim_root(dataset, variant) / "frames"


def labels_dir(dataset: str, variant: str = "full") -> Path:
    return interim_root(dataset, variant) / "labels"


def mot_dir(dataset: str, variant: str = "full") -> Path:
    return interim_root(dataset, variant) / "mot"


def index_path(dataset: str, variant: str = "full") -> Path:
    return interim_root(dataset, variant) / INDEX_FILENAME


def report_path(dataset: str, variant: str = "full") -> Path:
    return interim_root(dataset, variant) / REPORT_FILENAME


def stem_for(frame_index: int) -> str:
    """``7`` -> ``'000007'``."""
    return f"{frame_index:0{STEM_WIDTH}d}"


def write_index(dataset: str, variant: str, records: Iterable[FrameRecord]) -> Path:
    """Write the index atomically, sorted for reproducibility."""
    path = index_path(dataset, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    ordered = sorted(records, key=lambda r: (r.sequence_id, r.modality, r.frame_index))
    with tmp.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(
                __import__("json").dumps(record.to_dict(), ensure_ascii=False) + "\n"
            )
    tmp.replace(path)
    log.info("wrote frame index", extra={"dataset": dataset, "records": len(ordered)})
    return path


def append_record(dataset: str, variant: str, record: FrameRecord) -> None:
    """Append one record. Used by long-running ingest that may be interrupted."""
    append_jsonl(index_path(dataset, variant), record.to_dict())


def load_index(dataset: str, variant: str = "full") -> list[FrameRecord]:
    """Read the index. Returns ``[]`` when conversion has not run."""
    records = [FrameRecord.from_dict(row) for row in read_jsonl(index_path(dataset, variant))]
    log.debug("loaded frame index", extra={"dataset": dataset, "records": len(records)})
    return records


def load_report(dataset: str, variant: str = "full") -> ConvertReport | None:
    import json

    path = report_path(dataset, variant)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    known = {f for f in ConvertReport.__slots__}  # type: ignore[attr-defined]
    return ConvertReport(**{k: v for k, v in data.items() if k in known})


def write_report(dataset: str, variant: str, report: ConvertReport) -> Path:
    import json

    path = report_path(dataset, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, default=str)
        handle.write("\n")
    tmp.replace(path)
    return path


def iter_sequences(records: Iterable[FrameRecord]) -> Iterator[tuple[str, list[FrameRecord]]]:
    """Group records by ``group_key`` (sequence + modality), preserving order."""
    buckets: dict[str, list[FrameRecord]] = {}
    for record in records:
        buckets.setdefault(record.group_key, []).append(record)
    for key in sorted(buckets):
        yield key, buckets[key]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 60) -> str:
    """Lowercase, underscore-separated, filesystem-safe, deterministic.

    Sequence ids end up as directory names on Windows, so path separators,
    colons and trailing dots have to go.
    """
    cleaned = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "unnamed"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")
    return cleaned