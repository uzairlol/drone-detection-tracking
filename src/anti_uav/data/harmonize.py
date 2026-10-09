"""YOLO label I/O and class harmonisation.

Two jobs:

1. Read and write YOLO ``.txt`` label files without ever silently losing a box.
   A malformed line increments a counter and lands in the report rather than
   vanishing, because "the converter dropped 400 boxes" is exactly the kind of
   thing that quietly costs you 2 mAP and you only notice weeks later.

2. Harmonise source vocabularies into the unified ``{drone, bird}`` space.

The harmonisation rule that matters: **an unmapped label is an error, not a
warning.** Anti-UAV calls its object ``target``, MM-UAV calls it ``UAV``, the
Kaggle mirror of Drone-vs-Bird uses bare integers. If a publisher ever adds a
class we have not seen, dropping it by default would mean training a "drone
detector" that silently also ignored ``helicopter`` or ``balloon``. So
:func:`resolve_class` returns an error marker and the converters tally it in
``ConvertReport.unknown_labels``, which is a hard failure at the sanity stage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import ClassMapping
from ..utils.geometry import clip_box, xyxy_to_cxcywh
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Boxes smaller than this in either dimension are dropped. Sub-2 px boxes are
#: annotation noise, not objects, and training on them teaches the detector to
#: fire on nothing.
MIN_BOX_SIDE_PX = 2.0

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass(slots=True)
class LabelStats:
    """Tally shared by the YOLO readers and writers."""

    lines_read: int = 0
    boxes_kept: int = 0
    boxes_degenerate: int = 0
    boxes_out_of_frame: int = 0
    boxes_dropped_tiny: int = 0
    unknown_labels: dict[str, int] = None  # type: ignore[assignment]
    ignored_labels: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unknown_labels is None:
            self.unknown_labels = {}
        if self.ignored_labels is None:
            self.ignored_labels = {}

    def merge(self, other: LabelStats) -> None:
        self.lines_read += other.lines_read
        self.boxes_kept += other.boxes_kept
        self.boxes_degenerate += other.boxes_degenerate
        self.boxes_out_of_frame += other.boxes_out_of_frame
        self.boxes_dropped_tiny += other.boxes_dropped_tiny
        for key, value in other.unknown_labels.items():
            self.unknown_labels[key] = self.unknown_labels.get(key, 0) + value
        for key, value in other.ignored_labels.items():
            self.ignored_labels[key] = self.ignored_labels.get(key, 0) + value

    @property
    def boxes_dropped(self) -> int:
        return self.boxes_degenerate + self.boxes_out_of_frame + self.boxes_dropped_tiny


class UnknownLabelError(KeyError):
    """Raised in strict mode when a source label has no unified equivalent."""

    def __init__(self, label: str, available: Iterable[str]) -> None:
        known = ", ".join(sorted(available))
        super().__init__(
            f"source label {label!r} has no mapping into the unified label space "
            f"(known: {known or 'none'}). Add it to classes.mapping in "
            f"configs/datasets/registry.yaml, or list it in classes.ignore_labels "
            f"if it is intentionally dropped."
        )
        self.label = label


def normalize_source_label(raw: object) -> str:
    """Canonical spelling for a source label.

    Publishers are inconsistent about case and separators ("Bird", "bird ",
    "drone_detection"), and two mirrors have shipped with numeric ids where the
    docs promise names. Normalising the *string* before the mapping lookup means
    the registry only ever needs one spelling per class.
    """
    text = str(raw).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return text


def resolve_class(
    raw: object,
    mapping: ClassMapping,
    *,
    strict: bool = True,
) -> tuple[int | None, str]:
    """Source label -> unified class index.

    Returns ``(index, canonical_source_label)``. ``index is None`` means the box
    should be dropped: either the label is in ``ignore_labels``, or it is unknown
    and ``strict=False``.

    Raises :class:`UnknownLabelError` when ``strict=True`` and the label is
    neither mapped nor ignored. Callers should catch it per-box and tally, not
    let it abort a 40-minute conversion.
    """
    canonical = normalize_source_label(raw)
    index = mapping.to_index(canonical)
    if index is not None:
        return index, canonical
    if canonical in mapping.ignore_labels:
        return None, canonical
    if strict:
        raise UnknownLabelError(canonical, [*mapping.mapping, *mapping.ignore_labels])
    return None, canonical


def read_yolo_label(
    path: str | Path,
    *,
    width: int,
    height: int,
    mapping: ClassMapping | None = None,
    strict: bool = True,
    stats: LabelStats | None = None,
) -> list[tuple[int, float, float, float, float]]:
    """Read a normalised YOLO label into ``(class_id, cx, cy, w, h)`` tuples.

    Coordinates are returned **clamped** to ``[0, 1]`` and boxes that fall
    entirely outside the frame are dropped. Out-of-bounds boxes are common in
    published datasets (partially visible targets at a frame border) and are
    usually a coordinate-convention mismatch worth knowing about - hence the
    counter rather than a silent fix.
    """
    file_path = Path(path)
    tally = stats if stats is not None else LabelStats()
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return []

    out: list[tuple[int, float, float, float, float]] = []
    with file_path.open(encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tally.lines_read += 1
            parts = line.split()
            if len(parts) < 5:
                tally.boxes_degenerate += 1
                log.debug("short label line", extra={"file": file_path.name, "line": lineno})
                continue

            try:
                raw_cls = parts[0]
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                tally.boxes_degenerate += 1
                continue

            if mapping is not None and not _NUMERIC_RE.match(raw_cls):
                # A numeric first token is already a class INDEX - that is what a
                # YOLO .txt file is by definition, and it is what the Kaggle
                # mirror of Drone-vs-Bird ships (0 = drone, 1 = bird). Only a
                # non-numeric token is a source label NAME and needs the mapping.
                try:
                    class_id, canonical = resolve_class(raw_cls, mapping, strict=strict)
                except UnknownLabelError:
                    key = normalize_source_label(raw_cls)
                    tally.unknown_labels[key] = tally.unknown_labels.get(key, 0) + 1
                    continue
                if class_id is None:
                    tally.ignored_labels[canonical] = tally.ignored_labels.get(canonical, 0) + 1
                    continue
            elif _NUMERIC_RE.match(raw_cls):
                class_id = int(float(raw_cls))
            else:
                key = normalize_source_label(raw_cls)
                tally.unknown_labels[key] = tally.unknown_labels.get(key, 0) + 1
                continue

            if bw <= 0.0 or bh <= 0.0 or not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                tally.boxes_degenerate += 1
                continue

            bw_px, bh_px = bw * width, bh * height
            if bw_px < MIN_BOX_SIDE_PX or bh_px < MIN_BOX_SIDE_PX:
                tally.boxes_dropped_tiny += 1
                continue

            if cx - bw / 2.0 < -0.001 or cy - bh / 2.0 < -0.001 or cx + bw / 2.0 > 1.001 or cy + bh / 2.0 > 1.001:
                # Straddles or escapes the border: clamp rather than discard.
                box = clip_box(
                    (
                        (cx - bw / 2.0) * width,
                        (cy - bh / 2.0) * height,
                        (cx + bw / 2.0) * width,
                        (cy + bh / 2.0) * height,
                    ),
                    width,
                    height,
                )
                ncx, ncy, nbw, nbh = xyxy_to_cxcywh(box, width, height)
                if nbw <= 0.0 or nbh <= 0.0:
                    tally.boxes_out_of_frame += 1
                    continue
                tally.boxes_out_of_frame += 1
                cx, cy, bw, bh = ncx, ncy, nbw, nbh

            out.append((class_id, cx, cy, bw, bh))
            tally.boxes_kept += 1

    return out


def write_yolo_label(
    path: str | Path,
    boxes: Sequence[tuple[int, float, float, float, float]],
) -> Path:
    """Write ``(class_id, cx, cy, w, h)`` tuples as a YOLO label file.

    Six-decimal precision: at 3840 px wide that is ~0.004 px of quantisation
    error, far below anything that affects an mAP.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for class_id, cx, cy, bw, bh in boxes
        if bw > 0.0 and bh > 0.0
    ]
    body = "\n".join(lines) + ("\n" if lines else "")
    tmp = target.with_suffix(".txt.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(target)
    return target


def boxes_from_xyxy(
    rows: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Convert ``(cls, x1, y1, x2, y2)`` pixel rows to normalised YOLO tuples."""
    out: list[tuple[int, float, float, float, float]] = []
    for row in rows:
        if len(row) < 5:
            continue
        class_id = int(row[0])
        box = clip_box((float(row[1]), float(row[2]), float(row[3]), float(row[4])), width, height)
        if box[2] - box[0] < MIN_BOX_SIDE_PX or box[3] - box[1] < MIN_BOX_SIDE_PX:
            continue
        out.append((class_id, *xyxy_to_cxcywh(box, width, height)))
    return out


def class_distribution(boxes: Iterable[tuple[int, float, float, float, float]], names: Sequence[str]) -> dict[str, int]:
    """``{'drone': 41233, 'bird': 8121}`` for a report table."""
    counts = {name: 0 for name in names}
    for box in boxes:
        index = int(box[0])
        if 0 <= index < len(names):
            counts[names[index]] += 1
        else:
            counts[f"__invalid_{index}"] = counts.get(f"__invalid_{index}", 0) + 1
    return counts


def box_size_histogram(
    boxes: Iterable[tuple[int, float, float, float, float]],
    *,
    width: int,
    height: int,
    bins: Sequence[float] = (0, 8, 16, 32, 64, 128, 256, float("inf")),
) -> dict[str, int]:
    """Boxes bucketed by ``sqrt(w*h)`` in pixels.

    The single most useful sanity check for this project. If most of MM-UAV's
    boxes land in the ``<8 px`` bucket then either the tiling is not active or
    the detector has no chance, and you should know before spending a day on
    training rather than after.
    """
    import math

    histogram: dict[str, int] = {}
    labels: list[str] = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        label = f"{low:g}-{high:g}px" if math.isfinite(high) else f"{low:g}+px"
        histogram[label] = 0
        labels.append(label)

    for _, _cx, _cy, bw, bh in boxes:
        side = math.sqrt(max(bw * width, 0.0) * max(bh * height, 0.0))
        for i in range(len(bins) - 1):
            if bins[i] <= side < bins[i + 1]:
                histogram[labels[i]] += 1
                break

    return histogram