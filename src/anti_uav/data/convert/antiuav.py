"""Anti-UAV converter.

Anti-UAV is the awkward one of the four, and the awkwardness is the interesting
part, so it is spelled out here rather than papered over.

**Two modalities, unaligned.** Anti-UAV300 ships an RGB video and a thermal IR
video per sequence, and the publisher states explicitly that they are *not*
pixel-registered. So they are converted as two independent sequences -
``<seq>/rgb`` and ``<seq>/ir`` - never as a fused pair. The global tracker
treats them as separate observations; only the rule layer reasons across
cameras.

**Visibility flags are load-bearing.** Anti-UAV annotates every frame with a
``v`` flag: ``0`` means the target is absent from the scene, and a positive value
is the fraction of the target still visible (partial occlusion). This is the
reason Anti-UAV is in the matrix at all: it is the only source that teaches a
model what an *empty sky* looks like. Those frames become
``target_present=False`` records and are kept.

**Track id is the single-target id.** Anti-UAV is single-object tracking, so the
id is almost always 1. It is still preserved so the MOT-style tracker evaluation
has ground truth to score against.

JSON shapes seen in the wild for one sequence's ``<seq>.json``:

    {"frames": {"1": [{"xmin":..,"ymin":..,"xmax":..,"ymax":..,"v":1.0}]}}
    {"frame_1": [...]}
    [{"frame_id": 1, "objects": [...]}]

All three are handled; anything else is a hard error naming the shape it found,
because silently producing zero boxes from an unrecognised layout is the worst
possible failure mode here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ...utils.logging import get_logger
from ..frameindex import FrameRecord
from .video_common import VIDEO_SUFFIXES, Det, VideoConverter

log = get_logger(__name__)

_ID_IN_NAME = re.compile(r"(?:^|[_-])(\d{3,6})(?=[_.-]|$)")


class AntiUavConverter(VideoConverter):
    """Base: shared JSON walking and IR discovery."""

    alias = "antiuav"
    primary_modality = "rgb"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.drone_index = self.spec.classes.unified_labels.index("drone")

    def sequence_id_for(self, video: Path) -> str:
        from ..frameindex import slugify

        parent = video.parent.name if video.parent != self.raw_root else video.stem
        base = re.sub(r"(?i)[_\- ]?(rgb|visible|vis|thermal|ir|infrared)$", "", parent)
        return slugify(base or parent)

    def modality_for(self, video: Path) -> str:
        lowered = video.name.lower()
        if any(token in lowered for token in ("thermal", "_ir", "-ir", "infrared", "ir.")):
            return "ir"
        return "rgb"

    # -- shared ------------------------------------------------------------- #

    def _convert(self) -> None:
        pairs = self._discover_pairs()
        if not pairs:
            raise FileNotFoundError(
                f"{self.alias}: found no (video, annotation) pairs under {self.raw_root}. "
                f"Expected one directory per sequence containing <seq>.mp4 and <seq>.json."
            )

        self.note(
            f"Converting {sum(1 for _, _, m in pairs if m == 'rgb')} RGB and "
            f"{sum(1 for _, _, m in pairs if m == 'ir')} IR streams as INDEPENDENT "
            f"sequences. Anti-UAV states its modalities are unaligned, so they are never "
            f"fused into one frame record."
        )

        for index, (sequence_dir, video, modality) in enumerate(pairs, start=1):
            sequence_id = self.sequence_id_for(video)
            try:
                annotations = self.parse_annotations(video, sequence_id)
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"{sequence_dir.name}/{modality}: annotation parse failed "
                    f"({type(exc).__name__}: {exc}); stream skipped"
                )
                if message not in self.report.errors:
                    self.report.errors.append(message)
                log.warning("annotation parse failed", extra={"sequence": sequence_id, "modality": modality})
                continue

            self._extract_as(video, sequence_id, modality, annotations)
            # Persist the visibility-aware MOT rows for this sequence/modality.
            self.flush_mot_rows(sequence_id=sequence_id, modality=modality)

            if index % 20 == 0 or index == len(pairs):
                log.info(
                    "stream progress",
                    extra={
                        "dataset": self.alias,
                        "done": index,
                        "total": len(pairs),
                        "frames": self.report.frames_written,
                    },
                )

    def _extract_as(
        self,
        video: Path,
        sequence_id: str,
        modality: str,
        annotations: dict[int, list[Det]],
    ) -> int:
        """Same as the base extractor but with an explicit modality label."""
        original = self.primary_modality
        self.primary_modality = modality
        try:
            return self._extract(video, sequence_id, annotations)
        finally:
            self.primary_modality = original

    def mot_row_for(
        self,
        record: FrameRecord,
        pixel_boxes: Sequence[tuple[float, float, float, float]],
    ) -> Sequence[tuple[float, ...]]:
        """Emit one MOT row per box, carrying this dataset's visibility flags.

        Why Anti-UAV needs this: ``visibility_aware`` scoring is switched on for
        exactly one dataset, and it reads the ``vis`` column of the MOT file. With
        no MOT file the metric loader falls back to the YOLO labels, which have no
        visibility field and hardcode it to 1.0 — so every frame looked fully
        visible and the one thing that makes this dataset worth scoring was lost at
        the index. Frames the publisher marks `v <= 0` stay in the index as
        labelled negatives, so their absence from these rows is itself the signal
        the metric needs.
        """
        if record.track_id is None or len(record.visibility) != len(pixel_boxes):
            return ()
        return tuple(
            (
                record.frame_index + 1,
                record.track_id,
                x1,
                y1,
                x2,
                y2,
                1,
                self.drone_index,
                visibility,
            )
            for (x1, y1, x2, y2), visibility in zip(pixel_boxes, record.visibility, strict=True)
        )

    def _discover_pairs(self) -> list[tuple[Path, Path, str]]:
        """Every ``(sequence_dir, video, modality)`` triple with an annotation."""
        out: list[tuple[Path, Path, str]] = []
        for sequence_dir in sorted(p for p in self.raw_root.rglob("*") if p.is_dir()):
            videos = [
                p for p in sorted(sequence_dir.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
            ]
            if not videos:
                continue
            if not self._has_annotation(sequence_dir):
                self.note(f"{sequence_dir.name}: videos present but no .json annotation; skipped")
                continue
            for video in videos:
                out.append((sequence_dir, video, self.modality_for(video)))
        return out

    def _has_annotation(self, sequence_dir: Path) -> bool:
        return any(p.suffix.lower() == ".json" for p in sequence_dir.iterdir() if p.is_file())

    def parse_annotations(self, video: Path, sequence_id: str) -> dict[int, list[Det]]:
        annotation = self._annotation_file(video)
        if annotation is None:
            raise FileNotFoundError(f"no .json annotation found for {video.name}")

        with annotation.open(encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)

        rows = self._iter_frame_rows(data, annotation)
        out: dict[int, list[Det]] = {}
        for frame_index, objects in rows:
            dets: list[Det] = []
            for obj in objects:
                det = self._to_det(obj)
                if det is not None:
                    dets.append(det)
            # Keep the key even when the frame has no visible target: that is a
            # labelled negative, which is the point of this dataset.
            out[frame_index] = dets
        return out

    # -- annotation format -------------------------------------------------- #

    def _annotation_file(self, video: Path) -> Path | None:
        candidate = video.with_suffix(".json")
        if candidate.is_file():
            return candidate
        for path in sorted(video.parent.iterdir()):
            if path.is_file() and path.suffix.lower() == ".json":
                return path
        return None

    def _iter_frame_rows(
        self, data: Any, annotation: Path
    ) -> Iterable[tuple[int, list[dict[str, Any]]]]:
        """Normalise the three observed JSON layouts into ``(frame, objects)``."""
        if isinstance(data, dict) and isinstance(data.get("frames"), dict):
            for key, objects in data["frames"].items():
                index = _int_or_none(key)
                if index is None:
                    continue
                yield index - 1, [o for o in objects if isinstance(o, dict)]
            return

        if isinstance(data, dict) and data and all(_int_or_none(k) is not None for k in data):
            for key, objects in data.items():
                index = _int_or_none(key)
                if index is None or not isinstance(objects, list):
                    continue
                yield index - 1, [o for o in objects if isinstance(o, dict)]
            return

        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                raw_index = row.get("frame_id", row.get("frame", row.get("frame_index")))
                index = _int_or_none(raw_index)
                if index is None:
                    continue
                objects = row.get("objects", row.get("labels", row.get("annotations", [])))
                if isinstance(objects, dict):
                    objects = [objects]
                yield index - 1, [o for o in objects if isinstance(o, dict)]
            return

        raise ValueError(
            f"Unrecognised Anti-UAV annotation layout in {annotation.name}: "
            f"{type(data).__name__}"
            + (f" with keys {list(data)[:5]}" if isinstance(data, dict) else "")
            + ". Expected one of: {\"frames\": {\"1\": [...]}}, {\"1\": [...]}, "
            "or a list of {frame_id, objects} rows."
        )

    def _to_det(self, obj: dict[str, Any]) -> Det | None:
        """One object dict -> :class:`Det`, honouring the visibility flag."""
        visibility = _float_or(obj.get("v"), _float_or(obj.get("visibility"), 1.0))
        if visibility <= 0.0:
            # Publisher says the target is not in this frame. This is data, not a
            # missing annotation - the record is emitted with target_present
            # False by the extractor.
            return None

        x1 = _first_float(obj, ("xmin", "x1", "left", "x"))
        y1 = _first_float(obj, ("ymin", "y1", "top", "y"))
        x2 = _first_float(obj, ("xmax", "x2", "right"))
        y2 = _first_float(obj, ("ymax", "y2", "bottom"))
        if None in (x1, y1, x2, y2):
            return None

        track_id = _int_or_none(obj.get("id", obj.get("track_id", obj.get("object_id"))))
        return Det(
            x1=float(x1),  # type: ignore[arg-type]  # narrowed by the `None in` check above
            y1=float(y1),  # type: ignore[arg-type]
            x2=float(x2),  # type: ignore[arg-type]
            y2=float(y2),  # type: ignore[arg-type]
            class_id=self.drone_index,
            track_id=track_id if track_id is not None else 1,
            visibility=visibility,
            present=True,
        )


class AntiUavVariantConverter(AntiUavConverter):
    """Alias kept so the CLI can name the variant explicitly."""


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(obj: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in obj:
            value = _float_or(obj[key], None)  # type: ignore[arg-type]
            if value is not None:
                return value
    return None


def discover_variants(root: Path) -> dict[str, int]:
    """Count videos per variant directory, so the CLI can report what arrived."""
    counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            key = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "root"
            counts[key] = counts.get(key, 0) + 1
    return counts