"""MAV-VID converter.

The easiest of the four: the publisher already ships images with YOLO ``.txt``
labels, pre-split into ``train``/``val``. No video decoding, no annotation
parsing - this is a copy-and-renormalise job whose real work is resolving a stable
``sequence_id``.

The trap, and the reason this converter is not ten lines: MAV-VID's layout puts
every image in a flat directory per split, so nothing in the filename says which
video a frame came from. Grouping by split would put 30k frames of the same video
on both sides of the train/val boundary and inflate mAP badly.

So sequence recovery is layered, strongest signal first:

1. a per-video directory in the path (``video_003/``) - newer mirrors have these;
2. a video id embedded in the filename (``MultiUAV-001_000123.jpg``);
3. perceptual-hash clustering - consecutive frames of a video are near-identical,
   so a small Hamming distance means "same video".

Option 3 is a fallback and is called out in the report, because it can merge two
visually similar adjacent videos.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from ...utils.imaging import hamming, image_size, is_image, phash
from ...utils.logging import get_logger
from ..harmonize import LabelStats, UnknownLabelError, resolve_class
from .base import Converter

log = get_logger(__name__)

SPLIT_DIRS = ("train", "val", "valid", "validation", "test")
_VIDEO_DIR_TOKENS = ("video", "vid", "seq", "clip", "scene", "track")
#: Matches ``<something>-<digits>`` or ``<digits>`` used as a video identifier.
_VIDEO_IN_NAME = re.compile(r"(?:^|[_-])(\d{3,6})(?=[_.-]|$)")
#: Hamming distance below which two frames are considered the same video.
SAME_VIDEO_THRESHOLD = 4


def _label_candidates(root: Path, image: Path) -> list[Path]:
    """Locate the label file for an image under the layouts mirrors actually use."""
    relative = image.relative_to(root)
    parts = relative.parts

    if "images" in parts:
        index = parts.index("images")
        tail = Path(*parts[index + 1 :]).with_suffix(".txt")
        for labels_root in (root.joinpath(*parts[:index], "labels"), root / "labels"):
            for candidate in (labels_root / tail, labels_root / tail.with_stem(image.stem)):
                if candidate.is_file():
                    return [candidate]

    return [
        root / "labels" / f"{image.stem}.txt",
        root / f"{image.stem}.txt",
        image.with_suffix(".txt"),
    ]


def _sequence_from_layout(relative: Path) -> str:
    """Strongest signal: a dedicated per-video directory."""
    for part in relative.parts[:-1]:
        lowered = part.lower()
        if any(token in lowered for token in _VIDEO_DIR_TOKENS):
            return part
    return ""


def _sequence_from_name(relative: Path) -> str:
    """Middle signal: a numeric video id inside the filename."""
    match = _VIDEO_IN_NAME.search(relative.stem)
    if match:
        return f"vid_{match.group(1)}"
    parent = relative.parts[-2] if len(relative.parts) > 1 else ""
    return f"dir_{parent}" if parent else ""


class MavVidConverter(Converter):
    alias = "mavvid"

    def _convert(self) -> None:
        stats = LabelStats()
        representatives: dict[str, str] = {}
        strategy_counts = {"layout": 0, "filename": 0, "phash": 0, "orphan": 0}

        for split_name, split_root in self._split_roots():
            images = sorted(p for p in split_root.rglob("*") if is_image(p))
            self.report.images_seen += len(images)
            log.info(
                "converting split",
                extra={"dataset": self.alias, "split": split_name, "images": len(images)},
            )

            per_sequence_frame = 0
            current_sequence = ""

            for image in images:
                relative = image.relative_to(split_root)
                sequence_id, strategy = self._resolve_sequence(relative, image, representatives)
                strategy_counts[strategy] += 1

                if sequence_id != current_sequence:
                    current_sequence = sequence_id
                    per_sequence_frame = 0
                per_sequence_frame += 1

                width, height = image_size(image)
                boxes, source_labels = self._read_label(split_root, image, stats)

                self.emit_frame(
                    sequence_id=sequence_id,
                    frame_index=per_sequence_frame - 1,
                    modality="rgb",
                    source=image,
                    boxes=boxes,
                    source_labels=source_labels,
                    width=width,
                    height=height,
                    notes=f"source_split={split_name};group={strategy}",
                )

        self.report.unknown_labels = dict(stats.unknown_labels)
        self.report.sequences_seen = {k: v for k, v in strategy_counts.items() if v}

        if strategy_counts["phash"]:
            self.note(
                f"{strategy_counts['phash']} frames had no video identifier in their path, so "
                "their sequence was recovered from a perceptual hash. Verify with "
                "`anti-uav stats --dataset mavvid` that the sequence count is plausible "
                "(expect roughly 64); two visually similar videos may have been merged."
            )
        if strategy_counts["orphan"]:
            self.note(
                f"{strategy_counts['orphan']} frames could not be grouped at all and were "
                "each isolated into their own sequence. They cannot leak across splits, but "
                "they also get no motion context."
            )

    # -- helpers ------------------------------------------------------------ #

    def _split_roots(self) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        for name in SPLIT_DIRS:
            candidate = self.raw_root / name
            if candidate.is_dir() and any(is_image(p) for p in candidate.rglob("*")):
                found.append((name, candidate))
        if found:
            return found
        if any(is_image(p) for p in self.raw_root.rglob("*")):
            self.note(
                "No train/ or val/ directory found; treating the whole tree as one split "
                "and re-deriving the split from sequences. Expect a worse val estimate than "
                "the publisher's own split."
            )
            return [("all", self.raw_root)]
        raise FileNotFoundError(
            f"{self.alias}: no images under {self.raw_root}. Expected a `train/` and `val/` "
            "directory, or a flat image tree. Check that the Bitbucket clone completed and "
            "that you are pointing at the repo root."
        )

    def _resolve_sequence(
        self,
        relative: Path,
        image: Path,
        representatives: dict[str, str],
    ) -> tuple[str, str]:
        sequence_id = _sequence_from_layout(relative)
        if sequence_id:
            return sequence_id, "layout"

        sequence_id = _sequence_from_name(relative)
        if sequence_id:
            return sequence_id, "filename"

        try:
            digest = phash(image)
        except Exception as exc:  # noqa: BLE001 - a corrupt frame must not kill the run
            log.debug("phash failed", extra={"file": image.name, "error": str(exc)})
            return f"orphan_{image.stem}", "orphan"

        for known_hash, known_sequence in representatives.items():
            if hamming(digest, known_hash) <= SAME_VIDEO_THRESHOLD:
                return known_sequence, "phash"

        sequence_id = f"clip_{len(representatives):04d}"
        representatives[digest] = sequence_id
        return sequence_id, "phash"

    def _read_label(
        self,
        split_root: Path,
        image: Path,
        stats: LabelStats,
    ) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
        candidates = _label_candidates(split_root, image)
        label_file = next((c for c in candidates if c.is_file()), None)
        if label_file is None:
            self.report.images_without_labels += 1
            return [], []

        boxes: list[tuple[int, float, float, float, float]] = []
        labels: list[str] = []
        with label_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    stats.boxes_degenerate += 1
                    continue
                try:
                    index, canonical = resolve_class(
                        parts[0], self.spec.classes, strict=self.strict_labels
                    )
                except UnknownLabelError:
                    key = parts[0].strip().lower()
                    stats.unknown_labels[key] = stats.unknown_labels.get(key, 0) + 1
                    continue
                if index is None:
                    stats.ignored_labels[canonical] = stats.ignored_labels.get(canonical, 0) + 1
                    continue
                try:
                    values = [float(v) for v in parts[1:5]]
                except ValueError:
                    stats.boxes_degenerate += 1
                    continue
                if values[2] <= 0 or values[3] <= 0:
                    stats.boxes_degenerate += 1
                    continue
                stats.boxes_kept += 1
                boxes.append((index, values[0], values[1], values[2], values[3]))
                labels.append(canonical)
        return boxes, labels


def list_raw_sequences(root: Path) -> Iterable[Path]:
    """Every directory that looks like a per-video folder. Used by ``stats``."""
    for path in sorted(root.rglob("*")):
        if path.is_dir() and any(token in path.name.lower() for token in _VIDEO_DIR_TOKENS):
            yield path