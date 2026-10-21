"""Converters for the four YOLO-format community drone datasets.

``kaggle_cybersimar``, ``github_maciullo``, ``hf_seraphim`` and
``kaggle_banderastepan`` are the same shape: a publisher has already cropped
bounding boxes and shipped them as YOLO ``.txt`` files beside the images. So
there is no annotation parsing to do, no video to decode, and no video-specific
trap - the real work is the same for all four, and it is **sequence recovery**.

The trap, and the reason this is not ten lines: these are curated frame dumps.
The images come out of videos, but the publisher's directory layout usually
does not say which video a frame came from - a flat ``images/`` folder with
``frame_0001.jpg`` in it. Grouping by filename would put near-duplicate frames
of the same target on both sides of the train/val boundary, and mAP would
inflate by double digits and mean nothing. This project splits **by sequence**,
and ``anti-uav sanity`` fails the build if a sequence appears in two splits.

So sequence resolution is layered, strongest signal first:

1. a per-clip directory in the path (``video_003/``, ``clip_07/``) - the layouts
   that keep their folder structure;
2. a numeric clip id embedded in the filename (``drone_00123.jpg``);
3. the nearest meaningful parent directory name - weaker, but it is still
   grouping frames that plausibly came from the same source;
4. one sequence per image - the safe fallback. It cannot leak, because a
   single frame is its own group, but it also gets no motion context and it
   will scatter the val set, so it is counted and reported.

Step 4 is deliberately not an error. A frame-per-sequence dataset trains and
scores correctly; it just scores conservatively.

The class mapping is the other thing that has to be right, and it is the thing
the registry is for: ``classes.mapping`` in ``configs/datasets/registry.yaml``
names the source spelling of each class, and an unmapped label is a hard
failure rather than a silent drop. Only ``github_maciullo`` carries bird
annotations, so it is the only one of the four whose precision number can
falsify a false-positive claim.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...utils.imaging import is_image
from ...utils.logging import get_logger
from ..harmonize import LabelStats, UnknownLabelError, resolve_class
from .base import Converter

log = get_logger(__name__)

#: Directory names that mark a publisher's train/val split. These are NOT
#: sequences - every split would otherwise look like one giant group.
SPLIT_DIRS = ("train", "val", "valid", "validation", "test", "testing")

#: Directory-name tokens that mark a per-clip folder.
_CLIP_DIR_TOKENS = ("video", "vid", "seq", "clip", "scene", "track", "source", "sample")

#: Matches a numeric clip id inside a filename: ``<prefix>-000123<suffix>``.
_NUMERIC_ID = re.compile(r"(?:^|[_-])(\d{3,6})(?=[_.-]|$)")

#: Directory names that are packaging, not content. They must not become a
#: sequence id, or every frame of a dataset collapses into one group.
_WRAPPER_DIRS = (
    "images",
    "labels",
    "image",
    "label",
    "annotations",
    "dataset",
    "data",
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
    "all",
    "raw",
    "rgb",
    "drone",
    "drones",
    "bird",
    "birds",
)


def _label_candidates(root: Path, image: Path) -> list[Path]:
    """Locate the label file for an image under the layouts these mirrors use.

    Mirrors differ: some ship ``images/`` + ``labels/`` as parallel trees, some
    put the ``.txt`` next to the image, some nest both under a ``dataset/``
    folder. All three are tried, in that order, and the first hit wins.
    """
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


def _is_split_dir(name: str) -> bool:
    return name.lower() in SPLIT_DIRS


def _is_wrapper_dir(name: str) -> bool:
    lowered = name.lower()
    return lowered in _WRAPPER_DIRS or lowered.startswith(".")


def _sequence_from_layout(relative: Path) -> str:
    """Strongest signal: a dedicated per-clip directory."""
    for part in relative.parts[:-1]:
        lowered = part.lower()
        if _is_split_dir(lowered) or _is_wrapper_dir(lowered):
            continue
        if any(token in lowered for token in _CLIP_DIR_TOKENS):
            return part
    return ""


def _sequence_from_name(relative: Path) -> str:
    """Middle signal: a numeric clip id inside the filename."""
    match = _NUMERIC_ID.search(relative.stem)
    if match:
        return f"clip_{match.group(1)}"
    return ""


def _sequence_from_parent(relative: Path) -> str:
    """Weakest useful signal: the nearest non-wrapper parent directory."""
    for part in reversed(relative.parts[:-1]):
        if _is_split_dir(part) or _is_wrapper_dir(part):
            continue
        return part
    return ""


class YoloPairsConverter(Converter):
    """Shared implementation for the four YOLO-format community datasets.

    Subclasses only differ by :attr:`alias` and by which subdirectory of the
    published payload actually holds the images - the layout differs per source
    and is discovered rather than assumed.
    """

    alias = ""

    #: Subdirectory of the published payload that holds the images, when the
    #: mirror nests the frame tree under a known name. Empty means "search".
    images_subdir = ""

    def _convert(self) -> None:
        stats = LabelStats()
        strategy_counts = {"layout": 0, "filename": 0, "parent": 0, "orphan": 0}

        image_root = self._locate_image_root()
        images = sorted(p for p in image_root.rglob("*") if is_image(p))
        self.report.images_seen += len(images)
        log.info(
            "converting",
            extra={"dataset": self.alias, "images": len(images), "root": str(image_root)},
        )

        if not images:
            raise FileNotFoundError(
                f"{self.alias}: no images under {self.raw_root}. Check that the download "
                "completed and that you are pointing at the dataset root, not at a "
                "subdirectory."
            )

        per_sequence_frame: dict[str, int] = {}
        current_sequence = ""

        for image in images:
            relative = image.relative_to(image_root)
            sequence_id, strategy = self._resolve_sequence(relative)
            strategy_counts[strategy] += 1

            if sequence_id != current_sequence:
                current_sequence = sequence_id
            index = per_sequence_frame.get(sequence_id, 0)
            per_sequence_frame[sequence_id] = index + 1

            width, height = self._image_size(image)
            boxes, source_labels = self._read_label(image_root, image, stats)

            self.emit_frame(
                sequence_id=sequence_id,
                frame_index=index,
                modality="rgb",
                source=image,
                boxes=boxes,
                source_labels=source_labels,
                width=width,
                height=height,
                notes=f"group={strategy}",
            )

        self.report.unknown_labels = dict(stats.unknown_labels)
        self.report.sequences_seen = {k: v for k, v in strategy_counts.items() if v}

        if strategy_counts["orphan"]:
            self.note(
                f"{strategy_counts['orphan']} frames carried no usable sequence signal and "
                "were each isolated into their own sequence. That cannot leak across the "
                "train/val boundary, but it also means no motion context - expect a "
                "conservative val estimate rather than a wrong one."
            )
        if strategy_counts["parent"]:
            self.note(
                f"{strategy_counts['parent']} frames were grouped by their parent directory "
                "name. If two of those directories came from the same source video, the "
                "val estimate is optimistic. Check `anti-uav stats --dataset "
                f"{self.alias}` for a plausible sequence count."
            )

    # -- helpers ------------------------------------------------------------ #

    def _locate_image_root(self) -> Path:
        """Find the directory that actually holds the images.

        A Kaggle payload is often ``<slug>/dataset/images/...`` while a GitHub
        clone is the repo root and a HuggingFace snapshot is a parquet/arrow
        tree. Rather than hardcode each one, the tree is searched for the
        directory with the most images under it.
        """
        if self.images_subdir:
            candidate = self.raw_root / self.images_subdir
            if candidate.is_dir() and any(is_image(p) for p in candidate.rglob("*")):
                return candidate

        best: tuple[int, Path] = (0, self.raw_root)
        for candidate in self._image_dirs():
            count = sum(1 for p in candidate.rglob("*") if is_image(p))
            if count > best[0]:
                best = (count, candidate)
        return best[1]

    def _image_dirs(self) -> list[Path]:
        dirs: list[Path] = [self.raw_root]
        try:
            dirs.extend(sorted(p for p in self.raw_root.rglob("*") if p.is_dir()))
        except OSError:
            pass
        return [p for p in dirs if not any(part.startswith(".") for part in p.parts)]

    def _resolve_sequence(self, relative: Path) -> tuple[str, str]:
        sequence_id = _sequence_from_layout(relative)
        if sequence_id:
            return sequence_id, "layout"

        sequence_id = _sequence_from_name(relative)
        if sequence_id:
            return sequence_id, "filename"

        sequence_id = _sequence_from_parent(relative)
        if sequence_id:
            return sequence_id, "parent"

        return f"frame_{relative.stem}", "orphan"

    @staticmethod
    def _image_size(image: Path) -> tuple[int, int]:
        from ...utils.imaging import image_size

        return image_size(image)

    def _read_label(
        self,
        image_root: Path,
        image: Path,
        stats: LabelStats,
    ) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
        candidates = _label_candidates(image_root, image)
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


class KaggleCybersimarConverter(YoloPairsConverter):
    """Kaggle ``cybersimar08/drone-detection``. Drone-only, YOLO-format labels."""

    alias = "kaggle_cybersimar"


class GitHubMaciulloConverter(YoloPairsConverter):
    """GitHub ``Maciullo/DroneDetectionDataset``. Drone **and** bird labels."""

    alias = "github_maciullo"


class HfSeraphimConverter(YoloPairsConverter):
    """HuggingFace ``lgrzybowski/seraphim-drone-detection-dataset``. Drone-only."""

    alias = "hf_seraphim"


class KaggleBanderastepanConverter(YoloPairsConverter):
    """Kaggle ``banderastepan/drone-detection``. Drone-only, nested in ``dataset/``."""

    alias = "kaggle_banderastepan"
    images_subdir = "dataset"
