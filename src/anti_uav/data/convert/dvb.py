"""Drone-vs-Bird converter.

Annotation format (published by the WOSDETC challenge, documented in their
GitHub repo):

    <frame number> <number of objects> <x> <y> <width> <height> [x y w h ...]

One line per video frame, coordinates in **pixels**, origin top-left. Two details
worth knowing:

* The frame number is **1-based**. Row ``1`` describes the first decoded frame.
  Mixing this up with 0-based indexing shifts every label by one frame.
* An empty line (``0`` objects) is meaningful - it is a frame with no drone and
  no bird. Those are the most valuable frames in the dataset, because they are
  what teaches the model to *not* fire. They are kept.

The Kaggle mirror (``romsham/dronevsbird-foryolo``) ships images with YOLO
labels rather than videos, so :func:`convert` dispatches to
:class:`MavVidConverter`-style flat handling when it finds no videos. See
``_flat_convert`` below.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...utils.logging import get_logger
from ..harmonize import LabelStats, UnknownLabelError, resolve_class
from .base import Converter
from .video_common import VIDEO_SUFFIXES, Det, VideoConverter

log = get_logger(__name__)

_SPLIT_DIRS = ("train", "val", "valid", "validation", "test")

#: Label-directory spellings tried against the image's own stem. The mirror
#: splits labels by class (``labels_birds`` / ``labels_drones``); the generic
#: ``images/`` -> ``labels/`` convention is kept for other YOLO mirrors.
_LABEL_DIR_NAMES = ("labels", "labels_birds", "labels_drones")


def _images_dir_index(parts: tuple[str, ...]) -> int | None:
    """Index of the image directory component, if the path has one.

    Matches the bare ``images`` spelling that YOLO mirrors conventionally use and
    the mirror's class-suffixed ``images_birds`` / ``images_drones``. An exact
    ``parts.index("images")`` misses the suffixed forms entirely, which is how an
    entire 9000-image class half can end up silently unlabelled.
    """
    for index, part in enumerate(parts[:-1]):
        if part == "images" or part.startswith("images_"):
            return index
    return None


class DvbConverter(VideoConverter):
    """Video + per-video ``.txt`` annotation."""

    alias = "dvb"
    primary_modality = "rgb"

    def parse_annotations(self, video: Path, sequence_id: str) -> dict[int, list[Det]]:
        annotation = self._find_annotation(video)
        if annotation is None:
            raise FileNotFoundError(f"no annotation file found beside {video.name}")

        drone_index = self._resolve_class_index("drone")
        bird_index = self._resolve_class_index("bird")

        out: dict[int, list[Det]] = {}
        with annotation.open(encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    frame_number = int(float(parts[0]))
                    count = int(float(parts[1]))
                except ValueError:
                    self.note(f"{annotation.name}:{lineno}: unparseable header, skipped")
                    continue

                # 1-based -> 0-based. Everything downstream keys on the decoded
                # frame counter, which starts at 0.
                frame_index = frame_number - 1
                if count <= 0:
                    out.setdefault(frame_index, [])
                    continue
                if len(parts) < 2 + count * 4:
                    # Truncated row. Record nothing for this frame rather than
                    # inventing boxes from a partial line.
                    self.note(
                        f"{annotation.name}:{lineno}: declares {count} objects but has "
                        f"{max(0, (len(parts) - 2) // 4)}; frame skipped"
                    )
                    continue

                dets: list[Det] = []
                for i in range(count):
                    base = 2 + i * 4
                    try:
                        x, y, w, h = (float(v) for v in parts[base : base + 4])
                    except ValueError:
                        self.note(f"{annotation.name}:{lineno}: object {i} unparseable")
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    dets.append(
                        Det(
                            x1=x,
                            y1=y,
                            x2=x + w,
                            y2=y + h,
                            class_id=drone_index,
                            present=True,
                        )
                    )
                out[frame_index] = dets

        return out

    # -- helpers ------------------------------------------------------------ #

    def _resolve_class_index(self, name: str) -> int:
        try:
            index, _ = resolve_class(name, self.spec.classes, strict=self.strict_labels)
        except UnknownLabelError as exc:
            raise ValueError(
                f"{self.alias}: the registry does not map {name!r} into the unified label "
                f"space, but this converter needs it. Add it to "
                f"configs/datasets/registry.yaml. ({exc})"
            ) from exc
        if index is None:
            raise ValueError(
                f"{self.alias}: {name!r} is listed in ignore_labels, so this dataset would "
                f"produce no usable annotations. Remove it from ignore_labels."
            )
        return index

    def _find_annotation(self, video: Path) -> Path | None:
        for suffix in (".txt", ".csv"):
            exact = video.with_suffix(suffix)
            if exact.is_file():
                return exact
        for path in sorted(video.parent.iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".csv"} and path != video:
                return path
        return None


class DvbFlatConverter(Converter):
    """Kaggle-mirror handling: flat image dirs + YOLO labels, no videos.

    ``romsham/dronevsbird-foryolo`` ships ``train/`` and ``val/`` image folders
    with sibling ``.txt`` labels and **integer** class ids (0 = drone,
    1 = bird). Since the registry maps the *names* rather than the ids, this
    converter resolves ids positionally and cross-checks the resulting class
    ratio against the published statistics.

    That cross-check is not paranoia. A silent 0/1 inversion is invisible in a
    per-class mAP table unless you already know which way round it should be, and
    it would invert the one model output the system cares about.
    """

    alias = "dvb"

    #: From the IROS 2021 benchmark: 0.10% mean object area, and the challenge is
    #: drone-vs-bird, so drone boxes must outnumber bird boxes. If birds
    #: outnumber drones by more than this, the ids are probably swapped.
    MAX_BIRD_TO_DRONE_RATIO = 1.5

    def _convert(self) -> None:
        stats = LabelStats()
        roots = self._image_roots()
        if not roots:
            raise FileNotFoundError(
                f"{self.alias}: no images under {self.raw_root}. Expected train/ and val/ "
                f"directories containing images with sibling .txt labels."
            )

        counts = {"drone": 0, "bird": 0}
        resolved_ids: dict[str, int] = {}

        for split_name, root in roots:
            images = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
            self.report.images_seen += len(images)
            log.info("converting flat split", extra={"dataset": self.alias, "split": split_name, "images": len(images)})

            per_sequence = 0
            current = ""
            for image in images:
                sequence_id, strategy = self._resolve_sequence(image, root)
                if sequence_id != current:
                    current = sequence_id
                    per_sequence = 0
                per_sequence += 1

                boxes, source_labels = self._read_label(image, root, stats, resolved_ids)
                for label in source_labels:
                    counts[label] = counts.get(label, 0) + 1

                height, width = _shape(image)
                self.emit_frame(
                    sequence_id=sequence_id,
                    frame_index=per_sequence - 1,
                    modality="rgb",
                    source=image,
                    boxes=boxes,
                    source_labels=source_labels,
                    width=width,
                    height=height,
                    notes=f"source_split={split_name};group={strategy}",
                )

        self.report.unknown_labels = dict(stats.unknown_labels)
        self._validate_class_ratio(counts, resolved_ids)

    # -- helpers ------------------------------------------------------------ #

    def _image_roots(self) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        for name in _SPLIT_DIRS:
            candidate = self.raw_root / name
            if candidate.is_dir() and any(
                p.suffix.lower() in _IMAGE_SUFFIXES for p in candidate.rglob("*") if p.is_file()
            ):
                found.append((name, candidate))
        if found:
            return found
        if any(p.suffix.lower() in _IMAGE_SUFFIXES for p in self.raw_root.rglob("*") if p.is_file()):
            self.note(
                "No train/ or val/ directory found; treating the tree as one split. The "
                "validation estimate will be weaker than the publisher's own split."
            )
            return [("all", self.raw_root)]
        return []

    def _resolve_sequence(self, image: Path, root: Path) -> tuple[str, str]:
        relative = image.relative_to(root)
        for part in relative.parts[:-1]:
            lowered = part.lower()
            if any(token in lowered for token in ("video", "seq", "clip", "scene")):
                return part, "layout"
        match = re.search(r"(?:^|[_-])(\d{3,6})(?=[_.-]|$)", relative.stem)
        if match:
            return f"{self._id_prefix(relative)}{match.group(1)}", "filename"
        return f"orphan_{image.stem}", "orphan"

    def _id_prefix(self, relative: Path) -> str:
        """Disambiguator for the filename-derived sequence id.

        The numeric tail alone is not unique across this mirror. The drone half
        is numbered ``image_000123`` and the bird half ``bird_image_000123``, so a
        bare ``123`` would give both the same ``sequence_id``; they then land on the
        same interim ``frames/<seq>/rgb/000000.jpg`` and the second one overwrites
        the first, leaving a bird image paired with drone labels. Qualifying with
        the containing directory keeps ids unique per frame. These are independent
        still images rather than video frames, so one image per group is the
        honest grouping.
        """
        if len(relative.parts) < 2:
            return "vid_"
        return f"vid_{relative.parts[-2]}_"

    def _read_label(
        self,
        image: Path,
        root: Path,
        stats: LabelStats,
        resolved_ids: dict[str, int],
    ) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
        label_file = self._label_for(image, root)
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
                    raw_id = parts[0].strip().lower()
                    canonical = ""
                    class_id: int | None
                    if raw_id.isdigit():
                        # Positional: the mirror documents 0 = drone, 1 = bird.
                        class_id = self._id_to_index(raw_id, resolved_ids)
                        canonical = self.spec.classes.unified_labels[class_id]
                    else:
                        class_id, canonical = resolve_class(
                            raw_id, self.spec.classes, strict=self.strict_labels
                        )
                        if class_id is None:
                            stats.ignored_labels[canonical] = stats.ignored_labels.get(canonical, 0) + 1
                            continue
                    values = [float(v) for v in parts[1:5]]
                except (ValueError, UnknownLabelError) as exc:
                    stats.unknown_labels[parts[0].strip().lower()] = (
                        stats.unknown_labels.get(parts[0].strip().lower(), 0) + 1
                    )
                    del exc
                    continue

                if values[2] <= 0 or values[3] <= 0:
                    stats.boxes_degenerate += 1
                    continue
                stats.boxes_kept += 1
                boxes.append((class_id, values[0], values[1], values[2], values[3]))
                labels.append(canonical)
        return boxes, labels

    def _id_to_index(self, raw_id: str, resolved: dict[str, int]) -> int:
        names = self.spec.classes.unified_labels
        if raw_id not in resolved:
            index = int(raw_id)
            if not 0 <= index < len(names):
                raise UnknownLabelError(raw_id, names)
            resolved[raw_id] = index
        return resolved[raw_id]

    def _label_for(self, image: Path, root: Path) -> Path | None:
        """Locate the label file for ``image``.

        The mirror ships one label directory per class rather than a single
        ``labels/`` tree::

            Data/images_birds/bird_image_000000.png
            Data/labels_birds/bird_image_000000.txt
            Data/images_drones/image_000000.png
            Data/labels_drones/image_000000.txt

        Note the stems differ between the two halves - the bird side is
        ``bird_image_NNNNNN`` and the drone side is just ``image_NNNNNN`` - so the
        lookup has to key on the image's own stem and never on a reconstructed
        name. Resolving ``labels_birds`` -> ``labels`` by string substitution is
        what the generic ``images/`` branch below assumes, and it silently fails
        here: every frame comes back unlabelled and the resulting dataset trains
        a detector that has been taught nothing.
        """
        relative = image.relative_to(root)
        parts = relative.parts
        stem = image.stem

        candidates: list[Path] = []

        index = _images_dir_index(parts)
        if index is not None:
            prefix = parts[:index]
            tail_parts = parts[index + 1 : -1]
            for tail_dir in _LABEL_DIR_NAMES:
                candidates.append(
                    root.joinpath(*prefix, tail_dir, *tail_parts, f"{stem}.txt")
                )

        candidates.extend(
            [
                image.with_suffix(".txt"),
                root / "labels" / f"{stem}.txt",
                root / f"{stem}.txt",
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        self.note(
            f"no label file for {relative.as_posix()}; tried "
            + ", ".join(c.relative_to(root).as_posix() for c in candidates[:4])
        )
        return None

    def _validate_class_ratio(self, counts: dict[str, int], resolved_ids: dict[str, int]) -> None:
        drones, birds = counts.get("drone", 0), counts.get("bird", 0)
        if resolved_ids:
            self.note(
                f"Numeric class ids were interpreted positionally as "
                f"{resolved_ids} against the unified order {self.spec.classes.unified_labels}. "
                f"Resulting counts: drone={drones:,} bird={birds:,}."
            )
        if birds and drones and birds > drones * self.MAX_BIRD_TO_DRONE_RATIO:
            self.fail(
                f"Class ids look INVERTED: {birds:,} bird boxes vs {drones:,} drone boxes. "
                f"This dataset exists to find drones among birds, so drones should dominate. "
                f"Fix classes.mapping (or the id order in the mirror) and re-run."
            )
            raise ValueError(self.report.errors[-1])


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _shape(image: Path) -> tuple[int, int]:
    from ...utils.imaging import image_size

    width, height = image_size(image)
    return height, width


def detect_layout(raw_root: Path) -> str:
    """``video`` or ``flat`` - which converter the CLI should use."""
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            return "video"
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            return "flat"
    return "unknown"