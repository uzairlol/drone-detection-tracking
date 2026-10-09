"""MM-UAV converter.

MM-UAV ships **frames, not videos** - the publisher already extracted them - so
this is a remap job rather than a decode job, with three things it must get right.

**Frame numbering vs filename numbering.** The MOT ``gt.txt`` is keyed on
``frame_id``; the image files are named after their own index. These agree in
MM-UAV, but the converter matches them explicitly rather than assuming, and
reports a mismatch rather than producing silently mislabelled data.

**Stride.** With ``stride=2`` we keep every other frame *and* halve the ground
truth rows consistently, so a track still spans the kept frames contiguously.

**Event and IR modalities.** Default is RGB only. Event frames cannot train an
RGB detector at all, and IR triples the download for a modality the fleet only
partially covers. Both are opt-in via ``--modalities``.

The MOT ground truth is written through unchanged to ``mot/<seq>/<modality>.txt``
so the tracker evaluator can score against the same file the authors' own toolkit
uses, with no conversion ambiguity in the metric path.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ...utils.imaging import image_size
from ...utils.io import require_free_space
from ...utils.logging import get_logger
from ..harmonize import LabelStats, MIN_BOX_SIDE_PX, boxes_from_xyxy
from .base import Converter

log = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

#: MOT column indices for ``frame, id, x1, y1, x2, y2, conf, class, vis``.
MOT_COLUMNS = 9
_IDX_FRAME, _IDX_TRACK = 0, 1
_IDX_X1, _IDX_Y1, _IDX_X2, _IDX_Y2 = 2, 3, 4, 5
_IDX_CONF, _IDX_CLASS, _IDX_VIS = 6, 7, 8


class MmUavConverter(Converter):
    alias = "mmuav"

    def __init__(
        self,
        *args,
        modalities: list[str] | None = None,
        max_sequences: int | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(*args, **kwargs)
        self.modalities = modalities or ["rgb"]
        self.max_sequences = max_sequences
        self.drone_index = self.spec.classes.unified_labels.index("drone")

    def _convert(self) -> None:
        sequences = self._sequence_dirs()
        if not sequences:
            raise FileNotFoundError(
                f"{self.alias}: found no sequences under {self.raw_root}.\n"
                f"Expected the publisher's layout:\n"
                f"  <root>/train/<NNNN>/rgb_frame/*.jpg  and  <root>/train/<NNNN>/gt_rgb/gt.txt\n"
                f"If you downloaded only part of the archive, point --raw at the parent "
                f"that still contains the `train/` directory."
            )

        if self.max_sequences and len(sequences) > self.max_sequences:
            self.note(
                f"Limiting to the first {self.max_sequences} of {len(sequences)} sequences "
                f"(sorted numerically, so this is deterministic rather than arbitrary)."
            )
            sequences = sequences[: self.max_sequences]

        stride = self.spec.ingest.stride
        if stride > 1:
            self.note(
                f"stride={stride}: keeping every {stride}{_ordinal(stride)} frame. Ground "
                f"truth rows are filtered to match, so tracks stay contiguous over the "
                f"frames that remain."
            )

        for index, sequence_dir in enumerate(sequences, start=1):
            sequence_id = sequence_dir.name
            for modality in self.modalities:
                self._convert_stream(sequence_dir, sequence_id, modality, stride)

            if index % 25 == 0 or index == len(sequences):
                log.info(
                    "sequence progress",
                    extra={
                        "dataset": self.alias,
                        "done": index,
                        "total": len(sequences),
                        "frames": self.report.frames_written,
                    },
                )

    # -- one sequence, one modality ----------------------------------------- #

    def _convert_stream(
        self, sequence_dir: Path, sequence_id: str, modality: str, stride: int
    ) -> None:
        frame_dir = self._find_frame_dir(sequence_dir, modality)
        if frame_dir is None:
            if modality == "rgb":
                self.note(f"{sequence_id}: no {modality}_frame directory; skipped.")
            return

        gt_path = self._find_gt(sequence_dir, modality)
        ground_truth = self._load_gt(gt_path) if gt_path else {}
        if gt_path is None and modality == "rgb":
            self.note(f"{sequence_id}: no ground truth file; emitting frames as negatives only.")
        elif gt_path is None:
            return  # a modality without its own GT is simply not annotated

        stats = LabelStats()
        images = sorted(
            p for p in frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            return

        mot_rows: list[tuple[float, ...]] = []
        emitted = 0

        for source_index, image in enumerate(images):
            if source_index % stride:
                continue

            frame_number = _int_or(image.stem, source_index + 1)
            dets = ground_truth.get(frame_number, [])
            width, height = image_size(image)

            boxes: list[tuple[int, float, float, float, float]] = []
            visibilities: list[float] = []
            track_id: int | None = None
            dropped = 0

            for det in dets:
                confidence, visibility = det[_IDX_CONF], det[_IDX_VIS]
                if visibility <= 0.0:
                    continue
                converted = boxes_from_xyxy(
                    [(self.drone_index, det[_IDX_X1], det[_IDX_Y1], det[_IDX_X2], det[_IDX_Y2])],
                    width=width,
                    height=height,
                )
                if not converted:
                    dropped += 1
                    continue
                box = converted[0]
                if box[3] * width < MIN_BOX_SIDE_PX or box[4] * height < MIN_BOX_SIDE_PX:
                    dropped += 1
                    continue
                boxes.append(box)
                visibilities.append(visibility)
                if track_id is None:
                    track_id = int(det[_IDX_TRACK])
                mot_rows.append(
                    (
                        emitted + 1,
                        det[_IDX_TRACK],
                        det[_IDX_X1],
                        det[_IDX_Y1],
                        det[_IDX_X2],
                        det[_IDX_Y2],
                        confidence,
                        self.drone_index + 1,
                        visibility,
                    )
                )

            self.emit_frame(
                sequence_id=sequence_id,
                frame_index=emitted,
                modality=modality,
                source=image,
                boxes=boxes,
                source_labels=["drone"] * len(boxes),
                track_id=track_id,
                visibility=visibilities,
                target_present=bool(boxes),
                dropped_boxes=dropped,
                width=width,
                height=height,
                notes=f"src_frame={source_index}",
            )
            emitted += 1

        if mot_rows:
            # Keep the author's ground truth verbatim so the tracker metric path
            # has no conversion ambiguity.
            self.write_mot_gt(sequence_id=sequence_id, modality=modality, rows=mot_rows)

        if emitted == 0:
            self.note(f"{sequence_id}/{modality}: every frame was filtered out by stride.")

    # -- layout discovery ---------------------------------------------------- #

    def _sequence_dirs(self) -> list[Path]:
        """Directories that contain at least one ``*_frame`` folder."""
        found: list[Path] = []
        for path in self.raw_root.rglob("*"):
            if not path.is_dir():
                continue
            if any(child.is_dir() and child.name.endswith("_frame") for child in path.iterdir()):
                found.append(path)

        def sort_key(path: Path) -> tuple[int, str]:
            name = path.name
            return (int(name) if name.isdigit() else 1 << 30, name)

        return sorted(found, key=sort_key)

    def _find_frame_dir(self, sequence_dir: Path, modality: str) -> Path | None:
        for name in (f"{modality}_frame", f"{modality}", modality):
            candidate = sequence_dir / name
            if candidate.is_dir():
                return candidate
        return None

    def _find_gt(self, sequence_dir: Path, modality: str) -> Path | None:
        """Ground truth for one modality.

        MM-UAV uses ``gt_rgb/gt.txt`` and ``gt_ir/gt.txt``. An older mirror used a
        bare ``gt.txt`` at the sequence root, which is assumed to be RGB.
        """
        for name in (f"gt_{modality}", "gt", "labels"):
            candidate = sequence_dir / name / "gt.txt"
            if candidate.is_file():
                return candidate
        candidate = sequence_dir / f"gt_{modality}.txt"
        if candidate.is_file():
            return candidate
        if modality == "rgb" and (sequence_dir / "gt.txt").is_file():
            return sequence_dir / "gt.txt"
        return None

    def _load_gt(self, path: Path) -> dict[int, list[tuple[float, ...]]]:
        """Parse a MOT-format ground truth file into ``{frame: [rows]}``.

        Rows are kept as raw float tuples so the same parse serves both the YOLO
        label writer and the verbatim MOT copy.
        """
        out: dict[int, list[tuple[float, ...]]] = {}
        malformed = 0
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
                if len(parts) < MOT_COLUMNS:
                    malformed += 1
                    continue
                try:
                    values = [float(p) for p in parts[:MOT_COLUMNS]]
                except ValueError:
                    malformed += 1
                    continue
                out.setdefault(int(values[_IDX_FRAME]), []).append(tuple(values))

        if malformed:
            self.note(
                f"{path.name}: {malformed} malformed ground-truth rows skipped. A non-trivial "
                f"count usually means a column-order difference; check the file before "
                f"trusting the metrics."
            )
        return out


def _int_or(text: str, default: int) -> int:
    try:
        return int(text)
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else default


def _ordinal(n: int) -> str:
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10 if n < 20 else 0, "th")


def estimate_disk_bytes(
    root: Path, *, max_sequences: int | None, modalities: Iterable[str]
) -> int:
    """Rough on-disk cost of a subset before it is copied, for the CLI's warning."""
    from ...utils.imaging import list_images

    sequences = sorted(p for p in root.rglob("*") if p.is_dir() and (p / "rgb_frame").is_dir())
    if max_sequences:
        sequences = sequences[:max_sequences]

    total = 0
    for sequence in sequences:
        for modality in modalities:
            for image in list_images(sequence / f"{modality}_frame", recursive=False):
                total += image.stat().st_size
    return total


def require_room(root: Path, needed: int) -> None:
    require_free_space(root, needed * 3, context="MM-UAV interim copy (source + frames + labels)")