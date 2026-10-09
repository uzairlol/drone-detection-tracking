"""Per-dataset converters.

Each converter turns one publisher's layout into the normalised interim tree
described in :mod:`anti_uav.data.frameindex`. They differ only in how they *find*
the boxes; emitting frames, harmonising labels and reporting are shared.

Converter contract:

* never abort part-way on a bad annotation - tally it in ``ConvertReport`` and
  keep going, so one corrupt file does not cost you a 40-minute conversion;
* **do** abort if no source label could be mapped at all, because that means the
  class mapping is wrong and everything downstream would be silently wrong;
* record ``target_present=False`` for frames the publisher marks as
  target-absent. Those are real negatives and are the difference between learning
  "no drone here" and learning "always fire";
* resolve a stable ``sequence_id`` per frame. Split safety depends entirely on
  this being right, so it is never guessed from a single weak signal when a
  stronger one exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import shutil

from ...config.schema import DatasetSpec
from ...utils.imaging import read_image, write_image
from ...utils.logging import get_logger
from ..frameindex import (
    ConvertReport,
    FrameRecord,
    frames_dir,
    interim_root,
    labels_dir,
    mot_dir,
    stem_for,
    write_index,
    write_report,
)
from ..harmonize import write_yolo_label

log = get_logger(__name__)


class ConversionAborted(RuntimeError):
    """Raised when conversion cannot produce meaningful output."""

    def __init__(self, message: str, report: ConvertReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class Converter(ABC):
    """Base class. One instance per dataset variant."""

    #: Dataset alias this converter handles.
    alias: str = ""

    def __init__(
        self,
        spec: DatasetSpec,
        raw_root: Path,
        *,
        variant: str | None = None,
        strict_labels: bool = True,
        copy_images: bool = True,
    ) -> None:
        self.spec = spec
        self.raw_root = Path(raw_root)
        self.variant = self.resolve_variant(variant)
        self.strict_labels = strict_labels
        self.copy_images = copy_images
        self.records: list[FrameRecord] = []
        self._emitted: set[tuple[str, str, int]] = set()
        #: MOT rows gathered for the sequence currently being converted, drained
        #: by flush_mot_rows. See mot_row_for.
        self._mot_rows: list[tuple[float, ...]] = []
        self.report = ConvertReport(
            dataset=spec.alias,
            variant=self.variant,
            class_mapping_applied=dict(spec.classes.mapping),
        )
        self._interim = interim_root(self.alias, self.variant)

    # -- public API --------------------------------------------------------- #

    def run(self) -> ConvertReport:
        """Execute the conversion and return the report. Sets ``report.ok``."""
        if not self.raw_root.exists():
            raise ConversionAborted(
                f"{self.alias}: raw root does not exist: {self.raw_root}\n"
                f"Run: anti-uav download --dataset {self.alias} --dry-run   (to see the plan)\n"
                f"     anti-uav download --dataset {self.alias}             (to fetch it)"
            )

        self._convert()

        self.report.sequences = len({r.sequence_id for r in self.records})
        self.report.class_histogram = self._class_histogram()
        write_index(self.alias, self.variant, self.records)
        write_report(self.alias, self.variant, self.report)

        self.warn_on_unknown_labels()
        self.fail_fast_if_nothing_mapped()
        self.report.ok = True

        log.info(
            "conversion complete",
            extra={
                "dataset": self.alias,
                "variant": self.variant,
                "frames": self.report.frames_written,
                "boxes": self.report.boxes_written,
                "sequences": self.report.sequences,
                "dropped": self.report.boxes_dropped,
            },
        )
        return self.report

    @abstractmethod
    def _convert(self) -> None:
        """Populate ``self.records`` and ``self.report``. Implemented per dataset."""

    # -- shared emit path --------------------------------------------------- #

    def emit_frame(
        self,
        *,
        sequence_id: str,
        frame_index: int,
        modality: str,
        source: Path | np.ndarray,
        boxes: Sequence[tuple[int, float, float, float, float]],
        source_labels: Sequence[str] = (),
        track_id: int | None = None,
        visibility: Sequence[float] = (),
        target_present: bool = True,
        dropped_boxes: int = 0,
        width: int = 0,
        height: int = 0,
        notes: str = "",
    ) -> FrameRecord:
        """Write one image + label pair and return its index record.

        ``source`` may be a path on disk (image datasets) or a decoded BGR array
        (video datasets, which is the common case here and saves a redundant
        encode). This is the only place frames get written, so the on-disk layout
        is guaranteed identical across all four converters.
        """
        stem = stem_for(frame_index)
        seq = sequence_id
        out_image = frames_dir(self.alias, self.variant) / seq / modality / f"{stem}.jpg"
        out_label = labels_dir(self.alias, self.variant) / seq / modality / f"{stem}.txt"

        emitted_key = (seq, modality, frame_index)
        if emitted_key in self._emitted:
            self.fail(
                f"two frames resolved to the same output path {seq}/{modality}/{stem}. "
                f"The second one silently overwrote the first, so the dataset would "
                f"contain fewer frames than the index claims and could pair an image "
                f"with another frame's labels. This is a sequence_id collision: make "
                f"{self.alias}'s group pattern or filename matching unique per frame."
            )
            raise ConversionAborted(self.report.errors[-1], self.report)
        self._emitted.add(emitted_key)

        if isinstance(source, np.ndarray):
            if not width or not height:
                height, width = source.shape[:2]
            write_image(out_image, source)
        elif self.copy_images:
            write_image(out_image, read_image(source))
        else:
            shutil.copy2(source, out_image)

        write_yolo_label(out_label, boxes)

        record = FrameRecord(
            sequence_id=seq,
            frame_index=frame_index,
            modality=modality,
            dataset=self.alias,
            variant=self.variant,
            image=self._rel(out_image),
            label=self._rel(out_label),
            width=width,
            height=height,
            class_ids=[int(b[0]) for b in boxes],
            source_labels=list(source_labels),
            box_count=len(boxes),
            track_id=track_id,
            visibility=list(visibility),
            target_present=target_present,
            dropped_boxes=dropped_boxes,
            notes=notes,
        )
        self.records.append(record)
        self.report.frames_written += 1
        self.report.boxes_written += len(boxes)
        self.report.boxes_dropped += dropped_boxes
        if width and height:
            key = f"{width}x{height}"
            self.report.frame_size_histogram[key] = self.report.frame_size_histogram.get(key, 0) + 1

        rows = self.mot_row_for(record, self._pixel_boxes(boxes, width, height))
        if rows:
            self._mot_rows.extend(rows)
        return record

    @staticmethod
    def _pixel_boxes(
        boxes: Sequence[tuple[int, float, float, float, float]],
        width: int,
        height: int,
    ) -> list[tuple[float, float, float, float]]:
        """Normalised ``(cx, cy, w, h)`` boxes back to pixel ``(x1, y1, x2, y2)``.

        MOT rows are written in pixels, but ``boxes`` are normalised. Going back
        through the same 6-decimal form the labels were written with keeps the
        round trip inside one pixel for any realistic frame size.
        """
        if not width or not height:
            return []
        out: list[tuple[float, float, float, float]] = []
        for _class_id, cx, cy, bw, bh in boxes:
            out.append(
                (
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                )
            )
        return out

    def mot_row_for(
        self,
        record: FrameRecord,
        pixel_boxes: Sequence[tuple[float, float, float, float]],
    ) -> Sequence[tuple[float, ...]]:
        """Hook: the MOT rows this frame contributes. Empty by default.

        A converter that can supply MOT ground truth overrides this and lets
        :meth:`flush_mot_rows` write it. Without a MOT file the tracking metric
        loader falls back to the YOLO labels, which carry no identity and no
        visibility — so this hook is what keeps a dataset's identity and occlusion
        information from being discarded at conversion time.
        """
        del record, pixel_boxes
        return ()

    def flush_mot_rows(self, *, sequence_id: str, modality: str) -> Path | None:
        """Write and clear the MOT rows collected for one sequence, if any."""
        rows, self._mot_rows = self._mot_rows, []
        if not rows:
            return None
        return self.write_mot_gt(sequence_id=sequence_id, modality=modality, rows=rows)

    def write_mot_gt(
        self,
        *,
        sequence_id: str,
        modality: str,
        rows: Sequence[Sequence[float]],
    ) -> Path:
        """Write MOT-format ground truth for one sequence.

        Rows are ``(frame, id, x1, y1, x2, y2, conf, cls, vis)`` in pixels, which
        is the format both MM-UAV's own toolkit and TrackEval accept.
        """
        target = mot_dir(self.alias, self.variant) / sequence_id / f"{modality}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
            f"{r[6] if len(r) > 6 else 1},{r[7] if len(r) > 7 else 1},"
            f"{r[8] if len(r) > 8 else 1.0}"
            for r in rows
        ]
        body = "\n".join(lines) + ("\n" if lines else "")
        tmp = target.with_suffix(".txt.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(target)
        return target

    # -- helpers ------------------------------------------------------------ #

    def resolve_variant(self, variant: str | None) -> str:
        if variant:
            if self.spec.variants and variant not in (*self.spec.variants, "full", "subset"):
                raise ConversionAborted(
                    f"{self.alias}: variant {variant!r} not in {self.spec.variants}"
                )
            if variant in {"full", "subset"} and self.spec.default_variant:
                return self.spec.default_variant
            return variant
        return self.spec.default_variant or (self.spec.variants[0] if self.spec.variants else "full")

    def _rel(self, path: Path) -> str:
        """Path relative to the interim root, so the whole tree can be moved."""
        try:
            return path.resolve().relative_to(self._interim.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _class_histogram(self) -> dict[str, int]:
        names = self.spec.classes.unified_labels
        histogram = dict.fromkeys(names, 0)
        for record in self.records:
            for class_id in record.class_ids:
                if 0 <= class_id < len(names):
                    histogram[names[class_id]] += 1
        return histogram

    def note(self, message: str) -> None:
        if message not in self.report.warnings:
            self.report.warnings.append(message)

    def fail(self, message: str) -> None:
        if message not in self.report.errors:
            self.report.errors.append(message)

    def warn_on_unknown_labels(self) -> None:
        if self.report.unknown_labels:
            self.note(
                f"{len(self.report.unknown_labels)} distinct label(s) were dropped as unmapped: "
                f"{dict(sorted(self.report.unknown_labels.items()))}. Add them to "
                f"configs/datasets/registry.yaml or accept the loss deliberately."
            )

    def fail_fast_if_nothing_mapped(self) -> None:
        """Hard stop when the class mapping produced zero boxes.

        This catches a publisher changing their label vocabulary, or a
        coordinate convention we misread. Without it you would train on a
        dataset whose annotations were all discarded and only notice via a
        mysteriously flat loss curve.
        """
        if self.report.frames_written > 0 and self.report.boxes_written == 0:
            unknown = ", ".join(sorted(self.report.unknown_labels)) or "(none)"
            message = (
                f"{self.alias}: {self.report.frames_written} frames were read but zero boxes "
                f"survived conversion. Unmapped labels: {unknown}. If the labels ARE mapped, "
                f"the coordinate convention is probably being misread."
            )
            self.fail(message)
            raise ConversionAborted(message, self.report)