"""Shared machinery for the video-based datasets.

``dvb`` and ``antiuav`` both ship video files plus a separate annotation file per
video, so frame extraction, stride handling and the "there is no annotation line
for frame N" case are identical. The publishers' annotation formats differ, and
that parsing lives in the subclasses.

The rule that matters for correctness: **stride is applied to the frame index,
not to the output counter.** With ``stride=3`` and annotations keyed on the
original frame number, frame 7 must be matched against annotation row 7 - not
against the third annotation we happened to read. Getting this wrong shifts every
label by one to three frames, which for a target moving several pixels per frame
is the difference between a tight box and a systematically wrong one.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import NamedTuple

from ...utils.imaging import frames_from_video
from ...utils.logging import get_logger
from ..harmonize import MIN_BOX_SIDE_PX, boxes_from_xyxy
from .base import Converter

log = get_logger(__name__)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")


class Det(NamedTuple):
    """One annotation for one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    #: Unified class index, already resolved by the subclass.
    class_id: int
    #: MOT identity when the publisher supplies one.
    track_id: int | None = None
    #: Visibility in [0, 1]. Anti-UAV uses 0 for "target absent".
    visibility: float = 1.0
    #: False when the publisher explicitly marks the target as absent.
    present: bool = True


class VideoConverter(Converter):
    """Base for datasets distributed as ``one directory per video``."""

    #: Modality label for frames from the primary video stream.
    primary_modality: str = "rgb"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.stride = self.spec.ingest.stride
        self.max_frames = self.spec.ingest.max_frames_per_sequence

    # -- subclass contract -------------------------------------------------- #

    @abstractmethod
    def parse_annotations(self, video: Path, sequence_id: str) -> dict[int, list[Det]]:
        """``{original_frame_index: [Det, ...]}`` for one video.

        Must key on the *original* frame index so the caller can apply stride
        without shifting labels.
        """

    def sequence_id_for(self, video: Path) -> str:
        """Stable sequence id. Defaults to the slugified parent directory name."""
        from ..frameindex import slugify

        return slugify(video.parent.name if video.parent != self.raw_root else video.stem)

    # -- shared implementation ---------------------------------------------- #

    def _convert(self) -> None:
        sequence_dirs = self._sequence_dirs()
        if not sequence_dirs:
            raise FileNotFoundError(
                f"{self.alias}: found no sequence directories under {self.raw_root}. "
                f"Expected one directory per video, each containing the video and its "
                f"annotation file."
            )

        self.note(
            f"Extracting with stride={self.stride}, so roughly 1/{self.stride} of source "
            f"frames are kept. Raise the stride if disk is tight; drop to 1 only when "
            f"measuring detector precision rather than training."
        )

        failures = 0
        for index, sequence_dir in enumerate(sequence_dirs):
            video = self._primary_video(sequence_dir)
            if video is None:
                self.note(f"{sequence_dir.name}: no recognised video file, skipped.")
                continue

            sequence_id = self.sequence_id_for(video)
            try:
                annotations = self.parse_annotations(video, sequence_id)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                message = (
                    f"{sequence_dir.name}: annotation parse failed "
                    f"({type(exc).__name__}: {exc}); sequence skipped"
                )
                if message not in self.report.errors:
                    self.report.errors.append(message)
                log.warning("annotation parse failed", extra={"sequence": sequence_dir.name})
                continue

            emitted = self._extract(video, sequence_id, annotations)

            if (index + 1) % 10 == 0 or index + 1 == len(sequence_dirs):
                log.info(
                    "video progress",
                    extra={
                        "dataset": self.alias,
                        "done": index + 1,
                        "total": len(sequence_dirs),
                        "last_sequence": sequence_id,
                        "last_frames": emitted,
                        "frames_total": self.report.frames_written,
                    },
                )

        if failures:
            self.note(
                f"{failures} of {len(sequence_dirs)} sequences failed annotation parsing. "
                f"Check the format assumption in the converter before trusting the class "
                f"histogram - a format change usually shows up as zero boxes here."
            )

    def _extract(
        self,
        video: Path,
        sequence_id: str,
        annotations: dict[int, list[Det]],
    ) -> int:
        emitted = 0
        labels = self.spec.classes.unified_labels

        for frame_index, image in frames_from_video(
            video, stride=self.stride, max_frames=self.max_frames
        ):
            height, width = (int(v) for v in image.shape[:2])
            dets = annotations.get(frame_index, [])

            boxes: list[tuple[int, float, float, float, float]] = []
            source_labels: list[str] = []
            visibility: list[float] = []
            dropped = 0

            for det in dets:
                if not det.present:
                    continue
                converted = self._to_yolo(det, width, height)
                if converted is None:
                    dropped += 1
                    continue
                boxes.append(converted)
                source_labels.append(labels[converted[0]])
                visibility.append(det.visibility)

            self.emit_frame(
                sequence_id=sequence_id,
                frame_index=emitted,
                modality=self.primary_modality,
                source=image,
                boxes=boxes,
                source_labels=source_labels,
                track_id=next((d.track_id for d in dets if d.track_id is not None), None),
                visibility=visibility,
                target_present=bool(boxes),
                dropped_boxes=dropped,
                width=width,
                height=height,
                notes=f"src_frame={frame_index}",
            )
            emitted += 1

        return emitted

    @staticmethod
    def _to_yolo(
        det: Det, width: int, height: int
    ) -> tuple[int, float, float, float, float] | None:
        """Convert one annotation to a normalised YOLO tuple, or ``None`` to drop."""
        converted = boxes_from_xyxy(
            [(det.class_id, det.x1, det.y1, det.x2, det.y2)], width=width, height=height
        )
        if not converted:
            return None
        box = converted[0]
        if box[3] * width < MIN_BOX_SIDE_PX or box[4] * height < MIN_BOX_SIDE_PX:
            return None
        return box

    def _sequence_dirs(self) -> list[Path]:
        """Directories that contain at least one video file."""
        out: list[Path] = []
        for path in sorted(self.raw_root.rglob("*")):
            if not path.is_dir():
                continue
            for child in path.iterdir():
                if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES:
                    out.append(path)
                    break
        return out

    def _primary_video(self, sequence_dir: Path) -> Path | None:
        """Prefer a file named after the directory, else the largest video."""
        exact = sequence_dir / f"{sequence_dir.name}.mp4"
        if exact.is_file():
            return exact
        videos = [
            p for p in sequence_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        ]
        if not videos:
            return None
        return max(videos, key=lambda p: p.stat().st_size)