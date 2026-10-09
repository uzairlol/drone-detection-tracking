"""Inference: single frame, batch, video, and the tracker-facing path.

The predictor is deliberately thin. It exists to:

* resolve a run directory or a checkpoint to an ultralytics model;
* apply the settings from ``configs/app.yaml`` - note ``conf_threshold`` there is
  **0.25**, well below the rule layer's ``confidence.initiate`` of 0.60. That is
  intentional: the tracker needs a permissive detector to hold a track through a
  low-confidence stretch, and the rule layer decides what actually becomes an
  alert. Raising the detector threshold to 0.60 starves the tracker;
* normalise ultralytics output into plain numpy so the tracking code never
  depends on ultralytics internals. This matters more than it looks: the tracker,
  the rule engine and the API all consume :class:`Detection`, and only the
  predictor knows about ``Boxes``/``Results``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.loader import load_settings
from ..config.schema import AppSettings, ModelFamily, TrackerName
from ..utils.imaging import read_image
from ..utils.logging import get_logger
from ..utils.paths import relative_to_root

log = get_logger(__name__)


@dataclass(slots=True)
class Detection:
    """One detected object in pixel coordinates."""

    box: tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int
    class_name: str = ""
    #: Track id, filled in by a tracker. ``None`` before association.
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def height_px(self) -> float:
        return self.box[3] - self.box[1]

    def xyxy_array(self) -> np.ndarray:
        return np.asarray(self.box, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(v, 2) for v in self.box],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass(slots=True)
class FrameResult:
    """Everything produced for one frame."""

    frame_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    #: Populated by the tracker: ``{track_id: [Detection, ...]}``.
    tracks: dict[int, list[Detection]] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    source: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def drone_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name == "drone" or d.class_id == 0]

    @property
    def bird_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name == "bird" or d.class_id == 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "detections": [d.to_dict() for d in self.detections],
            "tracks": {str(k): [d.to_dict() for d in v] for k, v in self.tracks.items()},
        }


def resolve_weights(run_or_path: str | Path) -> Path:
    """Accept a checkpoint, a run directory, or a run name under ``artifacts/runs``."""
    from ..utils.paths import subdir

    candidate = Path(run_or_path)
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        for name in ("best.pt", "last.pt"):
            weights = candidate / "weights" / name
            if weights.is_file():
                return weights
            direct = candidate / name
            if direct.is_file():
                return direct
        found = sorted(candidate.rglob("*.pt"))
        if found:
            return found[0]
        raise FileNotFoundError(f"no .pt weights under {candidate}")

    searched = [subdir("runs")]
    for root in searched:
        for name in ("best.pt", "last.pt"):
            direct = root / candidate / "weights" / name
            if direct.is_file():
                return direct
            direct = root / candidate / name
            if direct.is_file():
                return direct

    raise FileNotFoundError(
        f"could not resolve {run_or_path!r} to a checkpoint.\n"
        f"Looked at the path itself and under {subdir('runs')}.\n"
        f"List what exists with: anti-uav list-runs"
    )


def infer_family(run_or_path: str | Path) -> ModelFamily:
    """Which family a checkpoint belongs to, so the right ultralytics class loads."""
    path = resolve_weights(run_or_path)
    name = path.name.lower()
    parent = path.parent.name.lower()
    combined = f"{parent}/{name}"
    if "rtdetr" in combined or "rtdetr" in str(run_or_path).lower():
        if "x2" in combined:
            return ModelFamily.RTDETR_X2
        return ModelFamily.RTDETR_L
    return ModelFamily.YOLO11N


class Detector:
    """A loaded detection model. Use as a context manager to free the model."""

    def __init__(
        self,
        run_or_path: str | Path,
        settings: AppSettings | None = None,
        *,
        family: ModelFamily | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        half: bool | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.weights = resolve_weights(run_or_path)
        self.family = family or infer_family(run_or_path)
        self.conf = conf if conf is not None else self.settings.conf_threshold
        self.iou = iou if iou is not None else self.settings.iou_threshold
        self.imgsz = imgsz or self.settings.imgsz
        self.device = device or self.settings.device
        self.half = self.settings.half if half is None else half
        self.class_names: dict[int, str] = {}
        self._model: Any = None

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> Detector:
        if self._model is not None:
            return self

        import ultralytics

        cls = ultralytics.RTDETR if self.family.value.startswith("rtdetr") else ultralytics.YOLO
        self._model = cls(str(self.weights))

        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.class_names = {i: str(n) for i, n in enumerate(names)}
        else:
            self.class_names = {0: "drone", 1: "bird"}

        log.info(
            "detector loaded",
            extra={
                "weights": relative_to_root(self.weights),
                "family": self.family.value,
                "classes": self.class_names,
                "conf": self.conf,
                "imgsz": self.imgsz,
                "device": self.device,
            },
        )
        return self

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> Detector:
        return self.load()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- inference ---------------------------------------------------------- #

    def predict_image(self, image: np.ndarray | str | Path, *, conf: float | None = None) -> FrameResult:
        """Detect in one BGR array or image path.

        ``conf`` overrides the detector's default for this call only, which is
        what :meth:`predict_stream` needs to honour a caller-supplied threshold
        without mutating shared state.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            array = read_image(path)
            source = relative_to_root(path)
        else:
            array = image
            source = ""

        height, width = array.shape[:2]
        results = self._call(array, **({"conf": conf} if conf is not None else {}))
        return FrameResult(
            frame_index=0,
            timestamp_s=0.0,
            detections=self._to_detections(results, width, height),
            width=width,
            height=height,
            source=source,
        )

    def predict_batch(self, images: Sequence[np.ndarray]) -> list[FrameResult]:
        out: list[FrameResult] = []
        for index, image in enumerate(images):
            result = self.predict_image(image)
            result.frame_index = index
            out.append(result)
        return out

    def predict_stream(
        self,
        source: str | Path,
        *,
        tracker: TrackerName | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        conf: float | None = None,
    ) -> Iterator[FrameResult]:
        """Yield a :class:`FrameResult` per frame of a video or a directory.

        Tracking is done by this repo's own trackers, not by ultralytics'.
        ultralytics 8.4 ships no ``track`` task - ``persist`` and ``tracker``
        are not in its predict cfg, so ``model.predict(..., persist=True)``
        raises ``SyntaxError: 'persist' is not a valid YOLO argument``. Using
        :func:`anti_uav.tracking.replayer.build_tracker` instead also means the
        video path and the metrics harness share one association implementation,
        so a number produced by ``track-eval`` describes what this stream does.

        Pass ``tracker=None`` for plain per-frame detection with no association.
        """
        from ..tracking.replayer import build_tracker
        from ..tracking.types import TrackObservation
        from ..utils.imaging import frames_from_video

        active_tracker = tracker
        path = Path(source)
        fps = _fps_of(path)
        threshold = conf if conf is not None else self.conf

        local = build_tracker(active_tracker) if active_tracker is not None else None

        for frame_index, frame in frames_from_video(path, stride=stride, max_frames=max_frames):
            # predict_image() always reports frame_index 0, since it sees one
            # frame at a time. The stream has to stamp the real index itself or
            # every yielded result claims to be frame 0.
            frame_result = self.predict_image(frame, conf=threshold)
            frame_result.frame_index = frame_index
            height, width = frame.shape[:2]

            if local is not None:
                timestamp = frame_index / fps if fps else 0.0
                tracks = local.update(
                    [
                        TrackObservation(
                            frame_index=frame_index,
                            timestamp_s=timestamp,
                            box=d.box,
                            confidence=d.confidence,
                            class_id=d.class_id,
                            class_name=d.class_name,
                            image_height_px=height,
                        )
                        for d in frame_result.detections
                    ],
                    frame_index=frame_index,
                    timestamp_s=timestamp,
                )
                for track in tracks.tracks:
                    for detection in frame_result.detections:
                        if detection.track_id is None and _overlaps(detection.box, track.box):
                            detection.track_id = track.track_id

            frame_result.width = width
            frame_result.height = height
            frame_result.timestamp_s = frame_index / fps if fps else 0.0
            frame_result.source = relative_to_root(path)
            yield frame_result

    # -- internals ---------------------------------------------------------- #

    def _call(self, image: np.ndarray, **overrides: Any) -> Any:
        self.load()
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.half,
            "verbose": False,
        }
        kwargs.update(overrides)
        results = self._model.predict(image, **kwargs)
        return results[0] if isinstance(results, (list, tuple)) else results

    def _to_detections(self, result: Any, width: int, height: int) -> list[Detection]:
        """Ultralytics ``Results`` -> plain :class:`Detection` list.

        RT-DETR is NMS-free: its decoder emits a fixed, already-deduplicated set
        of queries. The boxes come through the same ``.boxes`` attribute, so this
        path is identical for both families - which is exactly why RT-DETR can
        feed the same tracker.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.ids.cpu().numpy() if hasattr(boxes, "ids") else None
        except (AttributeError, RuntimeError):  # pragma: no cover
            return []

        if classes is None:
            classes = np.zeros(len(confs), dtype=int)

        out: list[Detection] = []
        for index in range(len(confs)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[index])
            if x2 - x1 <= 0.5 or y2 - y1 <= 0.5:
                continue
            class_id = int(classes[index])
            out.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(confs[index]),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                )
            )
        return out


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Do two xyxy boxes share more than half of the smaller one's area?

    Used only to copy a track id back onto the detection that produced it. A
    containment test rather than a strict IoU threshold, because a track box is
    the smoothed history while the detection is one frame's measurement, and the
    two can differ by more than IoU 0.5 on a small fast-moving target.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return False
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return smaller > 0.0 and inter / smaller > 0.5


def _group_by_track(detections: Sequence[Detection]) -> dict[int, list[Detection]]:
    groups: dict[int, list[Detection]] = {}
    for detection in detections:
        if detection.track_id is None:
            continue
        groups.setdefault(int(detection.track_id), []).append(detection)
    return groups


def _fps_of(path: Path) -> float:
    from ..utils.imaging import video_metadata

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return 30.0
    try:
        return float(video_metadata(path)["fps"]) or 30.0
    except (OSError, ValueError):
        return 30.0


def overlay(
    image: np.ndarray,
    result: FrameResult,
    *,
    show_track_ids: bool = True,
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw boxes, labels and track ids. Used by the UI and the CLI.

    Kept dependency-free (cv2 only) so the same rendering appears in the API
    preview and in any offline replay.
    """
    import cv2

    palette = colors or {
        "drone": (48, 48, 235),   # BGR red
        "bird": (40, 176, 64),    # BGR green
        "unknown": (200, 200, 200),
    }
    canvas = image.copy()

    for detection in result.detections:
        x1, y1, x2, y2 = (int(v) for v in detection.box)
        colour = palette.get(detection.class_name, palette["unknown"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        parts = [f"{detection.class_name} {detection.confidence:.2f}"]
        if show_track_ids and detection.track_id is not None:
            parts.insert(0, f"#{detection.track_id}")
        label = "  ".join(parts)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(0, y1 - baseline - 4)
        cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + baseline + 4), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 3, ty + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return canvas
