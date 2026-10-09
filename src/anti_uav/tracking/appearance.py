"""Appearance embeddings for the cross-camera association gate.

BoT-SORT's appearance term needs a vector per detection. This module provides
that in a way that degrades gracefully, because the embedding source is not
always available:

1. A **ReID checkpoint** (the intended path) - trained by
   :mod:`anti_uav.tracking.reid_train` on this project's own detection crops.
2. **Detector backbone features** - the detector's penultimate features, ROI-pooled
   over the detection box. Weaker (never trained to make crops of the same object
   similar) but free and always present, so the appearance gate is exercisable
   before the ReID model exists.
3. **A cheap colour+geometry descriptor** - a normalised colour histogram plus an
   aspect-ratio/sharpness term. Deliberately crude, and labelled as such: it is
   the fallback that lets the gate be *tested* on a CPU box, and it is not a
   ReID model and must not be reported as one.

The distinction matters because a weak appearance model is worse than none: it
will confidently reject correct associations. ``AppearanceEmbedder.describe``
names the source so no result is ever attributed to "ReID" when it came from a
colour histogram.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.geometry import BBox
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Side length of the square crop fed to the embedder. Small on purpose: a 15 px
#: target upsampled to 256 wastes almost all of the input on interpolation.
CROP_SIZE = 64

#: Minimum crop side in pixels. Below this the crop is mostly padding.
MIN_CROP_PX = 4.0

#: Context margin added around the box before cropping. Detection boxes are tight,
#: so a zero margin loses the rotor span that distinguishes one drone from another.
CROP_CONTEXT = 0.20


@dataclass(slots=True)
class AppearanceEmbedder:
    """Produces a unit-norm embedding for a detection crop."""

    source: str = "none"
    model: Any = None
    input_size: int = CROP_SIZE
    #: Set when the embedder came from a trained ReID checkpoint.
    is_reid: bool = False

    # -- crops -------------------------------------------------------------- #

    def crop(self, image: np.ndarray, box: BBox, *, context: float = CROP_CONTEXT) -> np.ndarray:
        return crop_box(image, box, context=context, out_size=self.input_size)

    def crops(self, image: np.ndarray, boxes: Sequence[BBox] | np.ndarray) -> list[np.ndarray]:
        return [self.crop(image, tuple(b)) for b in np.asarray(boxes, dtype=float).reshape(-1, 4)]

    # -- embeddings --------------------------------------------------------- #

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Unit-norm embedding, or ``None`` when this crop cannot be embedded.

        Returning ``None`` (rather than a zero vector) is important: the tracker's
        gate treats a missing embedding as neutral, and a zero vector would read as
        "maximally different" and veto every association.
        """
        if self.source == "none" or crop.size == 0:
            return None

        if self.source == "color":
            return colour_descriptor(crop)

        if self.source == "detector":
            features = self._detector_features(crop)
            return None if features is None else features

        if self.source == "reid":
            features = self._reid_features(crop)
            return None if features is None else features

        return colour_descriptor(crop)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        return [self.embed(crop) for crop in crops]

    # -- backends ----------------------------------------------------------- #

    def _reid_features(self, crop: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            import torch

            array = _to_tensor(crop, self.input_size)
            with torch.no_grad():
                tensor = array.to(next(self.model.parameters()).device)
                features = self.model(tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                vector = features.flatten(1).cpu().numpy()[0]
            return _normalise(vector)
        except Exception as exc:
            log.debug("reid forward failed", extra={"reason": str(exc)})
            return None

    def _detector_features(self, crop: np.ndarray) -> np.ndarray | None:
        """ROI-pool the detector's penultimate features.

        Requires a loaded ultralytics model whose ``model`` attribute exposes
        intermediate tensors, which differs across architectures. Returns ``None``
        when it cannot be reached, so the caller falls back rather than erroring.
        """
        if self.model is None:
            return None
        try:
            import torch

            array = _to_tensor(crop, self.input_size)
            with torch.no_grad():
                backbone = getattr(self.model, "model", None)
                if backbone is None:
                    return None
                tensor = array.to(next(backbone.parameters()).device)
                tensor = backbone.model[tensor] if hasattr(backbone, "model") else backbone(tensor)
                if isinstance(tensor, (tuple, list)):
                    tensor = tensor[-1]
                vector = tensor.flatten(1).cpu().numpy()[0]
            return _normalise(vector)
        except Exception as exc:
            log.debug("detector feature extraction failed", extra={"reason": str(exc)})
            return None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def load_default(
        cls,
        checkpoint: str | Path | None = None,
        *,
        input_size: int = CROP_SIZE,
        prefer_reid: bool = True,
    ) -> AppearanceEmbedder:
        """Best available embedder.

        Tries, in order: a trained ReID checkpoint (explicit path, else the
        conventional location), the loaded detector's features, then the colour
        descriptor. Never raises - an unavailable appearance model is a normal
        state, not an error.
        """
        if prefer_reid:
            path = Path(checkpoint) if checkpoint else default_checkpoint_path()
            if path and path.is_file():
                try:
                    return cls.load_reid(path, input_size=input_size)
                except Exception as exc:
                    log.warning(
                        "ReID checkpoint found but could not be loaded; falling back",
                        extra={"path": str(path), "reason": str(exc)},
                    )

        return cls(source="color", model=None, input_size=input_size, is_reid=False)

    @classmethod
    def load_reid(cls, path: str | Path, *, input_size: int = CROP_SIZE) -> AppearanceEmbedder:
        """Load a trained ReID classifier as a feature extractor.

        The training script in this repo fine-tunes a small backbone with an
        identity-classification head. The classification logits are not the right
        features - what we want is the pooled embedding beneath them, which is what
        makes two crops of the same target similar. We take the penultimate
        pooled features.
        """
        import torch

        from .reid_model import ARCHITECTURES, ReIdNet

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        config = state.get("config", {}) if isinstance(state, dict) else {}
        embedding_dim = int(config.get("embedding_dim", 512))

        # train_reid always writes `arch` into the checkpoint, so this default
        # only applies to a hand-made or third-party file. It used to be
        # "osnet_x0_25", which ReIdNet cannot build: the name is not in
        # ARCHITECTURES, so construction raised ValueError and load_default
        # swallowed it and silently fell back to the colour descriptor. Falling
        # back quietly is the failure mode here, so the default is a real
        # architecture and a bad value is reported rather than ignored.
        arch = str(config.get("arch") or "tiny_cnn")
        if arch not in ARCHITECTURES:
            raise ValueError(
                f"ReID checkpoint {str(path)!r} declares arch={arch!r}, which is not one of "
                f"{ARCHITECTURES}. Either the checkpoint was written by a different version "
                f"of this project or the key is corrupt."
            )

        model = ReIdNet(arch=arch, num_classes=int(config.get("num_classes", 0)), embedding_dim=embedding_dim)
        model.load_state_dict(state.get("model", state))
        model.eval()

        log.info(
            "reid model loaded",
            extra={"arch": arch, "dim": embedding_dim, "path": str(path)},
        )
        return cls(source="reid", model=model, input_size=input_size, is_reid=True)

    @classmethod
    def from_detector(cls, detector: Any, *, input_size: int = CROP_SIZE) -> AppearanceEmbedder:
        """Wrap a loaded ultralytics detector's backbone."""
        return cls(source="detector", model=detector, input_size=input_size, is_reid=False)

    def describe(self) -> str:
        if self.is_reid:
            return f"trained ReID ({self.model.__class__.__name__ if self.model else 'unloaded'})"
        if self.source == "detector":
            return "detector backbone features (NOT a trained ReID model)"
        if self.source == "color":
            return "colour + geometry descriptor (NOT a ReID model; for gate testing only)"
        return "none"


def default_checkpoint_path() -> Path | None:
    """Where :mod:`anti_uav.tracking.reid_train` writes.

    ``train_reid`` writes to ``artifacts/runs/reid/<dataset>_<variant>_<arch>/best.pt``,
    one directory per training run, so the newest of those is the default. A
    top-level ``artifacts/runs/reid/best.pt`` is also honoured if someone put one
    there by hand.

    This used to look only for the top-level path, which no training run ever
    produces — so the trained-ReID tier was unreachable unless the caller passed
    an explicit checkpoint, and the appearance gate silently ran on the colour
    descriptor instead.
    """
    from ..utils.paths import subdir

    root = subdir("runs") / "reid"
    direct = root / "best.pt"
    if direct.is_file():
        return direct
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.glob("*/best.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def crop_box(
    image: np.ndarray,
    box: BBox,
    *,
    context: float = CROP_CONTEXT,
    out_size: int = CROP_SIZE,
) -> np.ndarray:
    """Crop ``box`` with a margin, clamped, then resized to ``out_size``.

    Clamps rather than padding: a box hanging off the frame edge should give the
    largest crop that exists, not a mostly-grey one.
    """
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box

    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    margin_x = box_width * context
    margin_y = box_height * context

    cx1 = int(max(0, round(x1 - margin_x)))
    cy1 = int(max(0, round(y1 - margin_y)))
    cx2 = int(min(width, round(x2 + margin_x)))
    cy2 = int(min(height, round(y2 + margin_y)))

    if cx2 - cx1 < MIN_CROP_PX or cy2 - cy1 < MIN_CROP_PX:
        cx1 = int(np.clip(cx1, 0, max(width - 1, 0)))
        cy1 = int(np.clip(cy1, 0, max(height - 1, 0)))
        cx2 = int(np.clip(cx2, cx1 + 1, width))
        cy2 = int(np.clip(cy2, cy1 + 1, height))

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def colour_descriptor(crop: np.ndarray, *, bins: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Normalised HSV histogram plus shape terms.

    This is **not** a ReID embedding and must not be reported as one. It captures
    the two things that differ most between a drone and a bird in these datasets at
    tiny scales - overall tone and wing-vs-body ratio - which is enough to make the
    appearance gate behave plausibly when testing the handoff path on a CPU box,
    and nowhere near enough for real re-identification.
    """
    import cv2

    if crop.size == 0:
        return _normalise(np.zeros(bins[0] * bins[1] + 4, dtype=np.float32))

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shape = np.array(
        [
            grey.mean() / 255.0,
            grey.std() / 255.0,
            crop.shape[1] / max(crop.shape[0], 1),
            float(cv2.Laplacian(grey, cv2.CV_32F).var()) / 1000.0,
        ],
        dtype=np.float32,
    )
    return _normalise(np.concatenate([histogram.astype(np.float32), shape]))


def _to_tensor(crop: np.ndarray, size: int):
    """BGR ``uint8`` HWC crop -> normalised NCHW float tensor."""
    import torch

    if crop.shape[0] != size or crop.shape[1] != size:
        import cv2

        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    array = crop.astype(np.float32) / 255.0
    # BGR -> RGB, then ImageNet normalisation, matching the ReID training script.
    rgb = array[:, :, ::-1]
    normalised = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(np.transpose(normalised, (2, 0, 1)))[None]


def _normalise(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        return array
    return array / norm


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity, with the neutral 1.0 for a missing side.

    Neutral rather than zero on purpose: "we have no appearance evidence" is not
    the same claim as "the appearance evidence disagrees", and conflating them
    makes the gate veto every association whenever ReID is unavailable.
    """
    if a is None or b is None:
        return 1.0
    return float(np.dot(_normalise(a), _normalise(b)))


def gallery_similarity(
    gallery: Sequence[np.ndarray], embedding: np.ndarray | None
) -> float:
    """Maximum similarity against a track's gallery.

    Max rather than mean: a target that was briefly occluded looks different on
    re-entry, and averaging penalises that heavily. Max asks the right question -
    "does this look like *any* view we have seen of this target".
    """
    if not gallery or embedding is None:
        return 1.0
    return max(cosine(candidate, embedding) for candidate in gallery)
