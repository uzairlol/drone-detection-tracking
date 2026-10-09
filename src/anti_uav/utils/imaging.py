"""Image I/O, perceptual hashing and tile maths.

Tiling matters here: MM-UAV's average target is 12x5 px and Drone-vs-Bird's is
34x23 px. At a 640 px letterboxed input a 12x5 px drone shrinks to roughly
6x3 px, which is below what any of these detectors reliably sees. Slicing the
source image into overlapping tiles and training on the tiles restores the
target to a workable pixel scale - see :func:`tile_grid` and
``configs/train/tiling.yaml``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .logging import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGE_SUFFIX_SET = frozenset(IMAGE_SUFFIXES)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR ``uint8``. Prefers cv2, falls back to Pillow."""
    p = Path(path)
    try:
        import cv2

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {p}")
        return img
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        from PIL import Image

        with Image.open(p) as handle:
            return np.asarray(handle.convert("RGB"))[:, :, ::-1].copy()


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Write a BGR ``uint8`` array, creating parent directories."""
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"failed to write image: {target}")
    return target


def image_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` without decoding the pixels where possible."""
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
        if img is not None:
            return (img.shape[1] * 8, img.shape[0] * 8)
    except ImportError:  # pragma: no cover
        pass
    img = read_image(path)
    return (int(img.shape[1]), int(img.shape[0]))


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIX_SET


def list_images(root: str | Path, *, recursive: bool = True) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    globber = base.rglob if recursive else base.glob
    return sorted(p for p in globber("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX_SET)


def letterbox(
    image: np.ndarray,
    new_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio, then pad - the YOLO convention.

    Returns ``(image, scale, (pad_x, pad_y))`` so boxes can be mapped back.
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = new_size[1], new_size[0]
    scale = min(dst_h / max(src_h, 1), dst_w / max(src_w, 1))

    resized_w = round(src_w * scale)
    resized_h = round(src_h * scale)

    import cv2

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (dst_w - resized_w) / 2.0
    pad_y = (dst_h - resized_h) / 2.0
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, (pad_x, pad_y)


def unletterbox_box(
    box: tuple[float, float, float, float],
    scale: float,
    pad: tuple[float, float],
    src_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Inverse of :func:`letterbox` for a single box."""
    src_w, src_h = src_size
    pad_x, pad_y = pad
    x1, y1 = (box[0] - pad_x) / scale, (box[1] - pad_y) / scale
    x2, y2 = (box[2] - pad_x) / scale, (box[3] - pad_y) / scale
    x1 = min(max(x1, 0.0), src_w)
    x2 = min(max(x2, 0.0), src_w)
    y1 = min(max(y1, 0.0), src_h)
    y2 = min(max(y2, 0.0), src_h)
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# perceptual hashing - near-duplicate guard
# --------------------------------------------------------------------------- #


def phash(image: np.ndarray | str | Path, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """64-bit DCT perceptual hash as hex.

    Used by ``anti-uav dedup`` to catch the same frame appearing under two
    filenames - which happens constantly when a video is re-extracted at a
    slightly different stride, or when two datasets were shot at the same site.
    """
    import cv2

    side = hash_size * highfreq_factor
    grey = read_image(image) if isinstance(image, (str, Path)) else image
    if grey.ndim == 3:
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)

    dct = cv2.dct(resized)
    low = dct[:hash_size, :hash_size]
    med = float(np.median(np.asarray(low[1:] if low.size > 1 else low, dtype=np.float64)))

    bits = (low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex hashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def content_hash(path: str | Path, *, block: int = 1 << 20) -> str:
    """Exact SHA-256 of file bytes - for manifests and integrity checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Tile:
    """One window over a source image."""

    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def name(self) -> str:
        return f"t{self.index:04d}_x{self.x}_y{self.y}_w{self.width}_h{self.height}"


def tile_grid(
    width: int,
    height: int,
    *,
    tile_size: int = 640,
    overlap: float = 0.25,
) -> list[Tile]:
    """Cut an image into overlapping ``tile_size`` windows.

    A drone straddling a tile seam lands in at least one tile whole, which is
    why ``overlap`` defaults to 0.25 rather than 0. Tiles are emitted in
    row-major order and clipped at the borders so no window leaves the image.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    stride = max(round(tile_size * (1.0 - overlap)), 1)
    xs = _tile_starts(width, tile_size, stride)
    ys = _tile_starts(height, tile_size, stride)

    tiles: list[Tile] = []
    for y in ys:
        for x in xs:
            tw = min(tile_size, width - x)
            th = min(tile_size, height - y)
            if tw < 8 or th < 8:
                continue  # sliver windows produce nothing but false positives
            tiles.append(Tile(index=len(tiles), x=x, y=y, width=tw, height=th))
    return tiles


def _tile_starts(extent: int, tile_size: int, stride: int) -> list[int]:
    """Start offsets for full-width windows across ``extent``.

    The final window always ends exactly at ``extent`` so the image is fully
    covered. When the rounded stride puts that tail window uncomfortably close to
    its predecessor - e.g. 358 and 384 for a 256 px tile over 640 px - the
    *predecessor* is dropped rather than the tail. The two windows would be
    near-identical images carrying near-identical labels, so keeping both wastes
    training time and teaches nothing, whereas dropping the tail would leave the
    image's edge uncovered and any target sitting there undetected.
    """
    if extent <= tile_size:
        return [0]

    starts = list(range(0, extent - tile_size + 1, stride))
    tail = extent - tile_size
    if starts[-1] != tail:
        starts.append(tail)

    min_gap = max(1, tile_size // 8)
    while len(starts) > 1 and starts[-1] - starts[-2] < min_gap:
        starts.pop(-2)
    return starts


def tile_boxes(
    boxes: np.ndarray,
    tile: Tile,
    *,
    min_visibility: float = 0.35,
    clip: bool = True,
) -> np.ndarray:
    """Map ``(N, 4)`` xyxy boxes from image space into a tile.

    Boxes that fall mostly outside the tile are dropped rather than clipped to a
    sliver: a 3 px fragment of a bird wing is a false-positive factory.
    """
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)

    tile_box = np.asarray(tile.box, dtype=float)
    areas = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None)
    areas = areas[:, 0] * areas[:, 1]

    lt = np.maximum(boxes[:, :2], tile_box[:2])
    rb = np.minimum(boxes[:, 2:], tile_box[2:])
    wh = np.clip(rb - lt, 0.0, None)
    visible = wh[:, 0] * wh[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(areas > 0, visible / np.maximum(areas, 1e-9), 0.0)

    keep = ratio >= min_visibility
    if not np.any(keep):
        return np.empty((0, 4), dtype=float)

    # copy() because the tile offset is applied in place below; without it a
    # caller that passed a view into a larger array would see its data move.
    lt, rb = lt[keep], rb[keep]
    result = np.concatenate([lt, rb], axis=1) if clip else boxes[keep].copy()
    result[:, 0] -= tile.x
    result[:, 1] -= tile.y
    return result


def tile_fraction(boxes: np.ndarray, tile: Tile) -> float:
    """What fraction of the target area a tile captures. Drives dedup in the
    post-tile NMS pass."""
    if boxes.size == 0:
        return 0.0
    tile_box = np.asarray(tile.box, dtype=float)
    lt = np.maximum(boxes[:, :2], tile_box[:2])
    rb = np.minimum(boxes[:, 2:], tile_box[2:])
    wh = np.clip(rb - lt, 0.0, None)
    visible = (wh[:, 0] * wh[:, 1]).sum()
    areas = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None)
    total = float((areas[:, 0] * areas[:, 1]).sum())
    return visible / total if total > 0 else 0.0


def scale_box(
    box: Sequence[float],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Rescale an xyxy box between two image sizes."""
    fw, fh = from_size
    tw, th = to_size
    sx = tw / max(fw, 1)
    sy = th / max(fh, 1)
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def augment_hsv(
    image: np.ndarray,
    *,
    h_gain: float = 0.015,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
    p: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cheap HSV jitter for IR/RGB mixing.

    Our datasets mix thermal infrared (grey, high contrast) with visible RGB. A
    detector trained on the raw mix tends to key on palette. Randomising
    saturation around 0 desaturates the whole batch so colour stops being a
    shortcut for modality.
    """
    import cv2

    if p <= 0.0 or (rng is None and np.random.rand() > p):
        return image
    generator = rng or np.random.default_rng()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(generator.integers(-1, 2) * h_gain * 180)) % 180
    hsv[..., 1] = np.clip(
        hsv[..., 1].astype(np.float32) * (1.0 + generator.uniform(-s_gain, s_gain)), 0, 255
    )
    hsv[..., 2] = np.clip(
        hsv[..., 2].astype(np.float32) * (1.0 + generator.uniform(-v_gain, v_gain)), 0, 255
    )
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def cutout(image: np.ndarray, *, n_holes: int = 3, size: int = 48, p: float = 0.5, rng=None) -> np.ndarray:
    """Random rectangles filled with noise.

    The block diagram's whole premise is that birds and clutter produce false
    positives; cutout is the cheapest way to stop the detector relying on a
    clean, uncorrupted background.
    """
    if n_holes <= 0 or p <= 0.0 or (rng is None and np.random.rand() > p):
        return image
    generator = rng or np.random.default_rng()
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(n_holes):
        ch, cw = generator.integers(size // 2, size + 1, size=2)
        y = int(generator.integers(0, max(h - ch, 1)))
        x = int(generator.integers(0, max(w - cw, 1)))
        out[y : y + ch, x : x + cw] = generator.integers(0, 256, size=(ch, cw, 3), dtype=np.uint8)
    return out


def frames_from_video(
    path: str | Path,
    *,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_image)`` from a video file.

    ``stride`` is the temporal subsampling factor. It is the single most
    effective knob for these datasets: adjacent frames of a hovering drone are
    near-identical, so keeping every frame inflates dataset size roughly linearly
    while adding almost no information, and it is the main driver of train/val
    leakage if you forget it.
    """
    import cv2

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        log.debug(
            "opened video",
            extra={"file": video_path.name, "frames": total, "fps": round(fps, 2)},
        )

        index = 0
        emitted = 0
        while True:
            if max_frames is not None and emitted >= max_frames:
                return
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start_frame and (index - start_frame) % stride == 0:
                yield index, frame
                emitted += 1
            index += 1
    finally:
        capture.release()


def video_metadata(path: str | Path) -> dict[str, float | int]:
    """``{frames, fps, width, height, duration_s}`` for a video."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {
            "frames": frames,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "duration_s": round(frames / fps, 3) if fps else 0.0,
        }
    finally:
        capture.release()


def mosaic_batch(
    images: Sequence[np.ndarray],
    *,
    labels: Sequence[np.ndarray] | None = None,
    output_size: int = 640,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reference 4-image mosaic (kept for tests; ultralytics does this in C).

    Only used to unit-test the label remapping logic that the tiling code shares.
    """
    import cv2

    if len(images) != 4:
        raise ValueError("mosaic needs exactly 4 images")
    generator = rng or np.random.default_rng()

    canvas = np.full((output_size * 2, output_size * 2, 3), 114, dtype=np.uint8)
    canvas_labels: list[np.ndarray] = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        scale = min(output_size / w, output_size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        ox = int(generator.integers(0, output_size - nw + 1))
        oy = int(generator.integers(0, output_size - nh + 1))
        canvas[oy : oy + nh, ox : ox + nw] = resized

        if labels is not None:
            box_labels = np.asarray(labels[i], dtype=float).reshape(-1, 5).copy()
            if box_labels.size:
                box_labels[:, 0] = box_labels[:, 0] * w
                box_labels[:, 1] = box_labels[:, 1] * h
                box_labels[:, 2] = box_labels[:, 2] * w
                box_labels[:, 3] = box_labels[:, 3] * h
                box_labels[:, 0] = (box_labels[:, 0] - ox) / scale
                box_labels[:, 1] = (box_labels[:, 1] - oy) / scale
                box_labels[:, 2] = (box_labels[:, 2] - ox) / scale
                box_labels[:, 3] = (box_labels[:, 3] - oy) / scale
                canvas_labels.append(box_labels)

    final, _ = cv2.resize(canvas, (output_size, output_size)), None
    return final, canvas_labels


def focus_score(image: np.ndarray) -> float:
    """Variance of Laplacian - a cheap blur/sharpness proxy.

    Reported per-sequence in ``anti-uav stats`` so a blurry IR sequence is
    identifiable before it quietly drags a run's mAP down.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def tile_coverage_summary(tiles: Sequence[Tile], width: int, height: int) -> dict[str, float]:
    """Area-weighted statistics for a tile plan; feeds the tiling cost estimate."""
    total = max(width * height, 1)
    covered = sum(t.width * t.height for t in tiles)
    return {
        "tile_count": float(len(tiles)),
        "area_expansion": covered / total,
        "mean_tile_px": float(np.mean([t.width * t.height for t in tiles])) if tiles else 0.0,
    }


def estimate_tile_cost(width: int, height: int, tile_size: int, overlap: float) -> int:
    """Number of tiles per frame - used to warn about the tiling time blow-up."""
    return len(tile_grid(width, height, tile_size=tile_size, overlap=overlap))


def ideal_tile_size(width: int, height: int, target_px: int = 32) -> int:
    """Pick a tile size that scales a ``target_px`` target up to ~32 px.

    MM-UAV's 12 px targets in a 640-wide frame want a ~3x magnification, i.e.
    640/3 ~ 213; we floor at 512 because below that the receptive field of the
    backbone's early layers stops covering a drone's rotor span.
    """
    if target_px <= 0:
        return min(width, height)
    # A tile of size S magnifies a `target_px` target by roughly
    # S / min(width, height) * ... - practically, we want the target to occupy
    # ~32 px, so scale by 32 / target_px and shrink the tile accordingly.
    factor = 32.0 / max(float(target_px), 1.0)
    candidate = round(min(width, height) / max(factor, 1.0))
    return max(512, min(candidate, 1280))
