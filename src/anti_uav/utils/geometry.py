"""Bounding-box and planar-geometry primitives.

Conventions used everywhere in this repo:

* ``xyxy``  - ``(x1, y1, x2, y2)`` in pixels, ``x2 >= x1``.
* ``xywh``  - ``(x, y, w, h)`` with ``(x, y)`` at the top-left corner.
* ``cxcywh``- normalised centre form, which is what a YOLO ``.txt`` label stores.
* Image space has ``+y`` pointing down. Ground/plan space has ``+y`` pointing
  north, so a yaw of 0 faces north and increases clockwise. The conversion
  happens only at the edge, in :func:`pixel_to_ground` / :func:`ground_to_pixel`.

Keeping one convention end to end removes an entire class of sign bugs that
otherwise only show up as a tracker that drifts left instead of right.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

BBox = tuple[float, float, float, float]
BBoxArray = np.ndarray  # shape (N, 4) in xyxy

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# conversions
# --------------------------------------------------------------------------- #


def xyxy_to_xywh(box: BBox) -> BBox:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(box: BBox) -> BBox:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_cxcywh(box: BBox, width: float, height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def cxcywh_to_xyxy(values: Sequence[float], width: float, height: float) -> BBox:
    """Normalised centre-x/cy/width/height -> pixel xyxy.

    Ultralytics emits the normalised form, so `cx`/`cy`/`w`/`h` are all
    fractions of the image. The halves below are already converted to pixels;
    multiplying the *result* by the image size again (which this used to do)
    double-scales every coordinate and puts the box thousands of pixels
    off-canvas.
    """
    cx, cy, w, h = values
    centre_x, centre_y = cx * width, cy * height
    half_w, half_h = w * width / 2.0, h * height / 2.0
    return (
        centre_x - half_w,
        centre_y - half_h,
        centre_x + half_w,
        centre_y + half_h,
    )


def clip_box(box: BBox, width: float, height: float) -> BBox:
    """Clamp to the image, keeping the box valid even if it falls fully outside."""
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    return (x1, y1, max(x2, x1), max(y2, y1))


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def boxes_area(boxes: BBoxArray) -> np.ndarray:
    wh = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None)
    return wh[:, 0] * wh[:, 1]


def box_centre(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_wh(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def box_aspect(box: BBox) -> float:
    w, h = box_wh(box)
    return w / h if h > _EPS else math.inf


def box_contains_point(box: BBox, x: float, y: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_iou(a: BBox, b: BBox) -> float:
    # Index both axes. A single [0] slices off the first axis only and returns a
    # length-1 array, which numpy < 2.4 silently converted to a scalar (with a
    # DeprecationWarning) and numpy >= 2.4 rejects outright under NEP 50.
    return float(iou_matrix(np.array([a], dtype=float), np.array([b], dtype=float))[0, 0])


# --------------------------------------------------------------------------- #
# vectorised IoU - the workhorse for NMS, association and MOT metrics
# --------------------------------------------------------------------------- #


def iou_matrix(boxes_a: BBoxArray, boxes_b: BBoxArray) -> np.ndarray:
    """Pairwise IoU for ``(N, 4)`` and ``(M, 4)`` arrays in xyxy."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    area_a = boxes_area(a)[:, None]
    area_b = boxes_area(b)[None, :]

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a + area_b - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


def giou_matrix(boxes_a: BBoxArray, boxes_b: BBoxArray) -> np.ndarray:
    """Generalised IoU. Continuous in the IoU's zero region, which keeps the
    assignment solver's cost surface smooth when boxes do not overlap at all -
    exactly the situation for a 12x5 px drone against the sky."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    area_a = boxes_area(a)[:, None]
    area_b = boxes_area(b)[None, :]

    # intersection
    lt_i = np.maximum(a[:, None, :2], b[None, :, :2])
    rb_i = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh_i = np.clip(rb_i - lt_i, 0.0, None)
    inter = wh_i[..., 0] * wh_i[..., 1]
    union = area_a + area_b - inter
    iou = np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)

    # smallest box enclosing both
    lt_e = np.minimum(a[:, None, :2], b[None, :, :2])
    rb_e = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh_e = np.clip(rb_e - lt_e, 0.0, None)
    enclosing_area = wh_e[..., 0] * wh_e[..., 1]

    return iou - (enclosing_area - union) / np.maximum(enclosing_area, _EPS)


def iou_distance_matrix(boxes_a: BBoxArray, boxes_b: BBoxArray) -> np.ndarray:
    """Cost for assignment: ``1 - IoU``. Lower is better."""
    return 1.0 - iou_matrix(boxes_a, boxes_b)


def center_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Euclidean distance between two ``(N, 2)`` / ``(M, 2)`` point sets."""
    a = np.asarray(points_a, dtype=float).reshape(-1, 2)
    b = np.asarray(points_b, dtype=float).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def normalized_box_scale(box: BBox, width: float, height: float) -> float:
    """``sqrt(w*h) / sqrt(W*H)``.

    The convention used by the COCO-style ``AP_S``/``AP_M``/``AP_L`` buckets and
    by the datasets' own documentation. With MM-UAV's 12x5 px targets this lands
    near 0.004, so the "small" bucket is not a formality.
    """
    w, h = box_wh(box)
    return math.sqrt(max(w * h, 0.0)) / math.sqrt(max(width * height, _EPS))


def tiny_scale_class(box: BBox, width: float, height: float) -> str:
    """``small`` / ``medium`` / ``large`` at the canonical COCO area cut-offs."""
    scale = normalized_box_scale(box, width, height)
    if scale < 0.33:
        return "small"
    if scale < 0.66:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# greedy NMS + assignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NmsResult:
    """Indices kept by NMS, in descending score order."""

    keep: np.ndarray
    suppressed: np.ndarray


def greedy_nms(
    boxes: BBoxArray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    *,
    class_ids: np.ndarray | None = None,
    max_keep: int | None = None,
) -> NmsResult:
    """Greedy non-maximum suppression.

    Sorted by score, kept greedily. When ``class_ids`` is supplied, suppression
    runs independently per class - which is what we want for drone-vs-bird, where
    a bird box overlapping a drone box must not be deleted. Passing
    ``class_ids=None`` gives class-agnostic (single-class) suppression.

    Returned indices refer to the **input** ordering.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    count = boxes.shape[0]
    if count == 0:
        return NmsResult(keep=np.empty(0, dtype=int), suppressed=np.empty(0, dtype=int))

    order = np.argsort(-scores, kind="stable")
    sorted_boxes = boxes[order]

    if class_ids is None:
        groups = [np.arange(count)]
    else:
        labels = np.asarray(class_ids).reshape(-1)[order]
        groups = [
            np.flatnonzero(labels == value).reshape(-1) for value in np.unique(labels)
        ]

    keep_mask = np.zeros(count, dtype=bool)
    for group in groups:
        if group.size == 0:
            continue
        group_boxes = sorted_boxes[group]
        ious = iou_matrix(group_boxes, group_boxes)
        suppressed = np.zeros(group.size, dtype=bool)
        for i in range(group.size):
            if suppressed[i]:
                continue
            keep_mask[group[i]] = True
            later = np.flatnonzero(~suppressed[i + 1 :]) + i + 1
            if later.size:
                suppressed[later[ious[i, later] > iou_threshold]] = True

    keep = order[keep_mask]
    if max_keep is not None and keep.size > max_keep:
        keep = keep[:max_keep]
    suppressed_idx = np.setdiff1d(order, keep)
    return NmsResult(keep=keep, suppressed=suppressed_idx)


def linear_assignment(
    cost: np.ndarray,
    *,
    max_cost: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Solve a rectangular linear-sum-assignment.

    Returns ``[(row, col, cost), ...]`` for pairs with ``cost <= max_cost``.
    Falls back to greedy when ``lap``/``scipy`` is unavailable so that unit
    tests and CPU-only installs still exercise the association path.
    """
    matrix = np.asarray(cost, dtype=float)
    if matrix.size == 0:
        return []

    try:
        from lap import linear_sum_assignment as _lsa  # type: ignore[import-not-found]

        rows, cols = _lsa(matrix)
    except ImportError:  # pragma: no cover - depends on install extras
        # scipy is the fallback when lap is unavailable; it returns the same
        # (rows, cols) pair, so the alias is reused rather than redefined.
        from scipy.optimize import linear_sum_assignment as _lsa_scipy

        rows, cols = _lsa_scipy(matrix)

    pairs = [
        (int(r), int(c), float(matrix[r, c]))
        for r, c in zip(rows, cols, strict=False)
        if matrix[r, c] <= max_cost
    ]
    pairs.sort(key=lambda item: item[2])
    return pairs


# --------------------------------------------------------------------------- #
# polygons and planar geometry
# --------------------------------------------------------------------------- #


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area. ``points`` is ``(N, 2)``."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    area = polygon_area(p)
    if area < _EPS or p.shape[0] < 3:
        return p.mean(axis=0) if p.size else np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < _EPS:
        return p.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return np.array([cx, cy], dtype=float)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray casting. ``polygon`` is ``(N, 2)`` closed or open."""
    px, py = point
    p = np.asarray(polygon, dtype=float)
    n = p.shape[0]
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > py) != (yj > py):
            t = (py - yi) / ((yj - yi) + _EPS)
            if px < xi + t * (xj - xi):
                inside = not inside
        j = i
    return inside


def polygon_intersection_ratio(subject: np.ndarray, container: np.ndarray) -> float:
    """Fraction of ``subject``'s area that falls inside ``container``.

    Sampled, not analytic: the container polygons in this project are irregular
    camera-overlap zones and geofences, so a grid sample at ~2 px is accurate
    enough and avoids a shapely dependency.
    """
    subj = np.asarray(subject, dtype=float)
    if subj.shape[0] < 3:
        return 0.0
    if _bbox_overlap(subj, np.asarray(container, dtype=float)) < _EPS:
        return 0.0

    min_x, min_y = subj.min(axis=0)
    max_x, max_y = subj.max(axis=0)
    step = max((max_x - min_x), (max_y - min_y)) / 64.0
    step = max(step, 1.0)

    xs = np.arange(min_x, max_x, step)
    ys = np.arange(min_y, max_y, step)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    in_subj = np.array([point_in_polygon((float(x), float(y)), subj) for x, y in grid], dtype=bool)
    if not in_subj.any():
        return 0.0
    in_cont = np.array([point_in_polygon((float(x), float(y)), container) for x, y in grid], dtype=bool)
    return float((in_subj & in_cont).sum()) / float(in_subj.sum())


def _bbox_overlap(a: np.ndarray, b: np.ndarray) -> float:
    min_a, max_a = a.min(axis=0), a.max(axis=0)
    min_b, max_b = b.min(axis=0), b.max(axis=0)
    overlap = np.clip(np.minimum(max_a, max_b) - np.maximum(min_a, min_b), 0.0, None)
    return float(overlap.prod())


def segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection of segment p1p2 with segment p3p4, or ``None``."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _EPS:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def normalize_angle(radians: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    return float((radians + math.pi) % (2.0 * math.pi) - math.pi)


def bearing_to(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """Compass bearing in radians, 0 = north (+y), increasing clockwise."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return normalize_angle(math.atan2(dx, dy))


def unwrap_bearing(previous: float, current: float) -> float:
    """Resolve a ±2π ambiguity in a PTZ pan update."""
    delta = normalize_angle(current - previous)
    return previous + delta


# --------------------------------------------------------------------------- #
# image <-> ground projection (pinhole)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """Pinhole intrinsics + extrinsics for one camera.

    ``R_wc`` maps camera-frame vectors into the world frame: ``X_w = R_wc @ X_c``.
    ``t_wc`` is the camera centre expressed in world coordinates.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    r_wc: np.ndarray  # (3, 3)
    t_wc: np.ndarray  # (3,)
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_yaw_pitch_roll(
        cls,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        position: tuple[float, float, float],
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float = 0.0,
        distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> PinholeCamera:
        """Build from an intuitive pose. ``yaw`` is clockwise from north."""
        r_wc = euler_to_rotation(yaw_deg, pitch_deg, roll_deg)
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            r_wc=r_wc,
            t_wc=np.asarray(position, dtype=float),
            distortion=distortion,
        )

    @property
    def centre(self) -> np.ndarray:
        return self.t_wc

    @property
    def optical_axis_world(self) -> np.ndarray:
        """Unit vector the camera looks along, in world coordinates."""
        return self.r_wc @ np.array([0.0, 0.0, 1.0])

    @property
    def hfov_rad(self) -> float:
        return float(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_rad(self) -> float:
        return float(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def pixel_to_ray(self, x: float, y: float) -> np.ndarray:
        """Undistorted pixel -> unit ray in the **camera** frame.

        (u, v, 1) unprojects through K^-1; the Brown-Conrady distortion is
        applied by iteration because the inverse has no closed form.
        """
        xd = (x - self.cx) / self.fx
        yd = (y - self.cy) / self.fy
        xd, yd = _undistort(xd, yd, self.distortion)
        ray = np.array([xd, yd, 1.0], dtype=float)
        return ray / np.linalg.norm(ray)

    def pixel_to_ground(
        self, x: float, y: float, plane_z: float
    ) -> tuple[float, float] | None:
        """Project a pixel onto the horizontal plane at world height ``plane_z``.

        Returns ``None`` when the ray is parallel to, or points away from, the
        plane - which happens for every pixel above the horizon. The camera
        coordination module treats that as "unreachable ground" and uses it for
        the geofence test.
        """
        ray_world = self.r_wc @ self.pixel_to_ray(x, y)
        if abs(ray_world[2]) < 1e-6:
            return None
        t = (plane_z - self.t_wc[2]) / ray_world[2]
        if t <= 0:
            return None
        return (float(self.t_wc[0] + t * ray_world[0]), float(self.t_wc[1] + t * ray_world[1]))

    def ground_to_pixel(self, point: tuple[float, float, float]) -> tuple[float, float] | None:
        """World point -> pixel, or ``None`` when behind the camera."""
        d = np.asarray(point, dtype=float) - self.t_wc
        cam = self.r_wc.T @ d
        if cam[2] <= 1e-6:
            return None
        x = self.fx * (cam[0] / cam[2]) + self.cx
        y = self.fy * (cam[1] / cam[2]) + self.cy
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return (float(x), float(y))

    def fov_polygon(
        self, plane_z: float, *, max_range_m: float | None = None, samples: int = 16
    ) -> np.ndarray:
        """The visible region of a horizontal plane, as a bounded polygon.

        Naively projecting the four image corners fails for any camera that is
        looking anywhere near the sky - which is all of them here, since a 6 m pole
        watching for drones points above the horizon. The upper corners' rays never
        meet the ground plane, so a corner-only implementation returns an empty
        polygon and the whole coverage layer reports that no camera sees anything.

        Instead the image border is sampled and every border point is projected.
        Rays that point upward intersect the plane "at infinity" - beyond the
        horizon - so they are clipped at ``max_range_m``, which both bounds the
        polygon and gives a sensible far edge. Points are returned in border order,
        so the ring is already correctly ordered.
        """
        if samples < 4:
            samples = 4
        range_limit = max_range_m if max_range_m is not None else 1.0e9

        border: list[tuple[float, float]] = []
        # Top edge, left->right; right edge, top->bottom; bottom, right->left;
        # left edge, bottom->top. Clockwise in image coords.
        for i in range(samples):
            border.append((self.width * i / (samples - 1), 0.0))
        for i in range(1, samples):
            border.append((float(self.width), self.height * i / (samples - 1)))
        for i in range(1, samples):
            border.append((self.width * (samples - 1 - i) / (samples - 1), float(self.height)))
        for i in range(1, samples - 1):
            border.append((0.0, self.height * (samples - 1 - i) / (samples - 1)))

        points: list[tuple[float, float]] = []
        for x, y in border:
            ray_world = self.r_wc @ self.pixel_to_ray(x, y)
            vertical = ray_world[2]
            horizontal = float(math.hypot(ray_world[0], ray_world[1]))

            if abs(vertical) < 1e-9:
                if horizontal < 1e-9:
                    continue
                t = range_limit
            elif vertical < 0:
                # Ray descends: it meets the plane at t, unless the plane is above
                # the camera (then it never does within range).
                t = (plane_z - self.t_wc[2]) / vertical
                if t <= 0 or t > range_limit:
                    if t <= 0:
                        continue
                    t = range_limit
            else:
                # Ray ascends: clip at the range limit along its horizontal heading.
                t = range_limit

            points.append(
                (
                    float(self.t_wc[0] + t * ray_world[0] / max(horizontal, 1e-9)),
                    float(self.t_wc[1] + t * ray_world[1] / max(horizontal, 1e-9)),
                )
            )

        if len(points) < 3:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)


def euler_to_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera-to-world rotation from an intuitive mount pose.

    Frames: world is ``+x`` east, ``+y`` north, ``+z`` up. The camera frame is
    ``+x`` right, ``+y`` down, ``+z`` along the optical axis - so its forward axis
    points along ``+z``, which means a zero rotation would point it at the zenith,
    not at the horizon.

    Hence the base rotation, and the signs:

    * ``R_x(-90 deg)`` maps camera-forward from ``(0, 0, 1)`` to ``(0, 1, 0)``,
      i.e. north, which is what ``yaw = 0`` should mean;
    * yaw is applied as ``R_z(-yaw)`` because we want it **clockwise from north**
      (90 deg = east), which is what :func:`bearing_to` returns;
    * pitch rotates about the camera's own right axis, and a **negative** pitch
      tilts the optical axis down, matching the convention in
      ``configs/coverage/coverage_map.yaml``.

    Combined: ``R_wc = R_z(-yaw) @ R_x(pitch - 90 deg) @ R_y(roll)``.

    Getting this wrong is silent and severe: the coverage polygons come out empty,
    every camera appears to be staring at the sky, and the entire coordination layer
    reports "no camera can see anything" without raising.
    """
    pitch_total = math.radians(pitch_deg - 90.0)
    roll = math.radians(roll_deg)
    yaw = math.radians(yaw_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch_total), math.sin(pitch_total)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])  # clockwise yaw
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])

    return rz @ rx @ ry


def _undistort(xd: float, yd: float, coeffs: tuple[float, float, float, float, float]) -> tuple[float, float]:
    """Iteratively remove Brown-Conrady radial+tangential distortion."""
    k1, k2, p1, p2, k3 = coeffs
    if not any(coeffs):
        return xd, yd
    x, y = xd, yd
    for _ in range(10):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def ray_box_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    box: BBox,
    min_distance: float = 0.0,
) -> float | None:
    """Distance along ``direction`` to the first hit on an axis-aligned box.

    Slab method. ``direction`` need not be normalised; the returned distance is
    in the same units as ``direction``. Returns ``None`` on a miss, which the
    risk estimator treats as "cannot bound this target in this camera".
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    b = np.asarray(box, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (b[:2] - o[:2]) * inv[:2]
        t2 = (b[2:] - o[:2]) * inv[:2]
    t_low = np.nanmin(np.stack([t1, t2]), axis=0)
    t_high = np.nanmax(np.stack([t1, t2]), axis=0)
    t_low = np.nan_to_num(t_low, nan=-np.inf, posinf=np.inf)
    t_high = np.nan_to_num(t_high, nan=-np.inf, posinf=-np.inf)

    t_near = float(np.max(t_low))
    t_far = float(np.min(t_high))
    if t_near > t_far or t_far < 0:
        return None
    hit = max(t_near, 0.0)
    if hit < min_distance:
        return min_distance if t_far >= min_distance else None
    return hit


def mean_point(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    pts = np.asarray(list(points), dtype=float)
    if pts.size == 0:
        return (0.0, 0.0)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
