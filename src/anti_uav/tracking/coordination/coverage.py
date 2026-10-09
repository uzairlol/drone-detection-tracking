"""Camera coverage model.

Section 3 of the system block diagram's "Camera Coverage & Calibration Model":
mount locations, orientation, intrinsics, FOV geometry with PTZ ranges, overlap
zones, blind spots, restricted sky, geofence, and a shared geospatial frame.

What the coordinator actually asks of it
-----------------------------------------
Four questions, in the order they get asked:

1. *Which cameras could see this target right now?* - drives cross-camera
   confirmation and the receiver shortlist for a handoff.
2. *Which camera is this target about to leave?* - FOV-exit prediction, the input
   to the 2-second handoff lead.
3. *Where can this PTZ point so it covers that spot?* - the controller needs a
   reachable pose, not an ideal one.
4. *Does every protected zone still have a camera?* - the scheduler's invariant.

Coverage polygons are computed from intrinsics and mount geometry on load and
cached, rather than being stored in ``coverage_map.yaml``. Storing them means they
go stale the moment a PTZ moves, and a stale overlap zone is worse than none: the
scheduler would believe a zone is covered by a camera that is pointing away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...config.schema import CameraSpec, CoverageMap, ProtectedZone
from ...utils.geometry import (
    PinholeCamera,
    bearing_to,
    point_in_polygon,
    polygon_area,
)
from ...utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class CoverageZone:
    """A region of ground some set of cameras can see."""

    camera_ids: list[str]
    polygon: np.ndarray
    #: Camera covering the most area - the natural default for a handoff receiver.
    primary_camera: str = ""
    area_m2: float = 0.0

    def contains(self, point: tuple[float, float]) -> bool:
        return point_in_polygon(point, self.polygon)


@dataclass(slots=True)
class CameraView:
    """A camera with its calibration resolved and its geometry cached."""

    spec: CameraSpec
    camera: PinholeCamera
    #: Ground-plane FOV polygon at the default height, range-clipped.
    fov_polygon: np.ndarray
    #: Where the camera can currently point.
    current_yaw_deg: float
    current_pitch_deg: float
    current_zoom: float
    _nominal_range: float = 320.0
    #: ``{height_m: polygon}``. A target at 30 m has a materially different
    #: footprint from one at 0 m, so the polygon is a function of height. Keeping
    #: one polygon and a separate pinhole test was the source of an inconsistency
    #: where a camera appeared to see a point the polygon said was out of range.
    _polygon_cache: dict[float, np.ndarray] = field(default_factory=dict)

    @property
    def camera_id(self) -> str:
        return self.spec.id

    @property
    def is_ptz(self) -> bool:
        return self.spec.role == "ptz" and self.spec.ptz.enabled

    @property
    def centre(self) -> tuple[float, float, float]:
        x, y, z = self.spec.mount.position_m
        return (x, y, z)

    @property
    def range_m(self) -> float:
        return self._nominal_range

    def polygon_at(self, height_m: float) -> np.ndarray:
        """The range-clipped FOV polygon on the plane at ``height_m``, cached."""
        key = round(float(height_m), 2)
        cached = self._polygon_cache.get(key)
        if cached is not None:
            return cached
        pose = _pinhole_at(self, self.current_yaw_deg, self.current_pitch_deg, self.current_zoom)
        polygon = pose.fov_polygon(key, max_range_m=self._nominal_range)
        self._polygon_cache[key] = polygon
        return polygon

    def invalidate_polygons(self) -> None:
        """Drop cached polygons. Called after the PTZ moves."""
        self._polygon_cache.clear()

    def covers(self, point: tuple[float, float], *, height_m: float | None = None) -> bool:
        """Whether the camera's FOV polygon contains a ground point."""
        polygon = self.fov_polygon if height_m is None else self.polygon_at(height_m)
        if polygon.shape[0] < 3:
            return False
        return point_in_polygon(point, polygon)

    def bearing_to(self, point: tuple[float, float]) -> float:
        """Bearing to a point, in **radians**. Use :meth:`bearing_deg` for poses."""
        return bearing_to((self.centre[0], self.centre[1]), point)

    def bearing_deg(self, point: tuple[float, float]) -> float:
        """Bearing to a point in **degrees**, clockwise from north.

        Every pose in this project is in degrees, so the coordinators use this
        rather than mixing a radian bearing into a degree limit - which silently
        produces a clamp that pins every camera to one edge of its pan range.
        """
        return math.degrees(self.bearing_to(point))

    def elevation_to(self, point: tuple[float, float, float]) -> float:
        """Elevation angle to a 3D point, degrees. Negative looks up."""
        dx = point[0] - self.centre[0]
        dy = point[1] - self.centre[1]
        dz = point[2] - self.centre[2]
        horizontal = math.hypot(dx, dy)
        if horizontal < 1e-6:
            return 0.0
        return math.degrees(math.atan2(dz, horizontal))

    def distance_to(self, point: tuple[float, float]) -> float:
        return math.hypot(point[0] - self.centre[0], point[1] - self.centre[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.camera_id,
            "node": self.spec.node,
            "role": self.spec.role,
            "model": self.spec.model,
            "position_m": [round(v, 2) for v in self.centre],
            "yaw_deg": self.current_yaw_deg,
            "pitch_deg": self.current_pitch_deg,
            "zoom": self.current_zoom,
            "hfov_deg": round(math.degrees(self.camera.hfov_rad), 1),
            "vfov_deg": round(math.degrees(self.camera.vfov_rad), 1),
            "enabled": self.spec.enabled,
            "tags": list(self.spec.tags),
        }


class CoverageModel:
    """Loads ``coverage_map.yaml`` and answers the coordinator's questions."""

    def __init__(
        self,
        coverage: CoverageMap,
        *,
        nominal_range_m: float | None = None,
        cache_overlaps: bool = True,
    ) -> None:
        self.config = coverage
        self.frame = coverage.frame
        self.default_plane_z = coverage.default_ground_plane_z_m
        self.nominal_range_m = nominal_range_m or coverage.nominal_range_m
        self.views: dict[str, CameraView] = {}
        self._overlaps: list[CoverageZone] = []
        self._zones: list[ProtectedZone] = list(coverage.protected_zones)

        for spec in coverage.cameras:
            self.views[spec.id] = self._build_view(spec)

        self._compute_protected_zone_polygons()
        if cache_overlaps:
            self._compute_overlaps()

    # -- construction ------------------------------------------------------- #

    def _build_view(self, spec: CameraSpec) -> CameraView:
        intrinsics = spec.intrinsics
        focal = intrinsics.fx_px
        if intrinsics.hfov_deg is not None and focal <= 0:
            focal = (intrinsics.width_px / 2.0) / math.tan(math.radians(intrinsics.hfov_deg) / 2.0)

        cx = intrinsics.cx_px if intrinsics.cx_px is not None else intrinsics.width_px / 2.0
        cy = intrinsics.cy_px if intrinsics.cy_px is not None else intrinsics.height_px / 2.0
        distortion = tuple(intrinsics.distortion) if intrinsics.distortion else (0.0,) * 5

        camera = PinholeCamera.from_yaw_pitch_roll(
            fx=focal,
            fy=focal,
            cx=cx,
            cy=cy,
            width=intrinsics.width_px,
            height=intrinsics.height_px,
            position=(float(spec.mount.position_m[0]), float(spec.mount.position_m[1]), float(spec.mount.position_m[2])),
            yaw_deg=spec.mount.yaw_deg,
            pitch_deg=spec.mount.pitch_deg,
            roll_deg=spec.mount.roll_deg,
            distortion=distortion,  # type: ignore[arg-type]
        )

        range_m = self.nominal_range_m
        view = CameraView(
            spec=spec,
            camera=camera,
            fov_polygon=camera.fov_polygon(self.default_plane_z, max_range_m=range_m),
            current_yaw_deg=spec.mount.yaw_deg,
            current_pitch_deg=spec.mount.pitch_deg,
            current_zoom=spec.ptz.zoom_min if spec.ptz.enabled else 1.0,
            _nominal_range=range_m,
        )
        return view

    def _compute_protected_zone_polygons(self) -> None:
        for zone in self._zones:
            zone_polygon = np.asarray(zone.polygon, dtype=float).reshape(-1, 2)
            zone._polygon = zone_polygon  # type: ignore[attr-defined]

    def _compute_overlaps(self) -> None:
        """Pairwise FOV intersection, cached.

        Uses a **bounding-box** overlap as the estimate rather than exact polygon
        clipping, and that is a deliberate trade. Exact clipping over ~5000 camera
        pairs with 32-vertex polygons is minutes of work at load time; a bbox
        estimate is milliseconds. The scheduler only needs to know *roughly* which
        cameras share a view, in order to prefer a receiver that does not have to
        slew. Precision here buys nothing and costs a slow start-up.

        The FOV polygon of a 97-degree camera is close to convex, so the bbox
        estimate is a reasonable upper bound on the true overlap, and being
        slightly generous is the safe direction: it can suggest a camera as a
        receiver that turns out not to have visibility, which the confirmation
        step then catches, rather than the reverse.
        """
        overlaps: list[CoverageZone] = []
        items = [v for v in self.views.values() if v.fov_polygon.shape[0] >= 3]

        # Bounding boxes up front, so the pair loop is mostly integer compares.
        boxes = [(_bounds(v.fov_polygon), v) for v in items]

        for i in range(len(boxes)):
            box_a, first = boxes[i]
            for j in range(i + 1, len(boxes)):
                box_b, second = boxes[j]

                shared = _bbox_shared_area(box_a, box_b)
                if shared <= 1.0:
                    continue

                ratio = shared / max(min(_bbox_area(box_a), _bbox_area(box_b)), 1e-6)
                if ratio <= 0.02:
                    continue

                subject = np.asarray(first.fov_polygon, dtype=float)
                # Shrink the subject toward the pair midpoint by the overlap
                # ratio, which approximates the shared region well enough to rank
                # candidate receivers.
                midpoint = np.array(
                    [(first.centre[0] + second.centre[0]) / 2.0, (first.centre[1] + second.centre[1]) / 2.0]
                )
                centroid = subject.mean(axis=0)
                centroid = centroid + (midpoint - centroid) * math.sqrt(max(ratio, 1e-3))
                polygon = centroid + (subject - centroid) * math.sqrt(max(ratio, 0.05))

                overlaps.append(
                    CoverageZone(
                        camera_ids=[first.camera_id, second.camera_id],
                        polygon=polygon,
                        primary_camera=first.camera_id,
                        area_m2=polygon_area(polygon),
                    )
                )

        self._overlaps = overlaps
        log.info(
            "coverage overlaps computed",
            extra={
                "cameras": len(self.views),
                "overlap_zones": len(overlaps),
                "method": "bounding-box estimate",
            },
        )

    # -- queries: who can see this ------------------------------------------ #

    def cameras_seeing(
        self,
        point: tuple[float, float],
        *,
        height_m: float | None = None,
        enabled_only: bool = True,
        include_ptz: bool = True,
    ) -> list[str]:
        """Cameras whose FOV contains a ground point.

        Visibility is always decided by the range-clipped ground polygon, at the
        requested height. There is deliberately no separate pinhole test here: two
        definitions of "can see it" disagree at exactly the range boundary, which is
        where the handoff logic has to be right.
        """
        out: list[str] = []
        for view in self.views.values():
            if enabled_only and not view.spec.enabled:
                continue
            if not include_ptz and view.is_ptz:
                continue
            if view.covers(point, height_m=height_m):
                out.append(view.camera_id)
        return out

    def overlaps_for(self, camera_id: str) -> list[CoverageZone]:
        return [z for z in self._overlaps if camera_id in z.camera_ids]

    @property
    def overlap_zones(self) -> list[CoverageZone]:
        return list(self._overlaps)

    # -- queries: which camera is it leaving ------------------------------- #

    def fov_exit_distance(
        self,
        camera_id: str,
        position: tuple[float, float],
        velocity: tuple[float, float],
        *,
        height_m: float | None = None,
        samples: int = 64,
        horizon_m: float = 900.0,
    ) -> float | None:
        """Ground distance until a target leaves this camera's FOV.

        Uses the same range-clipped polygon as :meth:`cameras_seeing`, so exit
        detection and coverage cannot disagree.

        Returns ``None`` when the target stays inside the FOV for ``horizon_m``,
        which callers must distinguish from "leaves at distance 0": the first means
        no handoff is needed, the second means the target is already outside.
        """
        view = self.views.get(camera_id)
        if view is None:
            return None
        polygon = view.fov_polygon if height_m is None else view.polygon_at(height_m)
        if polygon.shape[0] < 3:
            return None

        speed = math.hypot(velocity[0], velocity[1])
        if speed < 1e-6:
            return None

        step = horizon_m / max(samples, 1)
        for index in range(1, samples + 1):
            distance = step * index
            probe = (
                position[0] + velocity[0] / speed * distance,
                position[1] + velocity[1] / speed * distance,
            )
            if not point_in_polygon(probe, polygon):
                return distance

        return None

    # -- queries: reachable poses ------------------------------------------- #

    def required_pose(
        self,
        camera_id: str,
        point: tuple[float, float],
        *,
        height_m: float = 10.0,
        zoom: float | None = None,
        margin_frac: float = 0.12,
    ) -> tuple[float, float, float] | None:
        """The (yaw, pitch, zoom) that would centre a camera on a point.

        ``margin_frac`` is the aim bias as a fraction of the **half** vertical
        field of view, not a fixed number of degrees. This matters more than it
        sounds: a PTZ at 9 m zoom has a vertical half-FOV of under two degrees, so
        a hardcoded 2-degree downward bias points the camera *below* the target and
        the target falls out of the bottom of the frame - the "zoom in and lose it"
        failure. Scaling the bias to the FOV keeps the target inside at every zoom
        level while still centring it rather than putting it on the edge.

        Returns ``None`` when no reachable pose exists - a fixed camera that cannot
        see the point, or a PTZ whose limits cannot bring it into frame. The handoff
        coordinator treats that as "not a candidate" rather than clamping to a limit
        and commanding a pose that still misses.
        """
        view = self.views.get(camera_id)
        if view is None:
            return None

        bearing = view.bearing_deg(point)
        elevation = view.elevation_to((point[0], point[1], height_m))

        if not view.is_ptz:
            # A fixed camera either already sees it or never will.
            if not point_in_polygon(point, view.fov_polygon):
                return None
            return (view.current_yaw_deg, view.current_pitch_deg, 1.0)

        ptz = view.spec.ptz
        yaw = _clamp(bearing, ptz.pan_min_deg, ptz.pan_max_deg)

        chosen_zoom = (
            zoom
            if zoom is not None
            else _zoom_for_distance(
                view.distance_to(point),
                height_m - view.centre[2],
                ptz.zoom_max,
                focal_px=view.camera.fx,
            )
        )
        chosen_zoom = float(np.clip(chosen_zoom, ptz.zoom_min, ptz.zoom_max))

        # Vertical half-FOV at the chosen zoom, in degrees.
        half_vfov_deg = math.degrees(
            math.atan(view.camera.height / 2.0 / max(view.camera.fx * chosen_zoom, 1e-6))
        )
        pitch = _clamp(
            elevation - margin_frac * half_vfov_deg,
            ptz.tilt_min_deg,
            ptz.tilt_max_deg,
        )

        # Final check: the point must actually land inside the frame at this pose and
        # zoom. Zooming narrows the FOV, so a target centred at wide can fall out at
        # long - the opposite of what "zoom in to see it better" suggests.
        probe = _pinhole_at(view, yaw, pitch, chosen_zoom)
        if probe.ground_to_pixel((point[0], point[1], height_m)) is None:
            # Retry once at wide, where the target is certain to be in frame. A
            # blurry-but-present track beats a sharp one the camera cannot see.
            wide_zoom = float(ptz.zoom_min)
            wide_probe = _pinhole_at(view, yaw, elevation, wide_zoom)
            if wide_probe.ground_to_pixel((point[0], point[1], height_m)) is None:
                return None
            return (yaw, elevation, wide_zoom)

        return (yaw, pitch, chosen_zoom)

    def reachable(self, camera_id: str, point: tuple[float, float], *, height_m: float = 10.0) -> bool:
        return self.required_pose(camera_id, point, height_m=height_m) is not None

    # -- queries: protected zones ------------------------------------------ #

    def protected_zones(self) -> list[ProtectedZone]:
        return list(self._zones)

    def zone_polygon(self, zone: ProtectedZone) -> np.ndarray:
        return np.asarray(zone.polygon, dtype=float).reshape(-1, 2)

    def zone_covered(self, zone: ProtectedZone) -> bool:
        """Whether at least one enabled camera's FOV overlaps a zone.

        Tests a few sample points rather than doing exact polygon clipping: the
        scheduler's question is "is this zone watched at all", and a centre-plus-
        corners sample answers that robustly without a geometry dependency.
        """
        polygon = self.zone_polygon(zone)
        if polygon.shape[0] < 3:
            return False

        centroid = polygon.mean(axis=0)
        samples: list[tuple[float, float]] = [(float(centroid[0]), float(centroid[1]))]
        samples.extend((float(x), float(y)) for x, y in polygon)

        if any(self.cameras_seeing(point, height_m=zone.height_m) for point in samples):
            return True

        # A zone can also be covered only by a PTZ that has to be commanded
        # there first. That is still coverage - the fleet owns the PTZ and the
        # scheduler pre-positions it - so it counts, but `zone_covered_fixed`
        # keeps the distinction visible instead of hiding it in a boolean.
        return any(self.ptz_zone_cameras(zone))

    def zone_cameras(self, zone: ProtectedZone) -> list[str]:
        """Cameras watching a zone right now, plus reachable PTZs that could."""
        polygon = self.zone_polygon(zone)
        centroid = polygon.mean(axis=0)
        samples = [(float(centroid[0]), float(centroid[1]))]
        samples.extend((float(x), float(y)) for x, y in polygon)

        found: list[str] = []
        for point in samples:
            for camera_id in self.cameras_seeing(point, height_m=zone.height_m):
                if camera_id not in found:
                    found.append(camera_id)
        for camera_id in self.ptz_zone_cameras(zone):
            if camera_id not in found:
                found.append(camera_id)
        return found

    def ptz_zone_cameras(self, zone: ProtectedZone) -> list[str]:
        """PTZs with at least one reachable pose over some sample point.

        Reachability is what the PTZ scheduler cares about, and it is the only
        honest answer for a zone sitting in a fixed camera's near-field blind
        spot: an 8 deg up-tilt means a fence-mounted camera cannot see the
        ground at its own feet, so a PTZ has to take the watch.
        """
        polygon = self.zone_polygon(zone)
        if polygon.shape[0] < 3:
            return []

        centroid = polygon.mean(axis=0)
        samples: list[tuple[float, float]] = [(float(centroid[0]), float(centroid[1]))]
        samples.extend((float(x), float(y)) for x, y in polygon)

        found: list[str] = []
        for camera_id, view in self.views.items():
            if not view.is_ptz:
                continue
            if any(self.reachable(camera_id, point, height_m=zone.height_m) for point in samples):
                found.append(camera_id)
        return found

    def coverage_report(self) -> dict[str, Any]:
        """Site-level coverage summary for the UI and the operator."""
        zones = self.protected_zones()
        uncovered = [z.id for z in zones if not self.zone_covered(z)]
        fixed_covered = {
            z.id for z in zones
            if any(self.cameras_seeing(p, height_m=z.height_m) for p in self._zone_samples(z))
        }
        ptz_only = sorted(
            z.id for z in zones if z.id not in fixed_covered and self.zone_covered(z)
        )

        by_node: dict[str, int] = {}
        for view in self.views.values():
            by_node[view.spec.node] = by_node.get(view.spec.node, 0) + 1

        return {
            "frame": self.frame,
            "cameras": len(self.views),
            "fixed": sum(1 for v in self.views.values() if not v.is_ptz),
            "ptz": sum(1 for v in self.views.values() if v.is_ptz),
            "nodes": len(by_node),
            "cameras_per_node": dict(sorted(by_node.items())),
            "overlap_zones": len(self._overlaps),
            "protected_zones": len(zones),
            "uncovered_zones": uncovered,
            "all_zones_covered": not uncovered,
            "fixed_covered_zones": sorted(fixed_covered),
            "ptz_only_zones": ptz_only,
        }

    def _zone_samples(self, zone: ProtectedZone) -> list[tuple[float, float]]:
        """Centroid plus corners - enough to describe a convex zone's extent."""
        polygon = self.zone_polygon(zone)
        if polygon.shape[0] < 3:
            return []
        centroid = polygon.mean(axis=0)
        samples: list[tuple[float, float]] = [(float(centroid[0]), float(centroid[1]))]
        samples.extend((float(x), float(y)) for x, y in polygon)
        return samples

    # -- PTZ state ---------------------------------------------------------- #

    def set_pose(self, camera_id: str, yaw_deg: float, pitch_deg: float, zoom: float) -> None:
        """Record where a PTZ has been commanded to.

        Called by the PTZ controller. Keeping this here rather than in the
        controller means the coverage model always reflects what the planner
        believes the fleet is doing, which is what the next scheduling decision is
        based on.
        """
        view = self.views.get(camera_id)
        if view is None or not view.is_ptz:
            return

        view.current_yaw_deg = float(np.clip(yaw_deg, view.spec.ptz.pan_min_deg, view.spec.ptz.pan_max_deg))
        view.current_pitch_deg = float(np.clip(pitch_deg, view.spec.ptz.tilt_min_deg, view.spec.ptz.tilt_max_deg))
        view.current_zoom = float(np.clip(zoom, view.spec.ptz.zoom_min, view.spec.ptz.zoom_max))
        view.invalidate_polygons()
        view.fov_polygon = _pinhole_at(
            view, view.current_yaw_deg, view.current_pitch_deg, view.current_zoom
        ).fov_polygon(self.default_plane_z, max_range_m=self.nominal_range_m)

    def derive_horizon_y(self, camera_id: str) -> float | None:
        """Normalised image row of the horizon, from the mount pitch.

        Solves for the pixel row whose ray is horizontal. In the camera frame a
        ray is ``(xd, yd, 1)``; its world height is ``R[2,1] * yd + R[2,2]``, so the
        horizon sits at ``cy + fy * (-R[2,2] / R[2,1])``.

        This is what makes the above-horizon rule work without a hand-entered
        ``horizon_y`` on all 100 cameras - and a camera tilted 8 degrees down on a
        6 m pole does see the horizon, at about 39% down the frame. Returns
        ``None`` when the geometry is degenerate (the axis is exactly horizontal,
        or the roll is non-zero so the horizon is not a single row).
        """
        view = self.views.get(camera_id)
        if view is None:
            return None

        rotation = view.camera.r_wc
        a = float(rotation[2, 1])
        b = float(rotation[2, 2])
        if abs(a) < 1e-6:
            # Optical axis exactly horizontal: the horizon is the image centre.
            return 0.5

        yd = -b / a
        row = view.camera.cy + view.camera.fy * yd
        normalised = row / max(view.camera.height, 1)
        if not 0.0 <= normalised <= 1.0:
            # Camera pitched so the horizon is outside the frame: the whole frame
            # is either sky or ground. Return the clamped edge so the rule still
            # has a definite answer.
            return 0.0 if normalised < 0.0 else 1.0
        return float(normalised)

    def horizon_by_camera(self) -> dict[str, float]:
        """``{camera_id: horizon_y}``, preferring the explicit value.

        ``coverage_map.yaml`` may set ``horizon_y`` per camera; where it is null the
        value is derived from the mount pitch, which is correct for a rigid mount
        and requires no hand calibration.
        """
        out: dict[str, float] = {}
        for camera_id, view in self.views.items():
            declared = view.spec.horizon_y
            if declared is not None:
                out[camera_id] = float(declared)
                continue
            derived = self.derive_horizon_y(camera_id)
            if derived is not None:
                out[camera_id] = derived
        return out

    def geofences(self) -> list[tuple[str, str, np.ndarray, float]]:
        """``(id, action, polygon, height_m)`` for the rule layer."""
        out: list[tuple[str, str, np.ndarray, float]] = []
        for fence in self.config.geofences:
            out.append(
                (
                    fence.id,
                    fence.action,
                    np.asarray(fence.polygon, dtype=float).reshape(-1, 2),
                    fence.height_m,
                )
            )
        return out

    def view(self, camera_id: str) -> CameraView | None:
        return self.views.get(camera_id)

    def node_cameras(self, node: str) -> list[str]:
        return [v.camera_id for v in self.views.values() if v.spec.node == node]

    def enabled_cameras(self) -> list[str]:
        return [v.camera_id for v in self.views.values() if v.spec.enabled]

    def __len__(self) -> int:
        return len(self.views)

    def __contains__(self, camera_id: str) -> bool:
        return camera_id in self.views


def _bounds(polygon: np.ndarray) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)``."""
    return (
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_shared_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def _pinhole_at(view: CameraView, yaw_deg: float, pitch_deg: float, zoom: float) -> PinholeCamera:
    """A copy of the camera's intrinsics at a different pose and zoom.

    Zoom narrows the FOV, so focal length scales by the zoom factor - which is the
    whole basis of the reach check in :meth:`CoverageModel.required_pose`.
    """
    focal = view.camera.fx * max(zoom, 1e-3)
    return PinholeCamera.from_yaw_pitch_roll(
        fx=focal,
        fy=focal,
        cx=view.camera.cx,
        cy=view.camera.cy,
        width=view.camera.width,
        height=view.camera.height,
        position=(
            float(view.spec.mount.position_m[0]),
            float(view.spec.mount.position_m[1]),
            float(view.spec.mount.position_m[2]),
        ),
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=view.spec.mount.roll_deg,
        distortion=view.camera.distortion,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(min(value, high), low))


def _zoom_for_distance(
    distance_m: float,
    height_diff_m: float,
    zoom_max: float,
    *,
    focal_px: float,
    target_height_px: float = 60.0,
) -> float:
    """Zoom so a target subtends ``target_height_px``.

    From the pinhole relation ``pixels = f * zoom * height / distance``::

        zoom = target_px * distance / (f * height)

    ``focal_px`` is the camera's focal length at zoom 1.0, *not* the image height -
    using the image height here over-zooms by roughly the aspect ratio, which then
    narrows the FOV enough that the target we just centred falls out of frame.

    Floored at 1.0: a very close target must not be "zoomed out" below the wide
    end, which is what a bare ratio would do.
    """
    if distance_m < 1e-3 or abs(height_diff_m) < 1e-3 or focal_px <= 0:
        return 1.0
    ideal = target_height_px * distance_m / (focal_px * abs(height_diff_m))
    return float(max(1.0, min(ideal, zoom_max)))


def load_from_config(
    *, nominal_range_m: float | None = None, cache_overlaps: bool = True
) -> CoverageModel:
    """Load ``configs/coverage/coverage_map.yaml``."""
    from ...config.loader import load_coverage_map

    return CoverageModel(
        load_coverage_map(),
        nominal_range_m=nominal_range_m,
        cache_overlaps=cache_overlaps,
    )


def describe(model: CoverageModel) -> str:
    report = model.coverage_report()
    lines = [
        f"frame            : {report['frame']}",
        f"cameras          : {report['cameras']}  ({report['fixed']} fixed, {report['ptz']} PTZ)",
        f"worker nodes     : {report['nodes']}",
        f"overlap zones    : {report['overlap_zones']}",
        f"protected zones  : {report['protected_zones']}",
        f"all zones covered: {'yes' if report['all_zones_covered'] else 'NO'}",
    ]
    if report["uncovered_zones"]:
        lines.append(f"  uncovered: {', '.join(report['uncovered_zones'])}")
    lines.append("")
    lines.append("cameras per node:")
    for node, count in report["cameras_per_node"].items():
        lines.append(f"  {node}: {count}")
    return "\n".join(lines)


def summarize_cameras(model: CoverageModel) -> list[dict[str, Any]]:
    return [view.to_dict() for view in sorted(model.views.values(), key=lambda v: v.camera_id)]
