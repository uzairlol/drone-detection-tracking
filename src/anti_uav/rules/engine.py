"""The drone rule / threshold engine.

Section 4 of the system block diagram, transcribed from ``drone_rules.yaml``:
confidence, persistence, kinematics, spatial and cross-camera gates, plus the
alert-severity decisions.

What this layer is for
----------------------
The detector answers "is there a drone in this box". This layer answers "is this
a *drone* worth waking someone for", and it is the only place that question gets
asked. That separation is deliberate and it is why ``conf_threshold`` in
``configs/app.yaml`` is 0.25 while ``confidence.initiate`` here is 0.60: the
detector stays permissive so the tracker can hold a track through a dip, and every
decision that matters is made here.

The gates are **conjunctive by default**. A track that fails any gate is not
alerted, and the reason is reported. That matters for triage: "we saw something
that moved at 40 m/s" is a different operational problem from "we saw a bird",
and the operator needs to know which rule fired.

A note on the false-positive case
---------------------------------
Two of the four datasets contain no birds at all, so a model trained on those
alone has never been told what a bird looks like. This engine cannot fix that - no
threshold can - which is why ``anti-uav stats`` and the eval tables both refuse to
present precision from a bird-free source as if it meant something. The gate order
below puts *cheap, geometric* tests first so that a bird is rejected before
anything expensive runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config.schema import DroneRules
from ..utils.logging import get_logger

log = get_logger(__name__)


class Severity(StrEnum):
    """Alert severity, ordered by :attr:`RulesConfig.severity_order`."""

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
    NONE = "none"


class RuleOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"  # the input needed for this rule was unavailable


@dataclass(slots=True)
class RuleResult:
    """One gate's verdict."""

    name: str
    outcome: RuleOutcome
    detail: str = ""
    measured: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome is not RuleOutcome.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "measured": {k: round(v, 3) for k, v in self.measured.items()},
        }


@dataclass(slots=True)
class RuleEvaluation:
    """The full verdict for one track, with every gate's reasoning."""

    track_id: int
    global_id: int | None
    alerted: bool
    severity: Severity
    reasons: list[str] = field(default_factory=list)
    results: list[RuleResult] = field(default_factory=list)
    measured: dict[str, float] = field(default_factory=dict)

    @property
    def first_failure(self) -> str | None:
        for result in self.results:
            if result.outcome is RuleOutcome.FAIL:
                return result.name
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "global_id": self.global_id,
            "alerted": self.alerted,
            "severity": self.severity.value,
            "first_failure": self.first_failure,
            "reasons": list(self.reasons),
            "results": [r.to_dict() for r in self.results],
            "measured": {k: round(v, 3) for k, v in self.measured.items()},
        }


@dataclass(slots=True)
class RuleConfig:
    """A live rule set, derived from ``drone_rules.yaml`` plus runtime state.

    Mutable by design: hysteresis means ``initiate`` and ``maintain`` differ
    depending on the track's current state, and a long-running engine should not
    reload the YAML to express that. A threshold change still goes through
    ``configs/rules/drone_rules.yaml`` and a redeploy - the runtime fields are
    counters, not policy.
    """

    rules: DroneRules
    #: Per-camera horizon lines in normalised image coords, from coverage_map.
    horizon_by_camera: dict[str, float] = field(default_factory=dict)
    #: Geofence polygons as ``{id: (action, np.ndarray, height_m)}``.
    geofences: dict[str, tuple[str, np.ndarray, float]] = field(default_factory=dict)
    #: Ground plane height, metres, for the speed estimate.
    ground_plane_z_m: float | None = None
    #: Cross-camera evidence: ``{global_id: [(camera_id, node, timestamp, xy), ...]}``.
    sightings: dict[int, list[tuple[str, str, float, tuple[float, float]]]] = field(
        default_factory=dict
    )
    evaluations: int = 0
    alerts_raised: int = 0

    @classmethod
    def from_rules(cls, rules: DroneRules) -> RuleConfig:
        return cls(rules=rules, ground_plane_z_m=rules.spatial.ground_plane_z_m)

    def record_sighting(
        self,
        global_id: int,
        camera_id: str,
        node: str,
        timestamp_s: float,
        ground_xy: tuple[float, float],
    ) -> None:
        """Note that a target was seen. Feeds the cross-camera confirmation rule."""
        self.sightings.setdefault(global_id, []).append(
            (camera_id, node, timestamp_s, ground_xy)
        )

    def trim_sightings(self, *, older_than_s: float = 60.0) -> None:
        """Drop stale sightings so the cross-camera window cannot grow unbounded."""
        for key in list(self.sightings):
            kept = [s for s in self.sightings[key] if s[2] >= older_than_s - 60.0]
            if kept:
                self.sightings[key] = kept
            else:
                del self.sightings[key]

    def add_geofence(
        self, geofence_id: str, action: str, polygon: Sequence[float], height_m: float
    ) -> None:
        self.geofences[geofence_id] = (
            action,
            np.asarray(polygon, dtype=float).reshape(-1, 2),
            height_m,
        )


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #


class RuleEngine:
    """Evaluates the rule layer against a track."""

    def __init__(self, config: RuleConfig) -> None:
        self.config = config

    # -- main entry point --------------------------------------------------- #

    def evaluate(
        self,
        track: Any,
        *,
        timestamp_s: float,
        already_alerted: bool = False,
    ) -> RuleEvaluation:
        """Run every gate against one track.

        ``track`` is duck-typed on purpose: it is a
        :class:`~anti_uav.tracking.types.Track` here, but the same engine has to
        accept a simplified object in the UI's preview and in tests without
        constructing a full track.
        """
        rules = self.config.rules
        self.config.evaluations += 1

        results: list[RuleResult] = []
        reasons: list[str] = []
        measured: dict[str, float] = {}

        # Gate 1: confidence, with hysteresis. ----------------------------- #
        initiate = rules.confidence.initiate
        maintain = rules.confidence.maintain
        already_alerted = already_alerted or bool(getattr(track, "alert", ""))
        confidence = float(getattr(track, "confidence_ema", 0.0) or 0.0)
        threshold = maintain if already_alerted else initiate
        confidence_result = self._confidence(confidence, threshold, already_alerted)
        results.append(confidence_result)
        measured["confidence"] = confidence
        measured["confidence_threshold"] = threshold

        # Gate 2: persistence. ---------------------------------------------- #
        persistence = self._persistence(track)
        results.append(persistence)
        measured["hits"] = float(getattr(track, "hits", 0))
        measured["misses"] = float(getattr(track, "misses", 0))

        # Gate 3: kinematics. Skipped when no ground plane is configured, and
        # says so - a silently skipped gate looks identical to a passing one. -- #
        kinematics = self._kinematics(track, timestamp_s)
        results.append(kinematics)
        if kinematics.outcome is RuleOutcome.SKIPPED:
            measured.setdefault("speed_m_s", 0.0)

        # Gate 4: spatial. ---------------------------------------------------- #
        spatial = self._spatial(track)
        results.append(spatial)

        # Gate 5: cross-camera confirmation. ---------------------------------- #
        cross = self._cross_camera(track, timestamp_s)
        results.append(cross)

        for result in results:
            if result.outcome is RuleOutcome.FAIL:
                reasons.append(f"{result.name}: {result.detail}")
            elif result.outcome is RuleOutcome.SKIPPED and result.detail:
                reasons.append(f"{result.name} skipped - {result.detail}")

        failed = [r for r in results if r.outcome is RuleOutcome.FAIL]
        alerted = not failed and getattr(track, "class_id", 0) == 0

        severity = Severity.NONE
        if alerted:
            severity = self._severity(track, measured)
            self.config.alerts_raised += 1
            if hasattr(track, "alert"):
                track.alert = severity.value

        return RuleEvaluation(
            track_id=int(getattr(track, "track_id", -1)),
            global_id=getattr(track, "global_id", None),
            alerted=alerted,
            severity=severity,
            reasons=reasons,
            results=results,
            measured=measured,
        )

    # -- individual gates --------------------------------------------------- #

    def _confidence(
        self, confidence: float, threshold: float, maintain_mode: bool
    ) -> RuleResult:
        label = "maintain" if maintain_mode else "initiate"
        if confidence >= threshold:
            return RuleResult("confidence", RuleOutcome.PASS, measured={"confidence": confidence})
        return RuleResult(
            "confidence",
            RuleOutcome.FAIL,
            detail=(
                f"{confidence:.3f} < {threshold:.3f} ({label} threshold)"
            ),
            measured={"confidence": confidence, "threshold": threshold},
        )

    def _persistence(self, track: Any) -> RuleResult:
        rules = self.config.rules.persistence
        hits = int(getattr(track, "hits", 0))
        duration = float(getattr(track, "duration_s", lambda: 0.0)())
        misses = int(getattr(track, "misses", 0))

        measured = {"hits": hits, "duration_s": duration, "misses": misses}

        if hits < rules.min_hits:
            return RuleResult(
                "persistence",
                RuleOutcome.FAIL,
                detail=f"{hits} hits < {rules.min_hits}",
                measured=measured,
            )
        if duration < rules.min_duration_s:
            return RuleResult(
                "persistence",
                RuleOutcome.FAIL,
                detail=(
                    f"tracked for {duration:.2f} s < {rules.min_duration_s:.2f} s "
                    f"(hit count alone is not enough - a degraded 1 fps stream would "
                    f"accumulate hits far too slowly)"
                ),
                measured=measured,
            )
        if misses > rules.max_gap_frames:
            return RuleResult(
                "persistence",
                RuleOutcome.FAIL,
                detail=f"gap of {misses} frames exceeds max_gap_frames={rules.max_gap_frames}",
                measured=measured,
            )
        return RuleResult("persistence", RuleOutcome.PASS, measured=measured)

    def _kinematics(self, track: Any, timestamp_s: float) -> RuleResult:
        rules = self.config.rules.kinematics

        ground_xy = getattr(track, "ground_xy", None)
        velocity = getattr(track, "velocity_m_s", None)

        if ground_xy is None or self.config.ground_plane_z_m is None:
            return RuleResult(
                "kinematics",
                RuleOutcome.SKIPPED,
                detail=(
                    "no ground position or ground_plane_z_m is configured, so speed "
                    "cannot be estimated from pixels. Set spatial.ground_plane_z_m and "
                    "calibrate the camera."
                ),
            )

        speed = math.hypot(velocity[0], velocity[1]) if velocity else 0.0
        measured: dict[str, float] = {"speed_m_s": speed}

        if speed < rules.min_speed_m_s:
            return RuleResult(
                "kinematics",
                RuleOutcome.FAIL,
                detail=(
                    f"{speed:.2f} m/s is below the {rules.min_speed_m_s:.1f} m/s floor - "
                    f"a stationary airborne object is far more likely to be a bird or debris"
                ),
                measured=measured,
            )
        if speed > rules.max_speed_m_s:
            return RuleResult(
                "kinematics",
                RuleOutcome.FAIL,
                detail=(
                    f"{speed:.2f} m/s exceeds the {rules.max_speed_m_s:.1f} m/s ceiling - "
                    f"nothing in the threat catalogue flies this fast"
                ),
                measured=measured,
            )

        turn_rate = self._turn_rate(track)
        if turn_rate is not None:
            measured["turn_rate_deg_s"] = turn_rate
            if turn_rate > rules.max_turn_rate_deg_s:
                return RuleResult(
                    "kinematics",
                    RuleOutcome.FAIL,
                    detail=(
                        f"turn rate {turn_rate:.1f} deg/s exceeds "
                        f"{rules.max_turn_rate_deg_s:.0f} deg/s - birds yaw faster than "
                        f"the quadcopters in this threat set"
                    ),
                    measured=measured,
                )

        hover = self._hover(track, timestamp_s)
        if hover is not None:
            measured["hover_m"] = hover
            if hover < rules.hover_threshold_m:
                return RuleResult(
                    "kinematics",
                    RuleOutcome.FAIL,
                    detail=(
                        f"vertical motion of {hover:.2f} m over "
                        f"{rules.hover_duration_s:.1f} s is under "
                        f"{rules.hover_threshold_m:.2f} m - flags, kites and tethered "
                        f"objects all look like this"
                    ),
                    measured=measured,
                )

        return RuleResult("kinematics", RuleOutcome.PASS, measured=measured)

    def _spatial(self, track: Any) -> RuleResult:
        rules = self.config.rules.spatial
        camera_id = str(getattr(track, "camera_id", "") or "")
        box = getattr(track, "box", None)
        height_px = float(getattr(track, "height_px", 0.0) or 0.0)
        image_height = float(getattr(track, "image_height_px", 0) or 0)

        measured: dict[str, float] = {}
        if height_px:
            measured["height_px"] = height_px

        if rules.require_above_horizon and box is not None:
            horizon = self.config.horizon_by_camera.get(camera_id, rules.horizon_y)
            if horizon is not None and image_height > 0:
                horizon_px = horizon * image_height
                measured["horizon_px"] = horizon_px
                centre_y = (box[1] + box[3]) / 2.0
                measured["centre_y"] = centre_y
                if centre_y >= horizon_px:
                    return RuleResult(
                        "spatial",
                        RuleOutcome.FAIL,
                        detail=(
                            f"target centre at y={centre_y:.0f} px is at or below the "
                            f"horizon ({horizon_px:.0f} px for {camera_id or 'camera'}) - "
                            f"anything there is ground clutter"
                        ),
                        measured=measured,
                    )
            elif horizon is None:
                return RuleResult(
                    "spatial",
                    RuleOutcome.SKIPPED,
                    detail=(
                        "no horizon_y for this camera. Add it to coverage_map.yaml, or "
                        "it is derived from the mount pitch when the camera is calibrated."
                    ),
                    measured=measured,
                )

        geofence = self._geofence(track)
        if geofence is not None:
            action, geofence_id = geofence
            measured["geofence"] = 0.0
            if action == "suppress" and rules.geofence_breach_suppresses:
                return RuleResult(
                    "spatial",
                    RuleOutcome.FAIL,
                    detail=f"inside suppress-zone {geofence_id!r}",
                    measured=measured,
                )

        return RuleResult("spatial", RuleOutcome.PASS, measured=measured)

    def _cross_camera(self, track: Any, timestamp_s: float) -> RuleResult:
        rules = self.config.rules.cross_camera
        global_id = getattr(track, "global_id", None)

        if global_id is None:
            # A local, per-camera track that has not been fused. When
            # cross-camera confirmation is required - and it is, by default - a
            # single view is NOT enough to alert, and that is a failure rather
            # than a skip. Treating it as "skipped" would let any lone
            # per-camera detection raise a drone alert, which is precisely the
            # false-positive behaviour the whole system exists to prevent.
            if rules.min_cameras >= 2:
                return RuleResult(
                    "cross_camera",
                    RuleOutcome.FAIL,
                    detail=(
                        "local track with no global identity: only one view exists, and "
                        f"cross_camera.min_cameras is {rules.min_cameras}. A single view "
                        f"cannot distinguish a drone from a bird with enough confidence "
                        f"to alert. Wait for fusion."
                    ),
                )
            return RuleResult(
                "cross_camera",
                RuleOutcome.SKIPPED,
                detail=(
                    "local (per-camera) track with no global identity, and "
                    "cross_camera.min_cameras is 1, so no confirmation is required."
                ),
            )

        recent = [
            (camera_id, node, ts, xy)
            for camera_id, node, ts, xy in self.config.sightings.get(global_id, [])
            if ts >= timestamp_s - rules.max_window_s
        ]
        cameras = {s[0] for s in recent}
        nodes = {s[1] for s in recent}

        measured = {
            "cameras": float(len(cameras)),
            "nodes": float(len(nodes)),
        }

        if len(cameras) < rules.min_cameras:
            return RuleResult(
                "cross_camera",
                RuleOutcome.FAIL,
                detail=(
                    f"{len(cameras)} camera(s) in the last {rules.max_window_s:.1f} s < "
                    f"{rules.min_cameras} - one view cannot distinguish a drone from a "
                    f"bird with enough confidence to alert"
                ),
                measured=measured,
            )

        if rules.require_distinct_nodes and len(nodes) < 2:
            return RuleResult(
                "cross_camera",
                RuleOutcome.FAIL,
                detail=(
                    f"all {len(cameras)} views come from {len(nodes)} worker node(s); "
                    f"the rule requires independent nodes, and cameras on one node share "
                    f"a decoder, a tracker and a failure domain"
                ),
                measured=measured,
            )

        positions = [s[3] for s in recent]
        if len(positions) >= 2:
            spread = max(
                math.dist(positions[0], other) for other in positions[1:]
            )
            measured["position_spread_m"] = spread
            if spread > rules.max_position_disagreement_m:
                return RuleResult(
                    "cross_camera",
                    RuleOutcome.FAIL,
                    detail=(
                        f"views disagree by {spread:.1f} m > "
                        f"{rules.max_position_disagreement_m:.0f} m - at least one is "
                        f"not this target, or the calibration is wrong"
                    ),
                    measured=measured,
                )

        return RuleResult("cross_camera", RuleOutcome.PASS, measured=measured)

    # -- derived measurements ----------------------------------------------- #

    def _turn_rate(self, track: Any) -> float | None:
        """Yaw rate in deg/s from recent heading changes, if any history exists."""
        history = getattr(track, "history", None)
        if not history or len(history) < 3:
            return None

        frames = sorted(history)[-6:]
        if len(frames) < 3:
            return None

        headings: list[tuple[float, float, float]] = []
        for index in range(1, len(frames)):
            previous = history[frames[index - 1]]
            current = history[frames[index]]
            dx = ((current[0] + current[2]) / 2.0) - ((previous[0] + previous[2]) / 2.0)
            dy = ((current[1] + current[3]) / 2.0) - ((previous[1] + previous[3]) / 2.0)
            if math.hypot(dx, dy) < 0.5:
                continue
            elapsed = (frames[index] - frames[index - 1]) / 30.0
            if elapsed <= 0:
                continue
            headings.append((math.atan2(dx, dy), elapsed, frames[index]))

        if len(headings) < 2:
            return None

        total_turn = 0.0
        total_time = 0.0
        for index in range(1, len(headings)):
            delta = abs(_wrap_pi(headings[index][0] - headings[index - 1][0]))
            total_turn += delta
            total_time += headings[index][1]
        if total_time <= 0:
            return None
        return math.degrees(total_turn / total_time)

    def _hover(self, track: Any, timestamp_s: float) -> float | None:
        """Vertical displacement over the hover window, in metres.

        Returns ``None`` when there is not enough of a time history, so the caller
        skips the check instead of assuming it passed.

        This is a *displacement* test, not a speed one: a hovering drone has near
        zero vertical velocity while a bird on a string also has near zero
        vertical velocity. What separates them is that a bird's string is attached
        to something, so its height barely changes either - which is why the
        threshold is 0.5 m over 2 s rather than a tolerance on velocity.
        """
        rules = self.config.rules.kinematics
        if self.config.ground_plane_z_m is None:
            return None
        if getattr(track, "ground_z", None) is None:
            return None

        stamps = getattr(track, "history_timestamps", None)
        if not stamps:
            return None

        window = [ts for ts in stamps.values() if ts >= timestamp_s - rules.hover_duration_s]
        if len(window) < 2:
            return None
        # Require the window to be substantially covered, otherwise a 2-second-old
        # track with two samples would be judged as "hovering".
        if (max(window) - min(window)) < rules.hover_duration_s * 0.8:
            return None

        heights = getattr(track, "history_heights", None)
        if not heights:
            return None
        recent_heights = [
            value
            for ts, value in zip(stamps.values(), heights, strict=False)
            if ts >= timestamp_s - rules.hover_duration_s
        ]
        if len(recent_heights) < 2:
            return None
        return float(max(recent_heights) - min(recent_heights))

    def _geofence(self, track: Any) -> tuple[str, str] | None:
        """First matching geofence as ``(action, id)``."""
        ground_xy = getattr(track, "ground_xy", None)
        if ground_xy is None or not self.config.geofences:
            return None
        ground_z = float(getattr(track, "ground_z", 0.0) or 0.0)

        from ..utils.geometry import point_in_polygon

        for geofence_id, (action, polygon, height_m) in self.config.geofences.items():
            if ground_z > height_m + 1e-6:
                continue
            if point_in_polygon(ground_xy, polygon):
                return (action, geofence_id)
        return None

    # -- severity ----------------------------------------------------------- #

    def _severity(self, track: Any, measured: dict[str, float]) -> Severity:
        """Severity from how much evidence corroborated the track.

        Escalating on corroboration rather than on confidence is the useful
        direction: a 0.61 detection confirmed by four independent cameras is more
        concerning than a 0.9 detection seen once by a PTZ at the edge of its
        range.
        """
        global_id = getattr(track, "global_id", None)
        if global_id is not None:
            recent = self.config.sightings.get(global_id, [])
            cameras = {s[0] for s in recent}
            if len(cameras) >= 4:
                return Severity.CRITICAL
            if len(cameras) >= 2:
                return Severity.WARN

        speed = measured.get("speed_m_s", 0.0)
        if speed >= 20.0:
            return Severity.WARN
        return Severity.INFO


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def build_engine(rules: DroneRules | None = None, *, with_coverage: bool = True) -> RuleEngine:
    """Build a rule engine, optionally seeded from ``coverage_map.yaml``."""
    from ..config.loader import load_coverage_map, load_rules
    from ..tracking.coordination.coverage import CoverageModel

    config = RuleConfig.from_rules(rules or load_rules())

    if with_coverage:
        try:
            coverage_map = load_coverage_map()
        except Exception as exc:
            log.warning(
                "coverage map unavailable; horizon and geofence rules will be skipped",
                extra={"reason": str(exc)},
            )
            return RuleEngine(config)

        for camera in coverage_map.cameras:
            if camera.horizon_y is not None:
                config.horizon_by_camera[camera.id] = camera.horizon_y

        for fence in coverage_map.geofences:
            config.add_geofence(
                fence.id, fence.action, fence.polygon, fence.height_m
            )
        if config.ground_plane_z_m is None:
            config.ground_plane_z_m = coverage_map.default_ground_plane_z_m

        # Derive the horizon for every camera that does not declare one. Without
        # this the above-horizon gate silently skips on all 100 cameras, which
        # reads as "no gate" rather than "no data".
        coverage = CoverageModel(coverage_map, cache_overlaps=False)
        derived = coverage.horizon_by_camera()
        for camera_id, value in derived.items():
            config.horizon_by_camera.setdefault(camera_id, value)

    return RuleEngine(config)


def validate_rules(rules: DroneRules | None = None) -> list[str]:
    """Cross-rule consistency checks pydantic cannot express.

    Returns a list of human-readable problems; empty means the rule set is
    internally consistent. Run by ``anti-uav rules validate``.
    """
    from ..config.loader import load_rules

    resolved = rules or load_rules()
    problems: list[str] = []

    if resolved.persistence.alert_min_hits < resolved.persistence.min_hits:
        problems.append(
            f"persistence.alert_min_hits ({resolved.persistence.alert_min_hits}) is below "
            f"persistence.min_hits ({resolved.persistence.min_hits}); alerts would be "
            f"allowed before a track is confirmed"
        )

    if resolved.persistence.max_gap_frames < resolved.tracking.max_target_age_frames:
        problems.append(
            f"persistence.max_gap_frames ({resolved.persistence.max_gap_frames}) is below "
            f"tracking.max_target_age_frames ({resolved.tracking.max_target_age_frames}); "
            f"a track would be terminated while the tracker still considers it alive"
        )

    if resolved.tracking.handoff_verify_frames * 0.033 > resolved.tracking.handoff_verify_window_s:
        problems.append(
            f"tracking.handoff_verify_frames={resolved.tracking.handoff_verify_frames} at "
            f"30 fps needs {resolved.tracking.handoff_verify_frames / 30.0:.2f} s, longer "
            f"than handoff_verify_window_s={resolved.tracking.handoff_verify_window_s:.2f} s; "
            f"a handoff can never confirm in time"
        )

    if resolved.tracking.recovery_budget_s < resolved.tracking.handoff_trigger_lead_s:
        problems.append(
            f"tracking.recovery_budget_s ({resolved.tracking.recovery_budget_s:.2f}) is "
            f"shorter than the handoff lead ({resolved.tracking.handoff_trigger_lead_s:.2f}); "
            f"recovery would always give up before a handoff could complete"
        )

    if resolved.cross_camera.max_window_s < resolved.persistence.min_duration_s:
        problems.append(
            f"cross_camera.max_window_s ({resolved.cross_camera.max_window_s:.2f}) is "
            f"shorter than persistence.min_duration_s "
            f"({resolved.persistence.min_duration_s:.2f}); a track can be confirmed and "
            f"still have no chance to gather a second view"
        )

    if resolved.tracking.max_uncertainty_radius_m > resolved.cross_camera.max_position_disagreement_m:
        log.info(
            "uncertainty radius exceeds the position disagreement limit; the covariance "
            "will not be the binding constraint on association",
            extra={
                "max_uncertainty_m": resolved.tracking.max_uncertainty_radius_m,
                "max_disagreement_m": resolved.cross_camera.max_position_disagreement_m,
            },
        )

    return problems


def format_evaluation(evaluation: RuleEvaluation) -> str:
    """Readable single-track report for the CLI and the operator UI."""
    lines = [
        f"track {evaluation.track_id}"
        + (f" (global {evaluation.global_id})" if evaluation.global_id is not None else ""),
        f"  verdict  : {'ALERT' if evaluation.alerted else 'no alert'} "
        f"[{evaluation.severity.value}]",
        "",
    ]
    for result in evaluation.results:
        marker = {"pass": "ok  ", "fail": "FAIL", "skipped": "skip"}[result.outcome.value]
        lines.append(f"  [{marker}] {result.name:<14} {result.detail}")
        if result.measured:
            measured = "  ".join(f"{k}={v:.2f}" for k, v in result.measured.items())
            lines.append(f"           {measured}")

    if evaluation.reasons:
        lines.append("")
        lines.append("  blocking reasons:")
        lines.extend(f"    - {r}" for r in evaluation.reasons)
    return "\n".join(lines)


def severity_order(rules: DroneRules | None = None) -> list[Severity]:
    from ..config.loader import load_rules

    resolved = rules or load_rules()
    return [Severity(name) for name in resolved.severity_order if name != Severity.NONE.value]
