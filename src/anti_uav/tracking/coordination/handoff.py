"""Handoff risk estimation and coordination.

Section 3 of the system block diagram:

* **Trackability / Handoff-Risk Estimator** - "Trigger handoff if predicted
  time-to-exit < 2.0 s", scoring FOV-boundary distance, target pixel size, PTZ
  state, occlusion risk, and receiver availability.
* **Handoff Coordinator** - a scored candidate-camera list, triggered at the
  predicted exit minus the lead, with a ~3-frame / 0.5 s confirmation step before
  the global identity is transferred.

Why the 2-second lead exists
----------------------------
A PTZ move is not instant. From ``coverage_map.yaml``, a V5925 pan takes up to
``max_pan_speed_deg_s`` to slew plus ``settle_time_s`` (~1.2 s) to physically stop,
and it overshoots by ``overshoot_ratio``. So the command has to be issued *before*
the target exits, and issued early enough that the camera has finished moving when
the target arrives. Two seconds is roughly that budget. Triggering on exit instead
of ahead of it means the receiver is still slewing while the target crosses - which
is why so many naive handoff implementations "hand off successfully" while never
actually transferring an identity.

The confirmation step matters just as much. A command that is sent but not confirmed
would transfer a global identity into a camera that is still moving or is looking at
the wrong place, and the identity would be lost with no way to recover it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...utils.logging import get_logger
from ..types import HandoffEvent, Track
from .coverage import CameraView, CoverageModel

log = get_logger(__name__)


@dataclass(slots=True)
class CandidateCamera:
    """A scored receiver for a handoff."""

    camera_id: str
    node: str
    score: float
    distance_m: float
    #: Pan delta needed, degrees. Large values mean a slow camera.
    pan_delta_deg: float
    tilt_delta_deg: float
    required_zoom: float
    time_to_slew_s: float
    time_to_settle_s: float
    #: Total time until this camera could have the target centred.
    time_to_acquire_s: float
    pose: tuple[float, float, float] | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.pose is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "node": self.node,
            "score": round(self.score, 4),
            "distance_m": round(self.distance_m, 2),
            "pan_delta_deg": round(self.pan_delta_deg, 2),
            "tilt_delta_deg": round(self.tilt_delta_deg, 2),
            "required_zoom": round(self.required_zoom, 2),
            "time_to_slew_s": round(self.time_to_slew_s, 2),
            "time_to_settle_s": round(self.time_to_settle_s, 2),
            "time_to_acquire_s": round(self.time_to_acquire_s, 2),
            "feasible": self.feasible,
            "pose": None if self.pose is None else [round(v, 2) for v in self.pose],
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class RiskAssessment:
    """Whether a track is about to become untrackable, and from which camera."""

    track_id: int
    camera_id: str
    trackable: bool
    #: Ground distance to the FOV boundary along the velocity vector.
    distance_to_exit_m: float | None
    time_to_exit_s: float | None
    #: The block diagram's headline trigger: exit time minus the lead.
    time_until_handoff_s: float | None
    should_trigger: bool
    target_height_px: float
    trackable_height: bool
    occlusion_risk: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "trackable": self.trackable,
            "distance_to_exit_m": (
                None if self.distance_to_exit_m is None else round(self.distance_to_exit_m, 2)
            ),
            "time_to_exit_s": None if self.time_to_exit_s is None else round(self.time_to_exit_s, 2),
            "time_until_handoff_s": (
                None if self.time_until_handoff_s is None else round(self.time_until_handoff_s, 2)
            ),
            "should_trigger": self.should_trigger,
            "target_height_px": round(self.target_height_px, 2),
            "trackable_height": self.trackable_height,
            "occlusion_risk": round(self.occlusion_risk, 3),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class HandoffRequest:
    """A handoff in flight, awaiting confirmation."""

    global_id: int
    from_camera: str
    to_camera: str
    started_frame: int
    position: tuple[float, float]
    confirmations: int = 0
    state: HandoffEvent = HandoffEvent.PENDING
    expected_position: tuple[float, float] = (0.0, 0.0)
    pose: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "started_frame": self.started_frame,
            "confirmations": self.confirmations,
            "state": self.state.value,
            "expected_position": [round(v, 2) for v in self.expected_position],
        }


class RiskEstimator:
    """Scores how trackable a target is and when a handoff must begin."""

    def __init__(
        self,
        coverage: CoverageModel,
        *,
        lead_time_s: float = 2.0,
        min_target_height_px: float = 15.0,
        max_time_to_exit_s: float = 12.0,
    ) -> None:
        self.coverage = coverage
        self.lead_time_s = lead_time_s
        self.min_target_height_px = min_target_height_px
        self.max_time_to_exit_s = max_time_to_exit_s

    def assess(
        self,
        track: Track,
        *,
        ground_position: tuple[float, float] | None = None,
        ground_velocity: tuple[float, float] | None = None,
        height_m: float = 10.0,
    ) -> RiskAssessment:
        """Evaluate one track.

        Every input can be absent (an uncalibrated camera gives no ground
        position). In that case the assessment reports what it *can* determine -
        the pixel-height gate - and says why the rest is unknown, rather than
        silently returning "no risk".
        """
        camera_id = track.camera_id
        reasons: list[str] = []

        height_px = float(track.box[3] - track.box[1])
        trackable_height = height_px >= self.min_target_height_px
        if not trackable_height:
            reasons.append(
                f"target is {height_px:.1f} px tall, below the {self.min_target_height_px:.0f} px "
                f"tracking floor - no tracker can hold this"
            )

        position = ground_position or track.ground_xy
        velocity = ground_velocity or track.velocity_m_s

        if position is None or camera_id not in self.coverage:
            if position is None:
                reasons.append(
                    "no ground position (camera not calibrated) - FOV exit cannot be predicted"
                )
            return RiskAssessment(
                track_id=track.track_id,
                camera_id=camera_id,
                trackable=trackable_height,
                distance_to_exit_m=None,
                time_to_exit_s=None,
                time_until_handoff_s=None,
                should_trigger=False,
                target_height_px=height_px,
                trackable_height=trackable_height,
                occlusion_risk=0.0,
                reasons=reasons,
            )

        speed = math.hypot(velocity[0], velocity[1])
        if speed < 1e-3:
            reasons.append("target is stationary - no FOV exit to predict")
            return RiskAssessment(
                track_id=track.track_id,
                camera_id=camera_id,
                trackable=trackable_height,
                distance_to_exit_m=None,
                time_to_exit_s=None,
                time_until_handoff_s=None,
                should_trigger=False,
                target_height_px=height_px,
                trackable_height=trackable_height,
                occlusion_risk=self._occlusion_risk(camera_id, position),
                reasons=reasons,
            )

        distance = self.coverage.fov_exit_distance(
            camera_id, position, velocity, height_m=height_m
        )
        time_to_exit = (distance / speed) if distance is not None else None

        occlusion = self._occlusion_risk(camera_id, position)

        should_trigger = False
        time_until: float | None = None

        if time_to_exit is not None:
            time_until = time_to_exit - self.lead_time_s
            should_trigger = time_until <= 0.0
            if should_trigger:
                reasons.append(
                    f"FOV exit in {time_to_exit:.2f} s, inside the {self.lead_time_s:.1f} s "
                    f"handoff lead - trigger now"
                )
        else:
            reasons.append(
                f"target stays in this camera's FOV for at least {self.max_time_to_exit_s:.0f} s"
            )

        if not trackable_height:
            reasons.append("below the pixel-height floor: this camera cannot continue the track")

        return RiskAssessment(
            track_id=track.track_id,
            camera_id=camera_id,
            trackable=trackable_height and not should_trigger,
            distance_to_exit_m=distance,
            time_to_exit_s=time_to_exit,
            time_until_handoff_s=time_until,
            should_trigger=should_trigger,
            target_height_px=height_px,
            trackable_height=trackable_height,
            occlusion_risk=occlusion,
            reasons=reasons,
        )

    def _occlusion_risk(self, camera_id: str, position: tuple[float, float]) -> float:
        """How likely the target is to pass behind structure.

        Estimated as the inverse of how many cameras share this camera's view here:
        where many cameras see the same spot, one of them will still have the
        target when this one loses it. A crude proxy, and labelled as one - a real
        implementation would use the depth map or a 3D site model.
        """
        overlapping = self.coverage.overlaps_for(camera_id)
        if not overlapping:
            return 1.0 if camera_id in self.coverage else 0.0
        shared = sum(
            1
            for zone in overlapping
            if point_in_zone(position, zone.polygon)
        )
        if shared <= 0:
            return 1.0
        return float(max(0.0, 1.0 - min(shared, 3) / 3.0))


def point_in_zone(point: tuple[float, float], polygon: np.ndarray) -> bool:
    from ...utils.geometry import point_in_polygon

    return polygon.shape[0] >= 3 and point_in_polygon(point, polygon)


class HandoffCoordinator:
    """Selects receivers, commands them, and confirms the transfer."""

    def __init__(
        self,
        coverage: CoverageModel,
        *,
        lead_time_s: float = 2.0,
        confirm_frames: int = 3,
        confirm_window_s: float = 0.5,
        max_position_disagreement_m: float = 25.0,
        avoid_sending_last_camera: bool = True,
    ) -> None:
        self.coverage = coverage
        self.lead_time_s = lead_time_s
        self.confirm_frames = confirm_frames
        self.confirm_window_s = confirm_window_s
        self.max_position_disagreement_m = max_position_disagreement_m
        self.avoid_sending_last_camera = avoid_sending_last_camera

        self.risk = RiskEstimator(
            coverage,
            lead_time_s=lead_time_s,
            min_target_height_px=15.0,
        )
        self.pending: dict[int, HandoffRequest] = {}
        self.history: list[HandoffRequest] = []

    # -- candidate scoring -------------------------------------------------- #

    def candidates(
        self,
        position: tuple[float, float],
        *,
        from_camera: str = "",
        height_m: float = 10.0,
        max_distance_m: float = 400.0,
        include: Sequence[str] | None = None,
    ) -> list[CandidateCamera]:
        """Score every camera as a potential receiver, best first.

        Scoring balances three things, in decreasing weight:

        * **acquisition time** - can it physically get there before the target
          leaves the sender? This is a gate, not a preference: a camera that cannot
          arrive in time is excluded rather than ranked low.
        * **continuity** - does it already overlap the sender's view? A camera
          already covering the overlap zone does not need to slew at all, so it is
          dramatically faster and far more likely to confirm.
        * **distance** - shorter is better, as a tiebreaker.
        """
        results: list[CandidateCamera] = []

        for view in self.coverage.views.values():
            if not view.spec.enabled:
                continue
            if view.camera_id == from_camera:
                continue
            if include is not None and view.camera_id not in include:
                continue

            distance = view.distance_to(position)
            if distance > max_distance_m:
                continue

            reasons: list[str] = []
            pose = self.coverage.required_pose(view.camera_id, position, height_m=height_m)

            if pose is None:
                reasons.append("no reachable pose: this camera cannot cover the position")
                results.append(
                    CandidateCamera(
                        camera_id=view.camera_id,
                        node=view.spec.node,
                        score=0.0,
                        distance_m=distance,
                        pan_delta_deg=180.0,
                        tilt_delta_deg=90.0,
                        required_zoom=1.0,
                        time_to_slew_s=math.inf,
                        time_to_settle_s=0.0,
                        time_to_acquire_s=math.inf,
                        pose=None,
                        reasons=reasons,
                    )
                )
                continue

            pan_delta, tilt_delta, slew, settle = self._motion_cost(view, pose)
            acquire = slew + settle

            overlaps_sender = bool(
                from_camera and any(from_camera in z.camera_ids for z in self.coverage.overlaps_for(view.camera_id))
            )
            if overlaps_sender:
                reasons.append("already overlaps the sender's view")
                if view.is_ptz:
                    slew *= 0.5
                    acquire = slew + settle

            if self.avoid_sending_last_camera and self._is_last_camera(view, position):
                reasons.append(
                    "sending the only camera covering its zone would leave it unobserved"
                )
                score = 0.05
            else:
                score = self._score(distance, acquire, overlaps_sender)

            results.append(
                CandidateCamera(
                    camera_id=view.camera_id,
                    node=view.spec.node,
                    score=score,
                    distance_m=distance,
                    pan_delta_deg=pan_delta,
                    tilt_delta_deg=tilt_delta,
                    required_zoom=pose[2],
                    time_to_slew_s=slew,
                    time_to_settle_s=settle,
                    time_to_acquire_s=acquire,
                    pose=pose,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda c: -c.score)
        return results

    def _motion_cost(
        self, view: CameraView, pose: tuple[float, float, float]
    ) -> tuple[float, float, float, float]:
        """Pan/tilt deltas and the time to slew there and settle."""
        from ...utils.geometry import normalize_angle

        if not view.is_ptz:
            return (0.0, 0.0, 0.0, 0.0)

        pan_delta = abs(normalize_angle(math.radians(pose[0] - view.current_yaw_deg))) * 180.0 / math.pi
        tilt_delta = abs(pose[1] - view.current_pitch_deg)

        ptz = view.spec.ptz
        slew = math.hypot(
            pan_delta / max(ptz.max_pan_speed_deg_s, 1e-3),
            tilt_delta / max(ptz.max_tilt_speed_deg_s, 1e-3),
        )
        # Overshoot means the camera does not stop where commanded, so the operator
        # (or the pipeline) has to correct - and that correction costs time too.
        settle = ptz.settle_time_s * (1.0 + ptz.overshoot_ratio)
        return (pan_delta, tilt_delta, slew, settle)

    def _is_last_camera(self, view: CameraView, position: tuple[float, float]) -> bool:
        """Whether this camera is the only one covering a protected zone it sees.

        The block diagram's invariant: "Avoids sending the last camera covering a
        region". Violating it means a region goes dark for as long as the move takes
        - which is exactly when an intruder would be exploiting it.
        """
        for zone in self.coverage.protected_zones():
            polygon = self.coverage.zone_polygon(zone)
            centroid = polygon.mean(axis=0)
            samples = [(float(centroid[0]), float(centroid[1]))]
            samples.extend((float(x), float(y)) for x, y in polygon)

            if not any(view.covers(sample) for sample in samples):
                continue

            watchers = [
                camera_id
                for camera_id in self.coverage.cameras_seeing(
                    (float(centroid[0]), float(centroid[1])), height_m=zone.height_m
                )
                if camera_id != view.camera_id
            ]
            if not watchers:
                return True

        return False

    @staticmethod
    def _score(distance_m: float, acquire_s: float, overlaps_sender: bool) -> float:
        """Higher is better. Kept explicit so the weighting is auditable."""
        distance_term = 1.0 / (1.0 + distance_m / 150.0)
        speed_term = 1.0 / (1.0 + acquire_s / 2.0)
        continuity_bonus = 0.25 if overlaps_sender else 0.0
        return float(np.clip(0.5 * distance_term + 0.5 * speed_term + continuity_bonus, 0.0, 1.0))

    # -- state machine ------------------------------------------------------ #

    def trigger(
        self,
        global_id: int,
        from_camera: str,
        position: tuple[float, float],
        *,
        frame_index: int,
        height_m: float = 10.0,
        max_time_to_acquire_s: float = 1.5,
    ) -> HandoffRequest | None:
        """Start a handoff, or return ``None`` if no camera can take the track.

        ``max_time_to_acquire_s`` is the deadline a receiver must beat. Default 1.5 s
        leaves roughly half the 2.0 s lead for the target's remaining flight inside
        the overlap zone, which is the part the confirmation step needs.
        """
        if global_id in self.pending:
            return self.pending[global_id]

        options = self.candidates(position, from_camera=from_camera, height_m=height_m)
        feasible = [
            c
            for c in options
            if c.feasible
            and c.time_to_acquire_s <= max_time_to_acquire_s
            and c.score > 0.0
        ]

        if not feasible:
            log.warning(
                "no viable handoff receiver",
                extra={
                    "global_id": global_id,
                    "from": from_camera,
                    "position": [round(v, 1) for v in position],
                    "deadline_s": max_time_to_acquire_s,
                    "best_rejected": options[0].to_dict() if options else None,
                },
            )
            return None

        chosen = feasible[0]
        request = HandoffRequest(
            global_id=global_id,
            from_camera=from_camera,
            to_camera=chosen.camera_id,
            started_frame=frame_index,
            position=position,
            expected_position=position,
            pose=chosen.pose,
        )
        self.pending[global_id] = request

        log.info(
            "handoff triggered",
            extra={
                "global_id": global_id,
                "from": from_camera,
                "to": chosen.camera_id,
                "distance_m": round(chosen.distance_m, 1),
                "acquire_s": round(chosen.time_to_acquire_s, 2),
                "pose": None if chosen.pose is None else [round(v, 1) for v in chosen.pose],
            },
        )
        return request

    def confirm(
        self,
        global_id: int,
        observation_position: tuple[float, float],
        camera_id: str,
        *,
        frame_index: int,
        appearance_similarity: float | None = None,
    ) -> bool:
        """Feed a confirmation into a pending handoff. ``True`` when it completes.

        Confirmation requires the receiving camera to actually report the target
        **at the position we predicted**. Matching on appearance alone would
        confirm a handoff onto a *different* drone - the worst possible outcome,
        an identity transferred to the wrong target - so position agreement is
        mandatory and appearance is an optional additional gate.
        """
        request = self.pending.get(global_id)
        if request is None:
            return False

        if camera_id != request.to_camera:
            return False

        displacement = math.dist(observation_position, request.expected_position)
        if displacement > self.max_position_disagreement_m:
            log.debug(
                "handoff confirmation rejected: position disagreement",
                extra={
                    "global_id": global_id,
                    "displacement_m": round(displacement, 1),
                    "limit_m": self.max_position_disagreement_m,
                },
            )
            request.confirmations = 0
            return False

        if appearance_similarity is not None and appearance_similarity < 0.55:
            log.debug(
                "handoff confirmation rejected: appearance disagreement",
                extra={"global_id": global_id, "similarity": round(appearance_similarity, 3)},
            )
            request.confirmations = 0
            return False

        request.confirmations += 1
        if request.confirmations >= self.confirm_frames:
            request.state = HandoffEvent.COMPLETED
            del self.pending[global_id]
            self.history.append(request)
            log.info(
                "handoff completed",
                extra={
                    "global_id": global_id,
                    "from": request.from_camera,
                    "to": request.to_camera,
                    "frames_to_confirm": request.confirmations,
                },
            )
            return True
        return False

    def update(
        self,
        global_id: int,
        predicted_position: tuple[float, float],
        *,
        frame_index: int,
    ) -> None:
        """Advance a pending handoff's expected position as the target moves."""
        request = self.pending.get(global_id)
        if request is not None:
            request.expected_position = predicted_position

    def expire(
        self,
        frame_index: int,
        *,
        fps: float = 30.0,
    ) -> list[HandoffRequest]:
        """Fail handoffs that could not be confirmed in time.

        Without this, a request whose camera never confirms would sit in ``pending``
        forever and the target would be stuck with a camera that is pointing away
        from it - a livelock, and one that never surfaces as an error.
        """
        budget_s = max(self.confirm_window_s, 0.5) * self.confirm_frames
        deadline_frames = max(1, int(budget_s * fps * 3))

        expired = [
            request
            for request in self.pending.values()
            if frame_index - request.started_frame > deadline_frames
        ]
        for request in expired:
            request.state = HandoffEvent.FAILED
            del self.pending[request.global_id]
            self.history.append(request)
            log.warning(
                "handoff failed: no confirmation in time",
                extra={
                    "global_id": request.global_id,
                    "to": request.to_camera,
                    "waited_frames": frame_index - request.started_frame,
                },
            )
        return expired

    # -- introspection ------------------------------------------------------ #

    def reset(self) -> None:
        self.pending.clear()

    def describe(self) -> str:
        lines = [f"pending handoffs: {len(self.pending)}", f"completed/failed  : {len(self.history)}"]
        for request in self.pending.values():
            lines.append(
                f"  global {request.global_id}: {request.from_camera} -> {request.to_camera} "
                f"(confirmations {request.confirmations}/{self.confirm_frames})"
            )
        return "\n".join(lines)
