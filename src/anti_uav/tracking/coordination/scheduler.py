"""Coverage-aware PTZ scheduling.

Section 3 of the system block diagram's "Coverage-Aware Scheduling & Planning":
keep at least one camera observing every protected zone, estimate coverage lost
when a PTZ moves, avoid sending the last camera covering a region, reposition
predictively to the acquisition area, and arbitrate between competing targets.

Twenty PTZ cameras, many simultaneous targets, and a coverage invariant. The
tension is real: pointing a PTZ at a drone necessarily points it away from whatever
it was covering. This module makes that trade explicit and priced instead of
implicit and regretted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ...utils.logging import get_logger
from .coverage import CoverageModel

log = get_logger(__name__)


class Priority(StrEnum):
    """Why a PTZ is being asked to move."""

    #: Following a handoff before the target exits the sender.
    HANDOFF = "handoff"
    #: Pre-positioning on a predicted flight path.
    PREPOSITION = "preposition"
    #: Restoring a protected zone that lost its only camera.
    COVERAGE_RESTORE = "coverage_restore"
    #: A manual operator request.
    MANUAL = "manual"


@dataclass(slots=True)
class PtzTask:
    """One request for a PTZ camera."""

    camera_id: str
    priority: Priority
    target_position: tuple[float, float]
    pose: tuple[float, float, float]
    #: Global identity this task serves, if any.
    global_id: int | None = None
    #: Higher wins when tasks contend for the same camera.
    urgency: float = 1.0
    #: Earliest frame this task may execute. Keeps a settling camera from being
    #: re-commanded before it has finished the previous move.
    earliest_frame: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "priority": self.priority.value,
            "position": [round(v, 2) for v in self.target_position],
            "pose": [round(v, 2) for v in self.pose],
            "global_id": self.global_id,
            "urgency": round(self.urgency, 3),
            "reason": self.reason,
        }


@dataclass(slots=True)
class SchedulingDecision:
    """What the scheduler decided this cycle."""

    frame_index: int
    assigned: list[PtzTask] = field(default_factory=list)
    deferred: list[PtzTask] = field(default_factory=list)
    rejected: list[tuple[PtzTask, str]] = field(default_factory=list)
    coverage_violations: list[str] = field(default_factory=list)
    zones_uncovered: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.coverage_violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "assigned": [t.to_dict() for t in self.assigned],
            "deferred": [t.to_dict() for t in self.deferred],
            "rejected": [{"task": t.to_dict(), "why": why} for t, why in self.rejected],
            "coverage_violations": list(self.coverage_violations),
            "zones_uncovered": list(self.zones_uncovered),
        }


class PtzScheduler:
    """Allocates 20 PTZ cameras across competing demands.

    Deliberately a *priority scheduler with a coverage veto*, not an optimiser.
    A full assignment problem across 20 cameras and unbounded targets is a
    combinatorial problem whose optimum is not worth the compute when the real
    constraint is "do not leave a zone dark". Greedy by priority, vetoed on
    coverage, and with every rejection logged, is the behaviour an operator can
    reason about at 3 a.m.
    """

    def __init__(
        self,
        coverage: CoverageModel,
        *,
        ptz_camera_ids: Sequence[str] | None = None,
        enforce_coverage: bool = True,
    ) -> None:
        self.coverage = coverage
        self.ptz_cameras = (
            list(ptz_camera_ids)
            if ptz_camera_ids is not None
            else [v.camera_id for v in coverage.views.values() if v.is_ptz]
        )
        self.enforce_coverage = enforce_coverage

        #: camera_id -> frame index before which no new command may be issued.
        self._settling_until: dict[str, int] = {}
        #: Currently assigned task per camera.
        self._assigned: dict[str, PtzTask] = {}

    # -- planning ----------------------------------------------------------- #

    def plan(self, tasks: Sequence[PtzTask], *, frame_index: int) -> SchedulingDecision:
        """Assign tasks to cameras for one cycle."""
        decision = SchedulingDecision(frame_index=frame_index)

        zones_uncovered = self._uncovered_zones()
        decision.zones_uncovered = zones_uncovered
        if zones_uncovered and self.enforce_coverage:
            decision.coverage_violations = zones_uncovered

        ordered = sorted(
            tasks,
            key=lambda t: (self._priority_rank(t.priority), -t.urgency),
        )

        for task in ordered:
            camera_id = task.camera_id
            if camera_id not in self.ptz_cameras:
                decision.rejected.append((task, f"{camera_id} is not a PTZ camera"))
                continue

            if frame_index < max(task.earliest_frame, self._settling_until.get(camera_id, 0)):
                decision.deferred.append(task)
                continue

            if self.enforce_coverage:
                violation = self._coverage_violation(camera_id, task)
                if violation is not None:
                    decision.rejected.append((task, violation))
                    continue

            self._assigned[camera_id] = task
            decision.assigned.append(task)
            self._settling_until[camera_id] = frame_index + self._settle_frames(camera_id)

        if decision.rejected:
            log.warning(
                "ptz tasks rejected",
                extra={
                    "frame": frame_index,
                    "count": len(decision.rejected),
                    "reasons": [why for _t, why in decision.rejected][:5],
                },
            )
        return decision

    def _priority_rank(self, priority: Priority) -> int:
        return {
            Priority.HANDOFF: 0,
            Priority.COVERAGE_RESTORE: 1,
            Priority.PREPOSITION: 2,
            Priority.MANUAL: 3,
        }[priority]

    def _coverage_violation(self, camera_id: str, task: PtzTask) -> str | None:
        """Whether moving this camera darkens a protected zone. ``None`` if fine.

        The block diagram's "Avoids sending the last camera covering a region",
        implemented as: would this camera still be the only watcher of any protected
        zone it currently covers, *and* is it not already covering the new task's
        position?
        """
        view = self.coverage.view(camera_id)
        if view is None:
            return "unknown camera"

        already_sees_target = any(
            view.covers(sample)
            for sample in _zone_samples(self.coverage, task.target_position)
        )
        if already_sees_target:
            return None

        for zone in self.coverage.protected_zones():
            polygon = self.coverage.zone_polygon(zone)
            centroid = polygon.mean(axis=0)
            zone_point = (float(centroid[0]), float(centroid[1]))

            if not view.covers(zone_point):
                continue

            watchers = self.coverage.cameras_seeing(zone_point, height_m=zone.height_m)
            if camera_id in watchers and len(watchers) == 1:
                return (
                    f"{camera_id} is the only camera covering {zone.id} "
                    f"(priority {zone.priority}); moving it would leave the zone dark"
                )
        return None

    def _settle_frames(self, camera_id: str) -> int:
        view = self.coverage.view(camera_id)
        if view is None or not view.is_ptz:
            return 0
        ptz = view.spec.ptz
        settle_s = ptz.settle_time_s * (1.0 + ptz.overshoot_ratio)
        return max(1, int(settle_s * 30.0))

    # -- coverage invariant ------------------------------------------------- #

    def _uncovered_zones(self) -> list[str]:
        return [
            zone.id for zone in self.coverage.protected_zones() if not self.coverage.zone_covered(zone)
        ]

    def restore_coverage(self, *, frame_index: int) -> list[PtzTask]:
        """Build restore tasks for zones that have lost their camera.

        Run after a handoff batch, since that is when coverage is most likely to
        have been given up.
        """
        tasks: list[PtzTask] = []
        for zone in self.coverage.protected_zones():
            if self.coverage.zone_covered(zone):
                continue

            polygon = self.coverage.zone_polygon(zone)
            centroid = polygon.mean(axis=0)
            zone_centre = (float(centroid[0]), float(centroid[1]))

            best: tuple[float, PtzTask] | None = None
            for camera_id in self.ptz_cameras:
                pose = self.coverage.required_pose(camera_id, zone_centre, height_m=zone.height_m)
                if pose is None:
                    continue
                view = self.coverage.view(camera_id)
                if view is None:
                    continue

                sweep = math.hypot(
                    view.bearing_deg(zone_centre) - view.current_yaw_deg,
                    view.elevation_to((zone_centre[0], zone_centre[1], zone.height_m)) - view.current_pitch_deg,
                )
                priority_boost = zone.priority * 2.0
                score = sweep - priority_boost
                task = PtzTask(
                    camera_id=camera_id,
                    priority=Priority.COVERAGE_RESTORE,
                    target_position=zone_centre,
                    pose=pose,
                    urgency=score,
                    reason=f"restore coverage of {zone.id} (priority {zone.priority})",
                )
                if best is None or score < best[0]:
                    best = (score, task)

            if best is not None:
                tasks.append(best[1])

        if tasks:
            log.warning(
                "coverage restore tasks created",
                extra={"frame": frame_index, "tasks": [t.camera_id for t in tasks]},
            )
        return tasks

    # -- pre-positioning --------------------------------------------------- #

    def preposition(
        self,
        global_id: int,
        current_camera: str,
        velocity: tuple[float, float],
        position: tuple[float, float],
        *,
        lead_s: float = 3.0,
        height_m: float = 10.0,
    ) -> PtzTask | None:
        """Aim a PTZ at where a target will be, not where it is.

        The block diagram's "Predictive PTZ repositioning to the acquisition area".
        Leading the target matters more than it sounds: the camera has to be settled
        and centred by the time the target arrives, so aiming at the current position
        guarantees arriving late.
        """
        speed = math.hypot(velocity[0], velocity[1])
        if speed < 1e-3:
            return None

        lead_point = (
            position[0] + velocity[0] * lead_s,
            position[1] + velocity[1] * lead_s,
        )

        candidates: list[tuple[float, str, tuple[float, float, float]]] = []
        for camera_id in self.ptz_cameras:
            if camera_id == current_camera:
                continue
            pose = self.coverage.required_pose(camera_id, lead_point, height_m=height_m)
            if pose is None:
                continue
            view = self.coverage.view(camera_id)
            if view is None:
                continue
            sweep = math.hypot(
                view.bearing_deg(lead_point) - view.current_yaw_deg,
                view.elevation_to((lead_point[0], lead_point[1], height_m))
                - view.current_pitch_deg,
            )
            candidates.append((view.distance_to(lead_point) + sweep * 25.0, camera_id, pose))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _score, camera_id, pose = candidates[0]

        return PtzTask(
            camera_id=camera_id,
            priority=Priority.PREPOSITION,
            target_position=lead_point,
            pose=pose,
            global_id=global_id,
            urgency=1.0,
            reason=f"pre-position {lead_s:.1f} s ahead of a target at {speed:.1f} m/s",
        )

    # -- execution ---------------------------------------------------------- #

    def apply(self, decision: SchedulingDecision) -> list[tuple[str, tuple[float, float, float]]]:
        """Commit assigned tasks to the coverage model. Returns the commands issued."""
        commands: list[tuple[str, tuple[float, float, float]]] = []
        for task in decision.assigned:
            self.coverage.set_pose(task.camera_id, *task.pose)
            commands.append((task.camera_id, task.pose))
        return commands

    def assigned_for(self, camera_id: str) -> PtzTask | None:
        return self._assigned.get(camera_id)

    def reset(self) -> None:
        self._settling_until.clear()
        self._assigned.clear()

    def describe(self) -> str:
        lines = [
            f"ptz cameras   : {len(self.ptz_cameras)}",
            f"coverage veto : {'on' if self.enforce_coverage else 'off'}",
            f"uncovered     : {', '.join(self._uncovered_zones()) or 'none'}",
            f"assigned      : {len(self._assigned)}",
        ]
        for camera_id, task in sorted(self._assigned.items()):
            lines.append(f"  {camera_id}: {task.priority.value} -> {task.reason}")
        return "\n".join(lines)


def _zone_samples(coverage: CoverageModel, position: tuple[float, float]) -> list[tuple[float, float]]:
    """Sample points around a task's target position, for a "does it already see it" test."""
    radius = 10.0
    return [
        position,
        (position[0] + radius, position[1]),
        (position[0] - radius, position[1]),
        (position[0], position[1] + radius),
        (position[0], position[1] - radius),
    ]


def build_handoff_tasks(
    requests: Sequence[Any],
    coverage: CoverageModel,
    *,
    frame_index: int,
    height_m: float = 10.0,
) -> list[PtzTask]:
    """Turn pending handoffs into scheduler tasks."""
    tasks: list[PtzTask] = []
    for request in requests:
        if request.pose is None:
            continue
        tasks.append(
            PtzTask(
                camera_id=request.to_camera,
                priority=Priority.HANDOFF,
                target_position=request.position,
                pose=request.pose,
                global_id=request.global_id,
                urgency=2.0,
                earliest_frame=frame_index,
                reason=(
                    f"handoff {request.from_camera} -> {request.to_camera} for "
                    f"global track {request.global_id}"
                ),
            )
        )
    del coverage, height_m
    return tasks
