"""Lost-target recovery.

Section 3 of the system block diagram's "Lost-Target Recovery": re-acquire within
~1.5 s of loss, triggered on detection gaps > ``maxTargetAge`` (4 frames), with a
re-acquisition sweep around the predicted position and re-prediction every ~100 ms.

The one design decision that matters
------------------------------------
Recovery does **not** search the whole site. It searches a shrinking disc around
the target's *predicted* position, whose radius is the track's 3-sigma uncertainty.

That is not an optimisation - it is what makes recovery feasible. The alternative,
re-running the detector everywhere, is a 100-camera sweep on a cluster that is
already saturated with 13-stream inference. And searching *wider* would be worse
than useless: a wider sweep finds more birds, and this system's core failure mode is
tracking birds.

Because the uncertainty grows while the target is missing, the sweep naturally
widens over time, which is exactly the right behaviour: the first searches are
precise, later ones accept more error. Searches stop at the recovery budget and the
track is declared lost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...utils.logging import get_logger
from ..types import HandoffEvent, TrackState

log = get_logger(__name__)


@dataclass(slots=True)
class RecoverySearch:
    """One sweep of candidate cameras during recovery."""

    frame_index: int
    global_id: int
    radius_m: float
    uncertainty_m: float
    cameras: list[str]
    #: Time budget for this sweep, given the observed camera latency.
    budget_s: float
    re_predict: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "global_id": self.global_id,
            "radius_m": round(self.radius_m, 2),
            "uncertainty_m": round(self.uncertainty_m, 2),
            "cameras": list(self.cameras),
            "budget_s": round(self.budget_s, 3),
        }


@dataclass(slots=True)
class RecoveryAttempt:
    """The full recovery effort for one lost track."""

    global_id: int
    last_camera: str
    last_position: tuple[float, float]
    started_frame: int
    state: HandoffEvent = HandoffEvent.RECOVERY
    searches: list[RecoverySearch] = field(default_factory=list)
    recovered_camera: str = ""
    recovered_frame: int = -1
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "last_camera": self.last_camera,
            "state": self.state.value,
            "started_frame": self.started_frame,
            "elapsed_s": round(self.elapsed_s, 3),
            "searches": [s.to_dict() for s in self.searches],
            "recovered_camera": self.recovered_camera,
            "recovered_frame": self.recovered_frame,
            "notes": list(self.notes),
        }


class RecoveryManager:
    """Re-acquires lost targets within a bounded time and radius."""

    def __init__(
        self,
        coverage: Any,
        *,
        budget_s: float = 1.5,
        max_gap_frames: int = 4,
        re_predict_interval_s: float = 0.1,
        camera_latency_s: float = 0.25,
        min_radius_m: float = 8.0,
        radius_growth: float = 1.6,
        max_radius_m: float = 60.0,
    ) -> None:
        """
        ``camera_latency_s`` is how long a PTZ re-acquisition takes end to end.
        It is the value that decides whether a camera is worth sweeping: with a
        1.5 s budget and 0.25 s per camera you can usefully visit about six, so
        the shortlist is capped accordingly rather than optimising a list you have
        no time to walk.
        """
        self.coverage = coverage
        self.budget_s = budget_s
        self.max_gap_frames = max_gap_frames
        self.re_predict_interval_s = re_predict_interval_s
        self.camera_latency_s = camera_latency_s
        self.min_radius_m = min_radius_m
        self.radius_growth = radius_growth
        self.max_radius_m = max_radius_m

        self.active: dict[int, RecoveryAttempt] = {}
        self.history: list[RecoveryAttempt] = []

    # -- trigger ------------------------------------------------------------ #

    def should_attempt(
        self,
        global_id: int,
        missed_frames: int,
        *,
        state: TrackState,
    ) -> bool:
        """Whether to start recovery.

        Two conditions, and both matter. The gap must exceed ``max_gap_frames``,
        or we would recover from every single dropped frame. And the track must
        still be recoverable - recovering a track the rule layer has already declared
        lost would resurrect an alert that an operator has been told is closed.
        """
        if global_id in self.active:
            return False
        if state is TrackState.DEAD:
            return False
        if missed_frames < self.max_gap_frames:
            return False
        # TENTATIVE is never confirmed, so there is no alert to protect.
        return state is not TrackState.TENTATIVE

    def begin(
        self,
        global_id: int,
        camera_id: str,
        position: tuple[float, float],
        *,
        frame_index: int,
        uncertainty_m: float = 0.0,
    ) -> RecoveryAttempt:
        """Start a recovery attempt."""
        attempt = RecoveryAttempt(
            global_id=global_id,
            last_camera=camera_id,
            last_position=position,
            started_frame=frame_index,
        )
        self.active[global_id] = attempt
        log.info(
            "recovery started",
            extra={
                "global_id": global_id,
                "last_camera": camera_id,
                "budget_s": self.budget_s,
                "initial_uncertainty_m": round(uncertainty_m, 2),
            },
        )
        return attempt

    # -- search ------------------------------------------------------------- #

    def next_search(
        self,
        attempt: RecoveryAttempt,
        predicted_position: tuple[float, float],
        *,
        frame_index: int,
        uncertainty_m: float,
        fps: float = 30.0,
    ) -> RecoverySearch | None:
        """Plan the next sweep, or ``None`` when the budget is exhausted."""
        elapsed = (frame_index - attempt.started_frame) / max(fps, 1e-6)
        attempt.elapsed_s = elapsed

        if elapsed > self.budget_s:
            attempt.state = HandoffEvent.RECOVERY_FAILED
            self.active.pop(attempt.global_id, None)
            self.history.append(attempt)
            log.warning(
                "recovery budget exhausted",
                extra={"global_id": attempt.global_id, "elapsed_s": round(elapsed, 3)},
            )
            return None

        # Radius = the uncertainty, floored so the first sweep still reaches the
        # cameras that overlap the lost one, and capped so it never becomes a
        # site-wide sweep.
        radius = float(np.clip(max(uncertainty_m, self.min_radius_m), self.min_radius_m, self.max_radius_m))
        if attempt.searches:
            radius = min(radius * self.radius_growth, self.max_radius_m)

        cameras = self.coverage.cameras_seeing(predicted_position) if hasattr(
            self.coverage, "cameras_seeing"
        ) else []
        if not cameras:
            attempt.notes.append("no camera's current FOV contains the predicted position")
            return None

        # Rank by how quickly each can be on target, then keep only as many as the
        # remaining budget allows.
        cameras = self._rank(cameras, predicted_position)
        affordable = max(0, int((self.budget_s - elapsed) / max(self.camera_latency_s, 1e-6)))
        chosen = cameras[: max(1, affordable)] if affordable else cameras[:1]

        last_frame = attempt.searches[-1].frame_index if attempt.searches else attempt.started_frame
        re_predict = (frame_index - last_frame) / max(fps, 1e-6) >= self.re_predict_interval_s

        search = RecoverySearch(
            frame_index=frame_index,
            global_id=attempt.global_id,
            radius_m=radius,
            uncertainty_m=uncertainty_m,
            cameras=chosen,
            budget_s=self.budget_s - elapsed,
            re_predict=re_predict,
        )
        attempt.searches.append(search)
        return search

    def _rank(self, camera_ids: Sequence[str], position: tuple[float, float]) -> list[str]:
        """Nearest-and-fastest first."""
        scored: list[tuple[float, str]] = []
        for camera_id in camera_ids:
            view = self.coverage.view(camera_id)
            if view is None:
                continue
            distance = view.distance_to(position)
            # A PTZ must slew; a fixed camera only needs the target to be in frame.
            slew = 0.0
            if view.is_ptz:
                delta = abs(view.bearing_deg(position) - view.current_yaw_deg)
                slew = delta / max(view.spec.ptz.max_pan_speed_deg_s, 1e-3)
                slew += view.spec.ptz.settle_time_s
            scored.append((distance + slew * 25.0, camera_id))

        scored.sort()
        return [camera_id for _score, camera_id in scored]

    # -- completion --------------------------------------------------------- #

    def attempt_recovery(
        self,
        global_id: int,
        camera_id: str,
        position: tuple[float, float],
        *,
        frame_index: int,
        appearance_similarity: float | None = None,
    ) -> bool:
        """Try to match an observation to an in-progress recovery.

        Requires both positional agreement and - when available - appearance.
        Positional agreement alone inside a 60 m disc at 10 m altitude is a weak
        test, and this system's bird population makes weak tests expensive.
        """
        attempt = self.active.get(global_id)
        if attempt is None:
            return False

        displacement = math.dist(position, attempt.last_position)
        tolerance = max(self.min_radius_m, attempt.searches[-1].radius_m if attempt.searches else self.min_radius_m)
        if displacement > tolerance:
            return False

        if appearance_similarity is not None and appearance_similarity < 0.55:
            attempt.notes.append(
                f"rejected a match at {camera_id}: appearance similarity "
                f"{appearance_similarity:.2f}"
            )
            return False

        attempt.state = HandoffEvent.RECOVERED
        attempt.recovered_camera = camera_id
        attempt.recovered_frame = frame_index
        self.active.pop(global_id, None)
        self.history.append(attempt)

        log.info(
            "target recovered",
            extra={
                "global_id": global_id,
                "from": attempt.last_camera,
                "to": camera_id,
                "elapsed_s": round(attempt.elapsed_s, 3),
                "searches": len(attempt.searches),
            },
        )
        return True

    def reset(self) -> None:
        self.active.clear()

    # -- reporting ---------------------------------------------------------- #

    def summary(self) -> dict[str, Any]:
        recovered = [a for a in self.history if a.state is HandoffEvent.RECOVERED]
        failed = [a for a in self.history if a.state is HandoffEvent.RECOVERY_FAILED]
        elapsed = [a.elapsed_s for a in recovered]
        return {
            "active": len(self.active),
            "attempted": len(self.history),
            "recovered": len(recovered),
            "failed": len(failed),
            "success_rate": len(recovered) / len(self.history) if self.history else 0.0,
            "mean_recovery_s": float(np.mean(elapsed)) if elapsed else 0.0,
            "budget_s": self.budget_s,
        }

    def describe(self) -> str:
        summary = self.summary()
        lines = [
            f"recovery budget : {summary['budget_s']:.2f} s",
            f"attempted       : {summary['attempted']}",
            f"recovered       : {summary['recovered']} ({summary['success_rate']:.0%})",
            f"failed          : {summary['failed']}",
            f"mean recovery   : {summary['mean_recovery_s']:.3f} s",
            f"in progress     : {summary['active']}",
        ]
        for attempt in self.active.values():
            lines.append(
                f"  global {attempt.global_id}: {len(attempt.searches)} searches, "
                f"{attempt.elapsed_s:.2f} s elapsed"
            )
        return "\n".join(lines)
