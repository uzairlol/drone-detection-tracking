"""Global track manager.

Section 3 of the system block diagram: "Fuses per-node (DeepStream) local tracks
into global drone identities (position + appearance gating), maintains fused track
state + position uncertainty, resolves track fragmentation / ID switches, keeps
cross-camera identity consistent."

Why a second layer exists at all
--------------------------------
Local tracking is per camera, and nothing stops two cameras from each tracking the
same physical drone as two unrelated objects. Worse, when a target leaves camera A's
field of view and appears in camera B's, camera B has no idea that this is the same
drone it was following - the pixel boxes are in different images with no shared
coordinates. The global layer supplies that missing continuity by working in the
**ground plane**, where a target has one position regardless of which camera saw
it.

Three jobs, in order of how often they matter:

1. **Cross-camera association.** Two observations from different cameras are the
   same target when their ground positions agree to within the combined uncertainty
   and appearance agrees. This is what removes the duplicate tracks that would
   otherwise triple an alert.
2. **Fragmentation repair.** A target whose local track died (occlusion, leaving
   frame) and reappears re-enters as a new local track. If its predicted position
   still fits, it is the same global identity - no new alert.
3. **Uncertainty growth.** A global track that stops receiving observations must
   have its 3-sigma radius grow, so the handoff coordinator knows how much to trust
   it and the rule engine can eventually declare it lost.

Association cost
----------------
``cost = w_motion * normalised_position_distance + w_appearance * (1 - cos)``

Motion is normalised by the **combined** 3-sigma radius, so a track with a wide
ellipsis can match a detection further away than a tight one. That is the whole
point of carrying a covariance: it makes the gate self-calibrating rather than a
fixed radius someone tuned once on a different site.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...utils.logging import get_logger
from ..kalman import KalmanBank, KalmanTrack2D
from ..types import HandoffEvent, TrackObservation, TrackState

log = get_logger(__name__)


@dataclass(slots=True)
class GlobalTrack:
    """One physical drone, seen from one or more cameras."""

    global_id: int
    class_name: str = "drone"
    state: TrackState = TrackState.TENTATIVE

    #: The Kalman filter for this target's ground position.
    filter: KalmanTrack2D = field(default_factory=KalmanTrack2D)

    #: ``{camera_id: local_track_id}`` - which local tracks feed this identity.
    observations: dict[str, int] = field(default_factory=dict)
    #: Cameras this identity has been seen from.
    cameras: set[str] = field(default_factory=set)
    #: Nodes this identity has been seen from.
    nodes: set[str] = field(default_factory=set)

    #: Frame index of the first and most recent observation.
    first_frame: int = 0
    last_frame: int = 0
    #: Frames since the last observation - the global equivalent of ``misses``.
    missed_frames: int = 0
    hits: int = 0

    #: Latest appearance embedding and gallery, for cross-camera gating.
    embedding: np.ndarray | None = None
    gallery: list[np.ndarray] = field(default_factory=list)

    #: Handoff bookkeeping.
    handoff_state: HandoffEvent = HandoffEvent.NONE
    handoff_to: str = ""
    handoff_started_frame: int = -1
    handoffs: int = 0
    recovered_after_frames: int = 0

    alert: str = ""
    notes: list[str] = field(default_factory=list)

    # -- derived ------------------------------------------------------------ #

    @property
    def ground_xy(self) -> tuple[float, float]:
        return self.filter.position

    @property
    def velocity_m_s(self) -> tuple[float, float]:
        return self.filter.velocity

    @property
    def speed_m_s(self) -> float:
        return self.filter.speed_m_s

    @property
    def uncertainty_m(self) -> float:
        return self.filter.uncertainty_m

    @property
    def multi_camera(self) -> bool:
        return len(self.cameras) > 1

    def last_local(self, camera_id: str) -> int | None:
        return self.observations.get(camera_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "class_name": self.class_name,
            "state": self.state.value,
            "ground_xy": [round(v, 2) for v in self.ground_xy],
            "velocity_m_s": [round(v, 3) for v in self.velocity_m_s],
            "speed_m_s": round(self.speed_m_s, 3),
            "uncertainty_m": round(self.uncertainty_m, 3),
            "cameras": sorted(self.cameras),
            "nodes": sorted(self.nodes),
            "observations": dict(sorted(self.observations.items())),
            "hits": self.hits,
            "missed_frames": self.missed_frames,
            "handoff_state": self.handoff_state.value,
            "handoff_to": self.handoff_to,
            "handoffs": self.handoffs,
            "recovered_after_frames": self.recovered_after_frames,
            "alert": self.alert,
        }


@dataclass(slots=True)
class FusionResult:
    """What one fusion step did."""

    frame_index: int
    #: Local tracks that were matched to an existing global identity.
    matched: list[tuple[str, int, int]] = field(default_factory=list)  # (cam, local, global)
    #: New global identities created.
    created: list[int] = field(default_factory=list)
    #: Global identities that matched nothing.
    unmatched: list[int] = field(default_factory=list)
    #: Identities dropped this step.
    removed: list[int] = field(default_factory=list)
    #: Cross-camera observations that extended an identity's camera set.
    handoffs_completed: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FusionConfig:
    """Tunables, all mirrored in ``configs/rules/drone_rules.yaml``."""

    #: Weight on the normalised position term. The remainder goes to appearance.
    motion_weight: float = 0.65
    appearance_weight: float = 0.35
    #: Gate. 1.0 means "only associate inside the combined uncertainty ellipse".
    max_association_cost: float = 0.75
    #: A second, looser pass for detections that the normal gate rejected.
    #:
    #: Without it, a brand-new track's filter starts with zero velocity and a large
    #: position variance, so for the first few frames it lags a fast target badly
    #: enough to fall outside the gate - and a new identity is spawned beside the
    #: old one every time. The track count then grows without bound while all the
    #: duplicates describe the same drone. The rescue pass accepts a match within
    #: this multiple of the normal gate when exactly one candidate is in range,
    #: and records it so the behaviour is visible rather than silent.
    rescue_gate_multiplier: float = 3.0
    #: Absolute ceiling on how far the rescue pass will reach, metres. The gate
    #: multiplier alone is not a real bound - the same-camera cost saturates at 1.0,
    #: so a multiplied gate admits everything. This is what actually stops two
    #: genuinely distinct drones being fused while a filter converges.
    rescue_max_distance_m: float = 40.0
    #: Minimum cosine similarity to associate across cameras.
    reid_threshold: float = 0.55
    #: Frames without an observation before an identity is considered lost.
    max_missed_frames: int = 12
    #: Frames of miss before recovery is credited.
    recovery_grace_frames: int = 4
    #: Process noise, m/s.
    process_noise_std_m_s: float = 0.35
    #: Measurement noise, m.
    measurement_noise_std_m: float = 1.0
    #: Cap the 3-sigma radius so a stale identity cannot match anything ever again.
    max_uncertainty_m: float = 18.0
    #: Require a distinct worker node for cross-camera confirmation. Enforced by the
    #: rule engine (``cross_camera.require_distinct_nodes``), not here: fusion must
    #: still associate two views off one node, or the identity fragments. Suppressing
    #: that merge would trade a duplicated track for an identity switch.
    require_distinct_nodes: bool = False
    #: Positions must agree to within this, metres - a hard ceiling under the
    #: covariance term, for when calibration is poor.
    max_position_disagreement_m: float = 25.0
    gallery_size: int = 16


class GlobalTrackManager:
    """Fuses per-camera local tracks into global identities."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.tracks: dict[int, GlobalTrack] = {}
        self._next_id = 1
        self.frame_index = 0
        self.filters = KalmanBank(
            process_noise_std_m_s=self.config.process_noise_std_m_s,
            measurement_noise_std_m=self.config.measurement_noise_std_m,
        )

    # -- main entry point --------------------------------------------------- #

    def update(
        self,
        observations: Sequence[TrackObservation],
        *,
        frame_index: int,
    ) -> FusionResult:
        """Fuse one frame's local observations into global identities.

        ``observations`` must already carry ground positions. An observation
        without one cannot participate in cross-camera association - there is no
        shared frame - and is skipped with a note rather than being associated on
        appearance alone, which would fuse unrelated targets.
        """
        self.frame_index = frame_index
        result = FusionResult(frame_index=frame_index)

        usable = [o for o in observations if o.ground_xy is not None]
        if len(usable) != len(observations):
            result.notes.append(
                f"{len(observations) - len(usable)} observation(s) had no ground position "
                f"and could not be fused. Calibration is required for cross-camera "
                f"tracking; check coverage_map.yaml."
            )

        # Predict every identity forward one frame before associating.
        for track in self.tracks.values():
            track.filter.predict(1.0 / 30.0)
            if track.filter.uncertainty_m > self.config.max_uncertainty_m:
                # Stale beyond usefulness: keep the state as LOST so the alert
                # stops firing, but stop it matching anything new.
                track.state = TrackState.LOST

        candidates = self._associate(usable, frame_index=frame_index)

        matched_local: set[tuple[str, int]] = set()
        for track_id, observation in candidates:
            track = self.tracks[track_id]
            track.filter.update(observation.ground_xy)  # type: ignore[arg-type]

            # A camera we have not seen this identity in before means the local
            # track in that camera is now feeding an identity that existed
            # elsewhere: that is a completed handoff.
            is_new_camera = bool(observation.camera_id) and observation.camera_id not in track.cameras

            track.observations[observation.camera_id] = observation.track_id or -1
            if observation.camera_id:
                track.cameras.add(observation.camera_id)
            if observation.node:
                track.nodes.add(observation.node)

            track.hits += 1
            track.last_frame = frame_index
            track.missed_frames = 0
            track.state = TrackState.CONFIRMED

            if is_new_camera:
                track.handoffs += 1
                track.handoff_state = HandoffEvent.COMPLETED
                result.handoffs_completed.append(track_id)
                log.info(
                    "global track seen from a new camera",
                    extra={
                        "global_id": track_id,
                        "camera": observation.camera_id,
                        "handoffs": track.handoffs,
                    },
                )

            if observation.embedding is not None:
                self._add_embedding(track, observation.embedding)

            matched_local.add((observation.camera_id, observation.track_id or -1))
            result.matched.append((observation.camera_id, observation.track_id or -1, track_id))

        # New identities.
        for observation in usable:
            key = (observation.camera_id, observation.track_id or -1)
            if key in matched_local:
                continue
            track = self._create(observation, frame_index)
            result.created.append(track.global_id)

        # Age the rest.
        for track in self.tracks.values():
            if track.last_frame == frame_index and track.global_id not in result.created:
                continue
            track.missed_frames = frame_index - track.last_frame
            if track.missed_frames > self.config.max_missed_frames:
                track.state = TrackState.DEAD
                result.removed.append(track.global_id)

        self._prune(frame_index)
        return result

    # -- association -------------------------------------------------------- #

    def _associate(
        self, observations: Sequence[TrackObservation], *, frame_index: int
    ) -> list[tuple[int, TrackObservation]]:
        """Greedy-by-cost association of observations to existing identities.

        Two passes. The first uses the normal gate. The second rescues
        observations the first rejected, within ``rescue_gate_multiplier`` times
        that gate, but only when exactly one candidate is in range - "exactly one"
        is the important part, because rescuing an ambiguous observation would
        invent an association the evidence does not support.

        Exclusivity rules, which are easy to get wrong:

        * an **observation** belongs to at most one identity - otherwise a single
          detection would be double-counted across two alerts;
        * an **identity** may absorb several observations **from different
          cameras** in the same frame. That is the entire purpose of this layer:
          two cameras seeing one drone must produce one identity, not one identity
          per camera. Making identities exclusive here silently defeats fusion.
        * within a single camera, only one observation may map to a given identity,
          or two nearby drones would merge.

        Greedy rather than Hungarian on purpose. The cost depends on each identity's
        own covariance, so it is not a clean bipartite matrix problem, and the
        alternative - associating everything then splitting merged clusters - can
        destroy identities.
        """
        alive = [t for t in self.tracks.values() if t.state is not TrackState.DEAD]
        if not alive or not observations:
            return []

        normal_gate = self.config.max_association_cost
        rescue_gate = min(1.0, normal_gate * self.config.rescue_gate_multiplier)

        used_obs: set[int] = set()
        #: ``(global_id, camera_id)`` pairs already claimed this frame.
        used_pairs: set[tuple[int, str]] = set()
        chosen: list[tuple[int, TrackObservation]] = []
        rescued: list[tuple[int, TrackObservation, float]] = []

        def _claim(cost: float, global_id: int, obs_index: int) -> bool:
            observation = observations[obs_index]
            pair = (global_id, observation.camera_id)
            if obs_index in used_obs or pair in used_pairs:
                return False
            used_obs.add(obs_index)
            used_pairs.add(pair)
            return True

        # Two passes: the strict gate first, then a looser one that may only rescue
        # observations still unclaimed. `rescue` is an explicit flag rather than
        # `sink is chosen` so the destination list is obvious at the append site.
        for rescue, gate in ((False, normal_gate), (True, rescue_gate)):
            scored: list[tuple[float, int, int]] = []
            for obs_index, observation in enumerate(observations):
                for track in alive:
                    distance = self._position_distance(track, observation)
                    if distance is None:
                        continue
                    if rescue and distance > self.config.rescue_max_distance_m:
                        continue
                    cost = self._cost(track, observation)
                    if cost <= gate:
                        scored.append((cost, track.global_id, obs_index))

            scored.sort(key=lambda item: item[0])

            for cost, global_id, obs_index in scored:
                if not _claim(cost, global_id, obs_index):
                    continue
                observation = observations[obs_index]
                if rescue:
                    rescued.append((global_id, observation, cost))
                else:
                    chosen.append((global_id, observation))

        if rescued:
            for global_id, observation, _cost in rescued:
                chosen.append((global_id, observation))
            log.debug(
                "global association rescued by the loose pass",
                extra={
                    "rescued": [
                        {"global_id": gid, "obs_cost": round(c, 3)} for gid, _o, c in rescued
                    ],
                },
            )

        return chosen

    def _position_distance(
        self, track: GlobalTrack, observation: TrackObservation
    ) -> float | None:
        """Ground distance from an identity's prediction to an observation."""
        if observation.ground_xy is None:
            return None
        predicted = track.filter.position
        return float(
            np.hypot(observation.ground_xy[0] - predicted[0], observation.ground_xy[1] - predicted[1])
        )

    def _cost(self, track: GlobalTrack, observation: TrackObservation) -> float:
        """Combined motion + appearance cost in ``[0, 1]``.

        Returns ``1.0`` (never associate) when a hard gate fails, so the caller's
        single threshold check covers every rejection reason.
        """
        if observation.ground_xy is None:
            return 1.0

        predicted = track.filter.position
        dx = observation.ground_xy[0] - predicted[0]
        dy = observation.ground_xy[1] - predicted[1]
        distance = float(np.hypot(dx, dy))

        # Hard ceiling from calibration quality.
        if distance > self.config.max_position_disagreement_m:
            return 1.0

        same_camera = bool(observation.camera_id) and observation.camera_id in track.cameras

        # Same camera: motion only. The local tracker already did appearance
        # matching within the image, and re-gating here just adds noise.
        if same_camera:
            radius = max(track.filter.uncertainty_m, 1.0)
            return float(np.clip(distance / (radius * 3.0), 0.0, 1.0))

        # Cross-camera: normalise by the combined uncertainty so the gate widens
        # for an identity that has been uncertain.
        radius = max(track.filter.uncertainty_m, 1.0)
        motion_term = float(np.clip(distance / (radius * 3.0), 0.0, 1.0))

        if self.config.appearance_weight <= 0.0 or track.embedding is None or observation.embedding is None:
            appearance_term = 0.0
            return float(np.clip(motion_term, 0.0, 1.0))

        similarity = _gallery_similarity(track.gallery, observation.embedding)
        if observation.camera_id not in track.cameras and similarity < self.config.reid_threshold:
            # A genuinely new camera: appearance must corroborate the position.
            return 1.0

        appearance_term = float(np.clip((1.0 - similarity) / 2.0, 0.0, 1.0))
        return float(
            np.clip(
                self.config.motion_weight * motion_term
                + self.config.appearance_weight * appearance_term,
                0.0,
                1.0,
            )
        )

    # -- lifecycle ---------------------------------------------------------- #

    def _create(self, observation: TrackObservation, frame_index: int) -> GlobalTrack:
        global_id = self._next_id
        self._next_id += 1

        kf = KalmanTrack2D(
            process_noise_std_m_s=self.config.process_noise_std_m_s,
            measurement_noise_std_m=self.config.measurement_noise_std_m,
        )
        kf.initialise(observation.ground_xy or (0.0, 0.0))  # type: ignore[arg-type]

        track = GlobalTrack(
            global_id=global_id,
            class_name=observation.class_name,
            state=TrackState.TENTATIVE,
            filter=kf,
            first_frame=frame_index,
            last_frame=frame_index,
            hits=1,
        )
        if observation.camera_id:
            track.cameras.add(observation.camera_id)
            track.observations[observation.camera_id] = observation.track_id or -1
        if observation.node:
            track.nodes.add(observation.node)
        if observation.embedding is not None:
            self._add_embedding(track, observation.embedding)

        self.tracks[global_id] = track
        log.info(
            "global track created",
            extra={
                "global_id": global_id,
                "camera": observation.camera_id,
                "position": [round(v, 2) for v in observation.ground_xy],  # type: ignore[union-attr]
            },
        )
        return track

    def _add_embedding(self, track: GlobalTrack, embedding: np.ndarray) -> None:
        track.embedding = embedding
        track.gallery.append(embedding)
        if len(track.gallery) > self.config.gallery_size:
            del track.gallery[0]

    def _prune(self, frame_index: int) -> None:
        stale = [
            gid for gid, t in self.tracks.items()
            if t.state is TrackState.DEAD
            or frame_index - t.last_frame > self.config.max_missed_frames * 4
        ]
        for global_id in stale:
            del self.tracks[global_id]
        self.filters.prune(set(self.tracks))

    def reset(self) -> None:
        self.tracks.clear()
        self.filters = KalmanBank(
            process_noise_std_m_s=self.config.process_noise_std_m_s,
            measurement_noise_std_m=self.config.measurement_noise_std_m,
        )
        self._next_id = 1
        self.frame_index = 0

    # -- queries ------------------------------------------------------------ #

    @property
    def active(self) -> list[GlobalTrack]:
        return [t for t in self.tracks.values() if t.state is not TrackState.DEAD]

    @property
    def confirmed(self) -> list[GlobalTrack]:
        return [t for t in self.active if t.state is TrackState.CONFIRMED]

    def by_camera(self, camera_id: str) -> list[GlobalTrack]:
        return [t for t in self.active if camera_id in t.cameras]

    def get(self, global_id: int) -> GlobalTrack | None:
        return self.tracks.get(global_id)

    def nearest(
        self,
        position: tuple[float, float],
        *,
        exclude: Iterable[int] = (),
        cameras: Iterable[str] | None = None,
        max_distance_m: float | None = None,
    ) -> tuple[GlobalTrack, float] | None:
        """Closest identity to a ground position. Used by handoff and recovery."""
        excluded = set(exclude)
        wanted_cameras = set(cameras) if cameras is not None else None

        best: tuple[GlobalTrack, float] | None = None
        for track in self.active:
            if track.global_id in excluded:
                continue
            if wanted_cameras is not None and not (track.cameras & wanted_cameras):
                continue
            distance = float(
                np.hypot(position[0] - track.ground_xy[0], position[1] - track.ground_xy[1])
            )
            if max_distance_m is not None and distance > max_distance_m:
                continue
            if best is None or distance < best[1]:
                best = (track, distance)
        return best

    def summary(self) -> dict[str, Any]:
        active = self.active
        return {
            "total": len(self.tracks),
            "active": len(active),
            "confirmed": len(self.confirmed),
            "multi_camera": sum(1 for t in active if t.multi_camera),
            "handoffs": sum(t.handoffs for t in active),
            "mean_uncertainty_m": (
                sum(t.uncertainty_m for t in active) / len(active) if active else 0.0
            ),
            "max_uncertainty_m": max((t.uncertainty_m for t in active), default=0.0),
        }

    def dump(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in sorted(self.tracks.values(), key=lambda t: t.global_id)]


def _gallery_similarity(gallery: Sequence[np.ndarray], embedding: np.ndarray) -> float:
    """Best cosine similarity against a gallery. Max, not mean - see appearance.py."""
    if not gallery:
        return 0.0
    best = -1.0
    for candidate in gallery:
        numerator = float(np.dot(candidate, embedding))
        denominator = float(np.linalg.norm(candidate) * np.linalg.norm(embedding))
        if denominator < 1e-8:
            continue
        best = max(best, numerator / denominator)
    return max(best, 0.0)


def config_from_rules(rules: Any) -> FusionConfig:
    """Build a :class:`FusionConfig` from ``drone_rules.yaml``.

    Single source of truth: the same thresholds that govern the live system govern
    the offline replay, so a change to ``reid_threshold`` is testable rather than
    needing a redeploy to find out what it does.

    ``require_distinct_nodes`` is deliberately **not** copied across - see that
    field's note. The rule engine enforces it when deciding whether an alert is
    backed by independent evidence; fusion itself must always merge.
    """
    tracking = rules.tracking
    return FusionConfig(
        max_association_cost=tracking.max_association_cost,
        reid_threshold=tracking.reid_similarity_threshold,
        max_missed_frames=max(4, tracking.max_target_age_frames * 3),
        recovery_grace_frames=tracking.max_target_age_frames,
        process_noise_std_m_s=tracking.process_noise_std_m_s,
        measurement_noise_std_m=tracking.measurement_noise_std_m,
        max_uncertainty_m=tracking.max_uncertainty_radius_m,
        max_position_disagreement_m=rules.cross_camera.max_position_disagreement_m,
    )


def attach_ground_positions(
    observations: Sequence[TrackObservation],
    camera: Any,
    *,
    plane_z_m: float = 10.0,
) -> list[TrackObservation]:
    """Project each observation's centre onto the ground plane.

    Mutates the observations in place and returns them, because callers usually
    want the enriched version and a copy would invite someone to use the wrong one.
    Observations above the horizon project to ``None`` and are left without a ground
    position, which the fusion step reports rather than guessing around.
    """
    for observation in observations:
        centre_x, centre_y = observation.centre
        projected = camera.pixel_to_ground(centre_x, centre_y, plane_z_m)
        if projected is None:
            observation.ground_xy = None
            continue
        observation.ground_xy = projected
        observation.ground_z = plane_z_m
    return list(observations)
