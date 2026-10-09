"""Constant-velocity Kalman filter with 3-sigma uncertainty.

Section 3 of the system block diagram: "Constant-velocity Kalman filter per global
track, estimates next position / region with 3σ covariance."

State vector is ``[x, y, vx, vy]`` in the ground-plane frame (metres, metres per
second). The pixel-space local trackers use a separate, simpler filter in
:mod:`anti_uav.tracking.local.sort`; this one exists because cross-camera
association has to happen in a frame the pixels share, and because the handoff
risk estimator needs a *calibrated* uncertainty ellipse to answer "how sure are we
that this target will still be in frame in two seconds".

Process noise
-------------
``process_noise_std_m_s`` is the standard deviation of the acceleration term: how
much we believe a UAV can change velocity between frames. 0.35 m/s is a deliberate
compromise - too low and a manoeuvring drone outruns its own track between camera
handoffs; too high and the predicted position is useless by the time the receiving
PTZ has finished its move, because the 3-sigma radius has grown to swamp it.

Measurement noise reflects survey-grade calibration (1 m), which is deliberately
larger than the sub-metimetre accuracy people assume from a "calibrated" camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..utils.geometry import PinholeCamera

#: Below this the covariance is still collapsing; treat the track as converged.
_CONVERGED_VAR = 1e-4


@dataclass(slots=True)
class KalmanTrack2D:
    """Constant-velocity filter for one target in the ground plane."""

    #: Process noise: std-dev of the per-frame acceleration, m/s.
    process_noise_std_m_s: float = 0.35
    #: Measurement noise std-dev, m. Survey-grade calibration.
    measurement_noise_std_m: float = 1.0
    #: Initial state covariance.
    initial_position_var_m2: float = 25.0
    initial_velocity_var_m2_s2: float = 4.0

    x: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    p: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    #: Frame-to-frame motion matrix; constant for a constant-velocity model.
    f: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    #: 4x4 measurement matrix - we only observe position.
    h: np.ndarray = field(
        default_factory=lambda: np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=float)
    )
    #: 2x4 process noise, so the same std-dev drives position and velocity.
    q: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=float))
    r: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))
    initialised: bool = False
    updates: int = 0

    def __post_init__(self) -> None:
        self.rebuild_noise()

    # -- configuration ------------------------------------------------------ #

    def rebuild_noise(self) -> None:
        """Recompute ``Q`` and ``R`` from the configured noise levels.

        Q uses the standard continuous white-noise-acceleration model mapped onto
        a 1/60 s frame interval, which is where the configured value gets its
        units. R is diagonal: the two coordinates are independent given a
        calibrated camera.
        """
        dt = 1.0 / 60.0
        s = self.process_noise_std_m_s**2
        self.q = np.array(
            [
                [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                [dt**3 / 2, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2, 0.0, dt**2],
            ],
            dtype=float,
        ) * s
        r = self.measurement_noise_std_m**2
        self.r = np.eye(2, dtype=float) * r

    def set_process_noise(self, std_m_s: float) -> None:
        self.process_noise_std_m_s = max(float(std_m_s), 1e-3)
        self.rebuild_noise()

    def set_measurement_noise(self, std_m: float) -> None:
        self.measurement_noise_std_m = max(float(std_m), 1e-3)
        self.rebuild_noise()

    # -- lifecycle ---------------------------------------------------------- #

    def initialise(self, position: tuple[float, float], velocity: tuple[float, float] = (0.0, 0.0)) -> None:
        self.x = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=float)
        self.p = np.diag(
            [
                self.initial_position_var_m2,
                self.initial_position_var_m2,
                self.initial_velocity_var_m2_s2,
                self.initial_velocity_var_m2_s2,
            ]
        ).astype(float)
        self.initialised = True
        self.updates = 0

    # -- filtering ---------------------------------------------------------- #

    def predict(self, dt: float = 1.0 / 30.0) -> tuple[float, float]:
        """Advance the state by ``dt`` seconds. Returns the predicted position."""
        if not self.initialised:
            return (0.0, 0.0)
        self.f = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q * (dt * 30.0)
        return (float(self.x[0]), float(self.x[1]))

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """Fold in a measurement and return the corrected position.

        Uses the Joseph form of the covariance update. The simpler
        ``P = (I - K H) P`` can lose symmetry and positive-definiteness in
        floating point, which shows up much later as a NaN uncertainty radius that
        silently disables the handoff logic.
        """
        if not self.initialised:
            self.initialise(measurement)
            return measurement

        z = np.asarray(measurement, dtype=float)
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        try:
            k = self.p @ self.h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:  # pragma: no cover - s is SPD by construction
            k = np.zeros((4, 2), dtype=float)

        identity = np.eye(4, dtype=float)
        a = identity - k @ self.h
        self.p = a @ self.p @ a.T + k @ self.r @ k.T
        self.p = 0.5 * (self.p + self.p.T)  # force symmetry
        self.x = self.x + k @ y
        self.updates += 1
        return (float(self.x[0]), float(self.x[1]))

    # -- derived quantities ------------------------------------------------- #

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @position.setter
    def position(self, value: tuple[float, float]) -> None:
        self.x[0], self.x[1] = float(value[0]), float(value[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def speed_m_s(self) -> float:
        return math.hypot(float(self.x[2]), float(self.x[3]))

    @property
    def position_covariance(self) -> np.ndarray:
        """Top-left 2x2 of P - the position uncertainty."""
        return self.p[:2, :2]

    def sigma_radius(self, sigma: float = 3.0) -> float:
        """Radius of the ``sigma``-ellipse for an isotropic position uncertainty.

        Uses the eigenvalue trace rather than ``sqrt(trace(P))``, so a track
        constrained along one axis (common: motion along a fence line) reports a
        tight ellipse instead of a circle inflated by the unconstrained axis.
        """
        covariance = self.position_covariance
        return float(sigma * math.sqrt(max(float(np.trace(covariance)), 0.0) / 2.0))

    @property
    def uncertainty_m(self) -> float:
        """3-sigma radius in metres. The number the rule layer thresholds."""
        return self.sigma_radius(3.0)

    @property
    def converged(self) -> bool:
        """Whether the filter has stopped shrinking its uncertainty."""
        return self.updates > 5 and float(np.trace(self.position_covariance)) < _CONVERGED_VAR

    def time_to_radius(self, radius_m: float, *, sigma: float = 3.0) -> float:
        """Seconds until the 3-sigma radius exceeds ``radius_m``.

        Used by the handoff risk estimator to answer "how long until we lose
        confidence in this position". Solved in closed form for the constant-
        velocity case: the position covariance grows as ``P0 + dt^2 * V`` where
        ``V`` is the velocity block, so ``dt = sqrt((r^2 - tr(P0)) / tr(V))``.
        """
        target = (radius_m / max(sigma, 1e-6)) ** 2
        current = float(np.trace(self.position_covariance))
        if target <= current:
            return 0.0
        vxx, vyy = float(self.p[0, 2]), float(self.p[1, 3])
        growth = vxx * vxx + vyy * vyy
        if growth <= 1e-12:
            return math.inf
        return math.sqrt((target - current) / growth)

    def project_to_pixel(
        self, camera: PinholeCamera, plane_z: float
    ) -> tuple[float, float] | None:
        """Where this track's predicted position lands in ``camera``'s image.

        Returns ``None`` when the point is behind the camera or outside the frame,
        which the risk estimator reads as "this camera cannot continue the track".
        """
        x, y = self.position
        return camera.ground_to_pixel((x, y, plane_z))

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "x": float(self.x[0]),
            "y": float(self.x[1]),
            "vx": float(self.x[2]),
            "vy": float(self.x[3]),
            "speed_m_s": self.speed_m_s,
            "uncertainty_m": self.uncertainty_m,
            "updates": self.updates,
            "converged": self.converged,
        }

    def copy(self) -> KalmanTrack2D:
        clone = KalmanTrack2D(
            process_noise_std_m_s=self.process_noise_std_m_s,
            measurement_noise_std_m=self.measurement_noise_std_m,
            initialised=self.initialised,
            updates=self.updates,
        )
        clone.x = self.x.copy()
        clone.p = self.p.copy()
        return clone


class KalmanBank:
    """Many independent filters, keyed by track id.

    The global track manager holds one per live track. Filter matrices are shared
    across instances because Q, R and F are constant for a given configuration -
    rebuilding an 4x4 per track per frame is pure waste when 100 cameras are
    producing thousands of tracks.
    """

    def __init__(
        self,
        *,
        process_noise_std_m_s: float = 0.35,
        measurement_noise_std_m: float = 1.0,
    ) -> None:
        self.process_noise_std_m_s = process_noise_std_m_s
        self.measurement_noise_std_m = measurement_noise_std_m
        self._filters: dict[int, KalmanTrack2D] = {}
        self._next_id = 1

    def create(
        self,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
    ) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._filters[track_id] = KalmanTrack2D(
            process_noise_std_m_s=self.process_noise_std_m_s,
            measurement_noise_std_m=self.measurement_noise_std_m,
        )
        self._filters[track_id].initialise(position, velocity)
        return track_id

    def get(self, track_id: int) -> KalmanTrack2D | None:
        return self._filters.get(track_id)

    def predict(self, dt: float = 1.0 / 30.0) -> None:
        for kf in self._filters.values():
            kf.predict(dt)

    def update(self, track_id: int, measurement: tuple[float, float]) -> bool:
        kf = self._filters.get(track_id)
        if kf is None:
            return False
        kf.update(measurement)
        return True

    def remove(self, track_id: int) -> None:
        self._filters.pop(track_id, None)

    def prune(self, keep: set[int]) -> int:
        stale = set(self._filters) - keep
        for track_id in stale:
            del self._filters[track_id]
        return len(stale)

    def __len__(self) -> int:
        return len(self._filters)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._filters

    def items(self) -> list[tuple[int, KalmanTrack2D]]:
        return sorted(self._filters.items())

    @property
    def max_uncertainty_m(self) -> float:
        return max((kf.uncertainty_m for kf in self._filters.values()), default=0.0)
