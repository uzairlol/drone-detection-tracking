"""Camera coordination.

The five modules of section 3 of the system block diagram, one file each:

==============  ==========================================================
module          diagram element
==============  ==========================================================
coverage        Camera Coverage & Calibration Model
risk/handoff    Trackability / Handoff-Risk Estimator, Handoff Coordinator
recovery        Lost-Target Recovery
scheduler       Coverage-Aware Scheduling & Planning
ptz             PTZ Controller (ONVIF absolute-move)
==============  ==========================================================

They are separate because they have different rates and different failure modes.
Coverage is static and rebuilt rarely; the risk estimator runs per track per frame;
PTZ commands are rate-limited by mechanics. Collapsing them into one "coordinator"
would mean a PTZ settling time could not be tuned without touching coverage
geometry.
"""

from __future__ import annotations

from .coverage import (
    CameraView,
    CoverageModel,
    CoverageZone,
    describe,
    load_from_config,
    summarize_cameras,
)
from .handoff import (
    CandidateCamera,
    HandoffCoordinator,
    HandoffRequest,
    RiskAssessment,
    RiskEstimator,
)
from .ptz import (
    MoveCommand,
    MoveState,
    PtzController,
    PtzPose,
    PtzStatus,
    build_absolute_move,
    build_stop_move,
)
from .recovery import RecoveryAttempt, RecoveryManager, RecoverySearch
from .scheduler import (
    Priority,
    PtzScheduler,
    PtzTask,
    SchedulingDecision,
    build_handoff_tasks,
)

__all__ = [
    "CameraView",
    "CandidateCamera",
    "CoverageModel",
    "CoverageZone",
    "HandoffCoordinator",
    "HandoffRequest",
    "MoveCommand",
    "MoveState",
    "Priority",
    "PtzController",
    "PtzPose",
    "PtzScheduler",
    "PtzStatus",
    "PtzTask",
    "RecoveryAttempt",
    "RecoveryManager",
    "RecoverySearch",
    "RiskAssessment",
    "RiskEstimator",
    "SchedulingDecision",
    "build_absolute_move",
    "build_handoff_tasks",
    "build_stop_move",
    "describe",
    "load_from_config",
    "summarize_cameras",
]
