"""Smoke test of global tracking + coordination on the generated 100-camera map.

Runs the whole section-3 stack against configs/coverage/coverage_map.yaml with a
scripted target, and asserts each stage produces sane output. Everything runs on
CPU - no detector or GPU is involved, because this layer's inputs are ground
positions, which is what makes it testable in isolation.
"""

from __future__ import annotations

import math

from anti_uav.config import load_coverage_map, load_rules
from anti_uav.tracking.coordination import (
    CoverageModel,
    HandoffCoordinator,
    Priority,
    PtzController,
    PtzPose,
    PtzScheduler,
    PtzTask,
    RecoveryManager,
    RiskEstimator,
)
from anti_uav.tracking.coordination import (
    describe as describe_coverage,
)
from anti_uav.tracking.global_tracker import (
    GlobalTrackManager,
    config_from_rules,
)
from anti_uav.tracking.types import Track, TrackObservation, TrackState


def main() -> int:
    rules = load_rules()
    coverage_map = load_coverage_map()
    coverage = CoverageModel(coverage_map)

    print("=" * 76)
    print("COVERAGE MODEL")
    print("=" * 76)
    print(describe_coverage(coverage))
    report = coverage.coverage_report()
    print(f"\n  coverage complete: {report['all_zones_covered']} "
          f"(uncovered: {report['uncovered_zones'] or 'none'})")

    centre = (410.0, 320.0)
    print(f"\n  cameras seeing site centre: {len(coverage.cameras_seeing(centre))}")
    print(f"  ptz cameras: {sum(1 for v in coverage.views.values() if v.is_ptz)}")

    ptzs = [v.camera_id for v in coverage.views.values() if v.is_ptz]
    print(f"  required_pose to centre (ptz[0]={ptzs[0]}): "
          f"{coverage.required_pose(ptzs[0], centre, height_m=10.0)}")

    print()
    print("=" * 76)
    print("GLOBAL TRACK FUSION - two cameras seeing one target")
    print("=" * 76)
    fusion = GlobalTrackManager(config_from_rules(rules))
    target = (300.0, 300.0)
    # 20 m/s, inside the 35 m/s ceiling in drone_rules.yaml. Fast, but a real UAV.
    speed = 20.0
    for f in range(20):
        pos = (target[0] + f * speed / 30.0, target[1] + f * speed * 0.3 / 30.0)
        obs = [
            TrackObservation(
                frame_index=f, timestamp_s=f / 30.0,
                box=(pos[0] - 8, pos[1] - 8, pos[0] + 8, pos[1] + 8),
                confidence=0.9, class_id=0, class_name="drone",
                camera_id="fixed-001", node="node-00", ground_xy=pos, track_id=1,
            )
        ]
        if f >= 10:
            obs.append(
                TrackObservation(
                    frame_index=f, timestamp_s=f / 30.0,
                    box=(pos[0] - 7, pos[1] - 7, pos[0] + 7, pos[1] + 7),
                    confidence=0.85, class_id=0, class_name="drone",
                    camera_id="fixed-040", node="node-00", ground_xy=pos, track_id=1,
                )
            )
        res = fusion.update(obs, frame_index=f)
        if f in (0, 9, 10, 19):
            print(f"  f={f:2d} active={len(fusion.active)} created={len(res.created)} "
                  f"matched={len(res.matched)} handoffs={len(res.handoffs_completed)}")
    summary = fusion.summary()
    print(f"\n  summary: {summary}")
    assert summary["active"] == 1, f"expected ONE global identity, got {summary['active']}"
    assert summary["handoffs"] >= 1, "expected a handoff once a 2nd camera saw the target"
    print("  ^ two cameras, one identity. That is the entire point of the global layer.")

    print()
    print("=" * 76)
    print("RISK ESTIMATOR - FOV exit prediction")
    print("=" * 76)
    risk = RiskEstimator(coverage, lead_time_s=rules.tracking.handoff_trigger_lead_s,
                         min_target_height_px=rules.tracking.min_trackable_height_px)

    # Start a target 60 m out along a bearing that points AWAY from the camera, so
    # its FOV exit is guaranteed rather than incidental.
    cam_id = "fixed-001"
    view = coverage.view(cam_id)
    bearing = view.bearing_to((view.centre[0] + 30.0, view.centre[1] + 30.0))
    pos = (
        view.centre[0] + 60.0 * math.sin(bearing),
        view.centre[1] + 60.0 * math.cos(bearing),
    )
    # Fly radially outward, directly away from the camera.
    direction = (pos[0] - view.centre[0], pos[1] - view.centre[1])
    norm = math.hypot(*direction) or 1.0
    direction = (direction[0] / norm, direction[1] / norm)
    speed = 20.0

    track = Track(track_id=1, camera_id=cam_id, box=(0, 0, 24, 24),
                  ground_xy=pos, state=TrackState.CONFIRMED, hits=20)

    triggered_at = None
    assessment = None
    for step in range(200):
        p = (pos[0] + direction[0] * step * 3.0, pos[1] + direction[1] * step * 3.0)
        track.ground_xy = p
        assessment = risk.assess(track, ground_position=p,
                                 ground_velocity=(direction[0] * speed, direction[1] * speed),
                                 height_m=10.0)
        if assessment.should_trigger:
            triggered_at = step
            print(f"  handoff trigger at step {step}:")
            for reason in assessment.reasons:
                print(f"    - {reason}")
            print(f"    distance_to_exit = {assessment.distance_to_exit_m:.1f} m")
            print(f"    time_to_exit     = {assessment.time_to_exit_s:.2f} s")
            print(f"    time_until       = {assessment.time_until_handoff_s:.2f} s")
            print(f"    occlusion_risk   = {assessment.occlusion_risk:.2f}")
            print(f"    target height    = {assessment.target_height_px:.1f} px "
                  f"(floor {rules.tracking.min_trackable_height_px:.0f})")
            break
    assert triggered_at is not None, "risk estimator never triggered a handoff"
    assert assessment is not None and assessment.time_until_handoff_s is not None
    assert assessment.time_until_handoff_s <= 0.0, "trigger fired before the lead window"

    # A target below the pixel floor must be flagged regardless of geometry.
    tiny = Track(track_id=2, camera_id=cam_id, box=(0, 0, 20, 8), state=TrackState.CONFIRMED)
    tiny_assess = risk.assess(tiny, ground_position=pos,
                              ground_velocity=(direction[0] * speed, direction[1] * speed))
    print(f"\n  8 px target: trackable_height={tiny_assess.trackable_height} "
          f"(floor is {rules.tracking.min_trackable_height_px:.0f} px)")
    assert not tiny_assess.trackable_height

    print()
    print("=" * 76)
    print("HANDOFF COORDINATOR - receiver scoring + confirmation")
    print("=" * 76)
    coordinator = HandoffCoordinator(
        coverage,
        lead_time_s=rules.tracking.handoff_trigger_lead_s,
        confirm_frames=rules.tracking.handoff_verify_frames,
        confirm_window_s=rules.tracking.handoff_verify_window_s,
    )
    candidates = coordinator.candidates(pos, from_camera=cam_id, height_m=10.0)
    print(f"  candidates: {len(candidates)}  feasible: {sum(1 for c in candidates if c.feasible)}")
    for c in candidates[:4]:
        print(f"    {c.camera_id:<12} score={c.score:.3f} d={c.distance_m:6.1f}m "
              f"acq={c.time_to_acquire_s:5.2f}s pan={c.pan_delta_deg:6.1f} "
              f"zoom={c.required_zoom:5.1f} feasible={c.feasible}")
        for r in c.reasons:
            print(f"        {r}")
    assert candidates, "no candidates at all"
    assert candidates[0].feasible, "the best candidate is not reachable"
    assert candidates[0].score >= candidates[-1].score, "candidates are not sorted"

    request = coordinator.trigger(99, cam_id, pos, frame_index=0, height_m=10.0)
    print(f"\n  triggered: {request is not None}")
    if request is not None:
        print(f"    {request.from_camera} -> {request.to_camera}  pose={request.pose}")
        # A wrong position must NOT confirm.
        wrong = (pos[0] + 400.0, pos[1] + 400.0)
        rejected = coordinator.confirm(99, wrong, request.to_camera, frame_index=1)
        print(f"    confirm at 400 m off: {rejected}")
        assert not rejected

        # A correct position confirms after exactly verify_frames calls. Stop at
        # the first True: once the handoff completes the request is removed, so
        # further calls correctly return False and would mask the result.
        confirmed = False
        calls = 0
        for i in range(rules.tracking.handoff_verify_frames * 2):
            calls += 1
            if coordinator.confirm(99, pos, request.to_camera, frame_index=2 + i):
                confirmed = True
                break
        print(f"    confirm at the predicted position: {confirmed} "
              f"after {calls} call(s)")
        assert confirmed, "handoff never confirmed at the correct position"
        assert calls == rules.tracking.handoff_verify_frames, (
            f"confirmed after {calls} calls, expected "
            f"{rules.tracking.handoff_verify_frames}"
        )
        print(f"    state: {request.state.value}")
    print(f"\n  {coordinator.describe()}")

    print()
    print("=" * 76)
    print("RECOVERY")
    print("=" * 76)
    recovery = RecoveryManager(
        coverage,
        budget_s=rules.tracking.recovery_budget_s,
        max_gap_frames=rules.tracking.max_target_age_frames,
    )
    attempt = recovery.begin(99, cam_id, pos, frame_index=0)
    search = recovery.next_search(attempt, pos, frame_index=1, uncertainty_m=3.0)
    print(f"  search radius={search.radius_m:.1f}m uncertainty={search.uncertainty_m:.1f}m "
          f"budget={search.budget_s:.2f}s cameras={len(search.cameras)}")
    assert search is not None and search.cameras
    print(f"    cameras: {search.cameras[:6]}")

    far = (pos[0] + 300.0, pos[1] + 300.0)
    rejected = recovery.attempt_recovery(99, search.cameras[0], far, frame_index=2)
    print(f"  attempt recovery 300 m away: {rejected}")
    assert not rejected
    accepted = recovery.attempt_recovery(99, search.cameras[0], pos, frame_index=3)
    print(f"  attempt recovery at the right place: {accepted}")
    assert accepted
    print(f"  second attempt (already resolved): "
          f"{recovery.attempt_recovery(99, search.cameras[0], pos, frame_index=4)}")

    exhausted = recovery.begin(100, cam_id, pos, frame_index=0)
    nxt = recovery.next_search(exhausted, pos, frame_index=10000, uncertainty_m=3.0)
    print(f"  search after the budget expires: {nxt}")
    assert nxt is None and exhausted.state.value == "recovery_failed"
    print(f"\n  {recovery.describe()}")

    print()
    print("=" * 76)
    print("PTZ SCHEDULER")
    print("=" * 76)
    scheduler = PtzScheduler(coverage)
    print(f"  {scheduler.describe()}")
    task = PtzTask(
        camera_id=ptzs[0], priority=Priority.HANDOFF,
        target_position=centre, pose=coverage.required_pose(ptzs[0], centre, height_m=10.0) or (0, 0, 1),
        global_id=99, urgency=2.0, reason="smoke test",
    )
    decision = scheduler.plan([task], frame_index=0)
    print(f"  assigned={len(decision.assigned)} rejected={len(decision.rejected)} "
          f"violations={decision.coverage_violations}")
    for t, why in decision.rejected:
        print(f"    rejected {t.camera_id}: {why}")

    restore = scheduler.restore_coverage(frame_index=1)
    print(f"  coverage restore tasks: {len(restore)}")
    for t in restore:
        print(f"    {t.camera_id}: {t.reason}")

    pre = scheduler.preposition(99, cam_id, (18.0, 9.0), pos)
    print(f"  preposition task: {pre.to_dict()['camera_id'] if pre else None}"
          f"  ({pre.reason if pre else 'no lead camera'})")

    print()
    print("=" * 76)
    print("PTZ CONTROLLER (ONVIF)")
    print("=" * 76)
    sent: list[tuple[str, str]] = []
    clock = {"t": 0.0}
    controller = PtzController(coverage, transport=lambda c, x: sent.append((c, x)), clock=lambda: clock["t"])
    target_pose = coverage.required_pose(ptzs[0], centre, height_m=10.0)
    assert target_pose is not None
    command = controller.move(ptzs[0], PtzPose(*target_pose))
    print(f"  commanded: {command.to_dict()}")
    print(f"  onvif options: {command.onvif_options}")
    print(f"  xml sent ({len(sent[0][1])} bytes), normalised pan/tilt present: "
          f"{'PanTilt' in sent[0][1]}")
    assert "PanTilt" in sent[0][1] and "ContinuousMove" in sent[0][1]

    print(f"  busy before settle: {controller.is_busy(ptzs[0])}")
    clock["t"] = command.predicted_slew_s + 0.01
    print(f"  poll during slew    : {controller.poll(ptzs[0])}")
    clock["t"] = command.predicted_arrival_s + 0.01
    print(f"  poll after arrival  : {controller.poll(ptzs[0])}")
    print(f"  final pose: {controller.pose(ptzs[0]).to_dict()}")
    assert not controller.is_busy(ptzs[0])

    limits = controller.limits(ptzs[0])
    print(f"  limits: {limits}")

    from anti_uav.tracking.coordination.ptz import build_stop_move, compose_pan_series

    print(f"  stop xml: {'Stop' in build_stop_move()}")
    # A 300 deg request normalises to -60 deg, so one short move is correct.
    direct = compose_pan_series(0.0, 300.0, max_step_deg=170.0)
    print(f"  pan 0->300 (normalised): {len(direct)} step(s) -> {[round(s) for s in direct]}")
    assert len(direct) == 1, "a 300 deg request should take the short way round, once"

    # A 120 deg request against a 100 deg step limit must be split.
    split = compose_pan_series(0.0, 120.0, max_step_deg=100.0)
    print(f"  pan 0->120 with a 100 deg limit: {len(split)} steps -> {[round(s) for s in split]}")
    assert len(split) > 1, "a 120 deg pan against a 100 deg limit should be split"
    assert abs(split[-1] - 120.0) < 1e-6, f"the series must land on 120, got {split[-1]}"

    print()
    print("=" * 76)
    print("ALL COORDINATION SMOKE CHECKS PASSED")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
