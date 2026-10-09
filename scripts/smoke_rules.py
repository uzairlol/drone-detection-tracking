"""Smoke test of the rule engine (not a test suite entry).

Feeds synthetic tracks through every gate and asserts that each one rejects what
it should. The point is to prove the gates actually discriminate - a rule set that
passes everything is worse than useless, because it looks like it is working.
"""

from __future__ import annotations

from anti_uav.config import load_rules
from anti_uav.rules import (
    RuleConfig,
    RuleEngine,
    Severity,
    build_engine,
    format_evaluation,
    validate_rules,
)
from anti_uav.tracking.types import Track, TrackState


def make_track(**kwargs):
    defaults = {
        "track_id":1,
        "class_id":0,
        "class_name":"drone",
        "camera_id":"fixed-001",
        "state":TrackState.CONFIRMED,
        "confirmed":True,
        "box":(400.0, 200.0, 424.0, 224.0),
        "confidence":0.85,
        "confidence_ema":0.85,
        "hits":30,
        "misses":0,
        "first_frame":0,
        "last_frame":30,
        "first_timestamp_s":0.0,
        "last_timestamp_s":1.0,
        "ground_xy":(300.0, 300.0),
        "ground_z":25.0,
        "velocity_m_s":(12.0, 4.0),
        "global_id":7,
        "image_height_px":1080,
    }
    defaults.update(kwargs)
    track = Track(**defaults)
    # Populate a plausible history so the kinematics sub-checks have something
    # to measure.
    for frame in range(0, 31):
        cx = 400.0 + frame * 6.0
        cy = 200.0 + frame * 1.5
        track.history[frame] = (cx, cy, cx + 24, cy + 24)
        track.history_timestamps[frame] = frame / 30.0
        track.history_heights[frame] = 25.0 + frame * 0.02
    return track


def main() -> int:
    print("=" * 76)
    print("RULE VALIDATION (cross-rule consistency)")
    print("=" * 76)
    problems = validate_rules()
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
    else:
        print("  no cross-rule problems found")

    rules = load_rules()
    engine = build_engine(rules)
    print(f"\n  engine built: {len(engine.config.geofences)} geofences, "
          f"{len(engine.config.horizon_by_camera)} camera horizons, "
          f"ground_plane_z_m={engine.config.ground_plane_z_m}")

    print()
    print("=" * 76)
    print("A NOMINAL TRACK (should alert)")
    print("=" * 76)
    config = engine.config
    config.sightings[7] = [
        ("fixed-001", "node-00", 0.9, (300.0, 300.0)),
        ("fixed-040", "node-00", 1.0, (302.0, 301.0)),
        ("fixed-012", "node-01", 1.0, (299.0, 302.0)),
    ]
    track = make_track()
    evaluation = engine.evaluate(track, timestamp_s=1.0)
    print(format_evaluation(evaluation))
    assert evaluation.alerted, "a nominal track should alert"
    assert evaluation.severity is Severity.WARN, (
        f"expected WARN (3 cameras), got {evaluation.severity}"
    )

    print()
    print("=" * 76)
    print("B - EACH GATE MUST REJECT")
    print("=" * 76)

    cases = [
        (
            "confidence below initiate",
            make_track(confidence=0.45, confidence_ema=0.45, global_id=None),
            1.0,
            "confidence",
        ),
        (
            "not enough hits (persistence)",
            make_track(hits=3, global_id=None),
            1.0,
            "persistence",
        ),
        (
            "tracked too briefly (persistence, duration floor)",
            make_track(hits=14, first_timestamp_s=0.95, global_id=None),
            1.0,
            "persistence",
        ),
        (
            "detection gap too long (persistence)",
            make_track(misses=9, global_id=None),
            1.0,
            "persistence",
        ),
        (
            "too slow (kinematics floor)",
            make_track(velocity_m_s=(0.4, 0.1), global_id=None),
            1.0,
            "kinematics",
        ),
        (
            "too fast (kinematics ceiling)",
            make_track(velocity_m_s=(40.0, 8.0), global_id=None),
            1.0,
            "kinematics",
        ),
        (
            "below the horizon (spatial)",
            make_track(box=(400.0, 1000.0, 424.0, 1024.0), global_id=None),
            1.0,
            "spatial",
        ),
        (
            "one camera only (cross-camera)",
            make_track(global_id=99),
            1.0,
            "cross_camera",
        ),
    ]

    for label, subject, now, expected_gate in cases:
        if subject.global_id == 99:
            config.sightings[99] = [("fixed-001", "node-00", now, (300.0, 300.0))]
        else:
            config.sightings.pop(subject.global_id, None)

        result = engine.evaluate(subject, timestamp_s=now)
        gate = result.first_failure
        mark = "ok " if gate == expected_gate and not result.alerted else "BAD"
        print(f"  [{mark}] {label:<44} -> rejected by {gate}")
        if mark == "BAD":
            print(f"          expected {expected_gate}; got {gate}, alerted={result.alerted}")
            for reason in result.reasons:
                print(f"          {reason}")
        assert mark == "ok ", f"{label}: expected {expected_gate}, got {gate}"

    print()
    print("=" * 76)
    print("C - HITS MORE GATES AT ONCE, AND SAYS WHICH")
    print("=" * 76)
    # Bird-like: slow, low confidence, below the horizon, one camera.
    bird = make_track(
        confidence=0.3, confidence_ema=0.3, hits=8,
        box=(400.0, 1010.0, 418.0, 1020.0),
        velocity_m_s=(0.0, 0.0),
        global_id=123,
    )
    config.sightings[123] = [("fixed-001", "node-00", 1.0, (300.0, 300.0))]
    result = engine.evaluate(bird, timestamp_s=1.0)
    failing = [r.name for r in result.results if r.outcome.value == "fail"]
    print(f"  bird-like track rejected by: {', '.join(failing)}")
    assert not result.alerted
    assert len(failing) >= 3, f"expected several gates to fire, got {failing}"
    print(f"  severity: {result.severity.value}")
    print("  ^ a rule set that reports every failing gate is what makes triage possible")

    print()
    print("=" * 76)
    print("D - HYSTERESIS: maintain < initiate")
    print("=" * 76)
    weak = make_track(confidence=0.45, confidence_ema=0.45, global_id=None)
    first = engine.evaluate(weak, timestamp_s=1.0)
    print(f"  cold (initiate 0.60):  alerted={first.alerted}  -> {first.first_failure}")

    # Now pretend it has already alerted: the maintain floor is 0.35, so 0.45 passes.
    weak.alert = "info"
    config.sightings.setdefault(weak.global_id or 55, [
        ("fixed-001", "node-00", 1.0, (300.0, 300.0)),
        ("fixed-040", "node-00", 1.0, (301.0, 300.0)),
    ])
    weak.global_id = 55
    second = engine.evaluate(weak, timestamp_s=1.0)
    print(f"  warm (maintain 0.35):  alerted={second.alerted}  -> {second.first_failure}")
    assert not first.alerted, "a 0.45 detection must not initiate a track"
    assert second.alerted, "a 0.45 detection should hold an already-alerted track"
    print("  ^ hysteresis: without it, a detector hovering at the threshold flickers")
    print("    tracks on and off and the persistence counter never accumulates")

    print()
    print("=" * 76)
    print("E - SEVERITY SCALES WITH CORROBORATION")
    print("=" * 76)
    for camera_count in (2, 4, 6):
        config.sightings[500] = [
            (f"fixed-{i:03d}", f"node-{i % 8:02d}", 1.0, (300.0 + i, 300.0))
            for i in range(camera_count)
        ]
        subject = make_track(global_id=500)
        result = engine.evaluate(subject, timestamp_s=1.0)
        print(f"  {camera_count} cameras -> {result.severity.value}")
    assert Severity.CRITICAL.value == engine.evaluate(
        make_track(global_id=500), timestamp_s=1.0
    ).severity.value

    print()
    print("=" * 76)
    print("F - A CLEAN ENGINE WITHOUT THE COVERAGE MAP")
    print("=" * 76)
    bare = RuleEngine(RuleConfig.from_rules(rules))
    subject = make_track(global_id=88)
    bare.config.sightings[88] = [
        ("fixed-001", "node-00", 1.0, (300.0, 300.0)),
        ("fixed-040", "node-00", 1.0, (301.0, 300.0)),
    ]
    result = bare.evaluate(subject, timestamp_s=1.0)
    skipped = [r.name for r in result.results if r.outcome.value == "skipped"]
    print(f"  skipped without calibration: {skipped}")
    print(f"  evaluated={bare.config.evaluations}  alerted={result.alerted}")
    assert "spatial" in skipped, "the horizon rule should report itself as skipped"
    assert "kinematics" in skipped, "the speed rule should report itself as skipped"
    print("  ^ a skipped gate says so; a silently-passing gate does not")

    print()
    print("=" * 76)
    print("ALL RULE ENGINE SMOKE CHECKS PASSED")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
