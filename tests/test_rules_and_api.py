"""The rule / threshold layer, and the API surface built on top of it.

The policy the engine implements is worth stating, because it is easy to
misread from the outside:

* only ``FAIL`` blocks an alert - ``alerted = not failed and class_id == 0``;
* a gate whose input is unavailable returns ``SKIPPED`` with a reason naming
  what to configure, rather than silently passing;
* a bird is rejected by ``class_id``, not by the geometry gates.

That means an *uncalibrated* site runs three of the five gates (confidence,
persistence, cross-camera) and reports the other two as skipped. Both halves of
that are tested here: the gates that must block, and the gates that must admit
they cannot decide.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from anti_uav.api.app import app
from anti_uav.config import load_rules
from anti_uav.config.schema import DroneRules
from anti_uav.rules import RuleEngine, RuleOutcome
from anti_uav.rules.engine import RuleConfig
from anti_uav.tracking.types import Track, TrackState


def make_engine(rules: DroneRules | None = None) -> RuleEngine:
    """A fresh engine per call.

    Sightings live on the config, not the engine, so a shared fixture would leak
    one test's cross-camera history into the next and let the negative cases pass
    for the wrong reason.
    """
    return RuleEngine(RuleConfig.from_rules(rules or load_rules()))


def calibrated_rules() -> DroneRules:
    """Rules with a ground plane and a horizon, so all five gates can run."""
    rules = load_rules().model_copy(deep=True)
    rules.spatial.ground_plane_z_m = 0.0
    rules.spatial.horizon_y = 0.45
    return rules


def two_camera_sightings(engine: RuleEngine) -> None:
    engine.config.record_sighting(7, "fixed-001", "node-00", 1.4, (300.0, 300.0))
    engine.config.record_sighting(7, "fixed-040", "node-00", 1.4, (301.0, 300.0))


def drone_track(**overrides):
    """A long, confident, fast, high, cross-camera-confirmed track."""
    defaults = {
        "track_id": 1,
        "class_id": 0,
        "class_name": "drone",
        "camera_id": "fixed-001",
        "node": "node-00",
        "state": TrackState.CONFIRMED,
        "confirmed": True,
        "box": (400.0, 200.0, 424.0, 224.0),
        "confidence": 0.85,
        "confidence_ema": 0.85,
        "hits": 30,
        "misses": 0,
        "first_frame": 0,
        "last_frame": 30,
        "first_timestamp_s": 0.0,
        "last_timestamp_s": 1.5,
        "ground_xy": (300.0, 300.0),
        "ground_z": 25.0,
        "velocity_m_s": (12.0, 4.0),
        "global_id": 7,
        "image_height_px": 1080,
    }
    defaults.update(overrides)
    return Track(**defaults)


def failing_gates(evaluation) -> list[str]:
    return [r.name for r in evaluation.results if r.outcome is RuleOutcome.FAIL]


def skipped_gates(evaluation) -> list[str]:
    return [r.name for r in evaluation.results if r.outcome is RuleOutcome.SKIPPED]


class TestHappyPath:
    def test_a_strong_calibrated_drone_alerts(self) -> None:
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        assert evaluation.alerted is True
        assert failing_gates(evaluation) == []
        assert skipped_gates(evaluation) == []

    def test_severity_is_reported(self) -> None:
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        assert evaluation.severity

    def test_every_gate_names_itself_and_explains_itself(self) -> None:
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        names = {r.name for r in evaluation.results}
        assert names == {"confidence", "persistence", "kinematics", "spatial", "cross_camera"}


class TestGateRejections:
    """Each gate must reject the case it exists to catch."""

    @pytest.mark.parametrize(
        "overrides, gate",
        [
            ({"confidence": 0.10, "confidence_ema": 0.10}, "confidence"),
            ({"hits": 3}, "persistence"),
            ({"first_timestamp_s": 1.45, "last_timestamp_s": 1.5}, "persistence"),
            ({"misses": 9}, "persistence"),
            ({"global_id": None}, "cross_camera"),
        ],
    )
    def test_gates_that_need_no_calibration(self, overrides: dict, gate: str) -> None:
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(**overrides), timestamp_s=1.5)
        assert evaluation.alerted is False, overrides
        assert gate in failing_gates(evaluation)

    @pytest.mark.parametrize(
        "overrides, gate",
        [
            ({"velocity_m_s": (0.2, 0.0)}, "kinematics"),
            ({"velocity_m_s": (80.0, 0.0)}, "kinematics"),
            ({"box": (400.0, 1010.0, 418.0, 1020.0)}, "spatial"),
        ],
    )
    def test_gates_that_need_calibration(self, overrides: dict, gate: str) -> None:
        """Speed and horizon can only be judged once the site is calibrated."""
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(**overrides), timestamp_s=1.5)
        assert evaluation.alerted is False, overrides
        assert gate in failing_gates(evaluation)

    def test_one_camera_is_not_enough(self) -> None:
        engine = make_engine(calibrated_rules())
        engine.config.record_sighting(7, "fixed-001", "node-00", 1.4, (300.0, 300.0))
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        assert "cross_camera" in failing_gates(evaluation)

    def test_a_bird_is_rejected_by_class_not_by_geometry(self) -> None:
        """Documented behaviour: the gates describe a drone; class_id is the
        discriminator. A bird can satisfy every geometric gate and still be
        rejected."""
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        evaluation = engine.evaluate(
            drone_track(class_id=1, class_name="bird"), timestamp_s=1.5
        )
        assert evaluation.alerted is False
        assert failing_gates(evaluation) == []


class TestSkippedGates:
    """A gate that cannot decide must say so, and must name what to configure."""

    def test_uncalibrated_site_skips_kinematics_and_spatial(self) -> None:
        engine = make_engine()
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        assert set(skipped_gates(evaluation)) == {"kinematics", "spatial"}

    def test_skip_is_reported_in_reasons(self) -> None:
        engine = make_engine()
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        assert any("kinematics skipped" in reason for reason in evaluation.reasons)

    def test_skip_detail_names_the_missing_setting(self) -> None:
        engine = make_engine()
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        spatial = next(r for r in evaluation.results if r.name == "spatial")
        assert "horizon_y" in spatial.detail

    def test_skip_is_not_passed(self) -> None:
        engine = make_engine()
        two_camera_sightings(engine)
        evaluation = engine.evaluate(drone_track(), timestamp_s=1.5)
        for gate in evaluation.results:
            assert gate.outcome in set(RuleOutcome)
        spatial = next(r for r in evaluation.results if r.name == "spatial")
        assert spatial.outcome is RuleOutcome.SKIPPED
        assert spatial.outcome is not RuleOutcome.PASS


class TestHysteresis:
    def test_an_alerted_track_survives_a_confidence_drop(self) -> None:
        """Between `maintain` and `initiate`, an alerting track must not flicker
        off - that is what the two thresholds are for."""
        rules = load_rules()
        between = (rules.confidence.maintain + rules.confidence.initiate) / 2
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        track = drone_track(confidence=between, confidence_ema=between)
        assert engine.evaluate(track, timestamp_s=1.5, already_alerted=True).alerted

    def test_the_same_track_does_not_alert_when_it_was_not_already_alerting(self) -> None:
        rules = load_rules()
        between = (rules.confidence.maintain + rules.confidence.initiate) / 2
        engine = make_engine(calibrated_rules())
        two_camera_sightings(engine)
        track = drone_track(confidence=between, confidence_ema=between)
        assert not engine.evaluate(track, timestamp_s=1.5).alerted


class TestApi:
    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        return TestClient(app, raise_server_exceptions=False)

    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_datasets_flags_the_drone_only_ones(self, client: TestClient) -> None:
        response = client.get("/api/datasets")
        assert response.status_code == 200
        datasets = {d["alias"]: d for d in response.json()}
        assert set(datasets) == {"dvb", "mavvid", "antiuav", "mmuav"}
        assert not datasets["antiuav"]["has_bird_negatives"]
        assert not datasets["mmuav"]["has_bird_negatives"]
        assert datasets["dvb"]["has_bird_negatives"]
        assert datasets["mavvid"]["has_bird_negatives"]

    def test_matrix_has_fourteen_untrained_cells(self, client: TestClient) -> None:
        response = client.get("/api/matrix")
        assert response.status_code == 200
        body = response.json()
        assert len(body["cells"]) == 14
        assert all(cell["map50"] is None for cell in body["cells"]), (
            "an untrained cell must report null, never 0"
        )

    def test_coverage_reports_the_fleet(self, client: TestClient) -> None:
        response = client.get("/api/coverage")
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["cameras"] == 100
        assert summary["fixed"] == 80
        assert summary["ptz"] == 20
        assert summary["all_zones_covered"] is True
        assert len(summary["fixed_covered_zones"]) + len(summary["ptz_only_zones"]) == 5

    def test_rules_endpoint_returns_the_active_config(self, client: TestClient) -> None:
        response = client.get("/api/rules")
        assert response.status_code == 200
        body = response.json()
        assert body["rules"]["confidence"]["initiate"] > 0
        assert isinstance(body["problems"], list)

    def test_rules_explain_accepts_a_strong_track(self, client: TestClient) -> None:
        response = client.post(
            "/api/rules/explain",
            json={
                "box": [400, 200, 424, 224],
                "confidence": 0.85,
                "hits": 30,
                "duration_s": 1.5,
                "camera_id": "fixed-001",
                "image_height_px": 1080,
                "global_id": 7,
                "velocity_m_s": [12.0, 4.0],
                "ground_xy": [300.0, 300.0],
                "ground_z": 25.0,
                "sightings": {
                    "7": [
                        {"camera_id": "fixed-001", "node": "node-00", "t": 1.4, "ground_xy": [300, 300]},
                        {"camera_id": "fixed-040", "node": "node-00", "t": 1.4, "ground_xy": [301, 300]},
                    ]
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["alerted"] is True
        assert body["gates"]

    def test_rules_explain_rejects_a_bird(self, client: TestClient) -> None:
        response = client.post(
            "/api/rules/explain",
            json={
                "box": [400, 1010, 418, 1020],
                "confidence": 0.30,
                "hits": 4,
                "duration_s": 0.1,
                "camera_id": "fixed-001",
                "image_height_px": 1080,
                "ground_xy": [300.0, 300.0],
                "ground_z": 25.0,
            },
        )
        assert response.status_code == 200
        assert response.json()["alerted"] is False

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/does-not-exist").status_code == 404

    def test_unknown_dataset_is_404(self, client: TestClient) -> None:
        assert client.get("/api/datasets/nope/stats").status_code == 404

    def test_ui_is_served_offline(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "chart.umd.min.js" in response.text
        assert "cdn" not in response.text.lower()

    def test_vendored_chart_is_served(self, client: TestClient) -> None:
        assert client.get("/static/vendor/chart.umd.min.js").status_code == 200

    def test_static_bundle_is_served(self, client: TestClient) -> None:
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/app.css").status_code == 200
