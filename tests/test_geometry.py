"""Geometry primitives and the camera coverage model.

The coverage map decides which camera takes over a handoff, so an error in the
FOV maths silently hands tracks to a camera that cannot see the target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anti_uav.tracking.coordination import load_from_config


class TestBoxGeometry:
    def test_identical_boxes_have_iou_one(self) -> None:
        from anti_uav.utils.geometry import box_iou

        box = (10.0, 10.0, 30.0, 30.0)
        assert box_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self) -> None:
        from anti_uav.utils.geometry import box_iou

        assert box_iou((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)) == 0.0

    def test_half_overlap(self) -> None:
        from anti_uav.utils.geometry import box_iou

        # 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
        assert box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)

    def test_zero_area_box_has_no_iou(self) -> None:
        from anti_uav.utils.geometry import box_iou

        assert box_iou((5.0, 5.0, 5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_iou_is_symmetric(self) -> None:
        from anti_uav.utils.geometry import box_iou

        a, b = (0.0, 0.0, 10.0, 10.0), (3.0, 4.0, 20.0, 18.0)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))

    def test_iou_matrix_shape(self) -> None:
        from anti_uav.utils.geometry import iou_matrix

        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [90.0, 90.0, 99.0, 99.0]]
        )
        matrix = iou_matrix(boxes, boxes)
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0)
        assert matrix[0, 2] == 0.0


class TestConversions:
    def test_cxcywh_round_trips(self) -> None:
        from anti_uav.utils.geometry import cxcywh_to_xyxy, xyxy_to_cxcywh

        box = (10.0, 20.0, 30.0, 60.0)
        values = xyxy_to_cxcywh(box, 640.0, 360.0)
        assert cxcywh_to_xyxy(values, 640.0, 360.0) == pytest.approx(box)

    def test_xywh_round_trips(self) -> None:
        from anti_uav.utils.geometry import xywh_to_xyxy, xyxy_to_xywh

        box = (10.0, 20.0, 30.0, 60.0)
        assert xywh_to_xyxy(xyxy_to_xywh(box)) == pytest.approx(box)

    def test_cxcywh_normalises_by_image_size(self) -> None:
        """The detector emits normalised cxcywh; this is where it becomes pixels.

        Centre (0.5, 0.5) of a 640x360 frame is (320, 180); a 0.25 x 0.5 box is
        160 x 180 px, so half-extents are 80 and 90.
        """
        from anti_uav.utils.geometry import cxcywh_to_xyxy

        assert cxcywh_to_xyxy((0.5, 0.5, 0.25, 0.5), 640.0, 360.0) == pytest.approx(
            (240.0, 90.0, 400.0, 270.0)
        )

    def test_cxcywh_does_not_double_scale(self) -> None:
        """A regression guard: the old code multiplied by the image size twice."""
        from anti_uav.utils.geometry import cxcywh_to_xyxy

        x1, y1, x2, y2 = cxcywh_to_xyxy((0.5, 0.5, 0.25, 0.5), 640.0, 360.0)
        assert 0.0 <= x1 < x2 <= 640.0
        assert 0.0 <= y1 < y2 <= 360.0

    def test_box_centre(self) -> None:
        from anti_uav.utils.geometry import box_centre

        assert box_centre((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


class TestPolygons:
    def test_point_in_polygon(self) -> None:
        from anti_uav.utils.geometry import point_in_polygon

        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        assert point_in_polygon((5.0, 5.0), square)
        assert not point_in_polygon((15.0, 5.0), square)

    def test_polygon_area_matches_the_rectangle(self) -> None:
        from anti_uav.utils.geometry import polygon_area

        rectangle = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
        assert polygon_area(rectangle) == pytest.approx(12.0)

    def test_fully_contained_subject_has_ratio_one(self) -> None:
        from anti_uav.utils.geometry import polygon_intersection_ratio

        big = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        small = np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]])
        assert polygon_intersection_ratio(small, big) == pytest.approx(1.0)

    def test_disjoint_polygons_have_ratio_zero(self) -> None:
        from anti_uav.utils.geometry import polygon_intersection_ratio

        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        b = np.array([[50.0, 50.0], [51.0, 50.0], [51.0, 51.0], [50.0, 51.0]])
        assert polygon_intersection_ratio(a, b) == 0.0


class TestAngles:
    @pytest.mark.parametrize("raw", [-4.0, -math.pi, 0.0, math.pi, 4.0, 10.0])
    def test_normalize_angle_stays_in_range(self, raw: float) -> None:
        from anti_uav.utils.geometry import normalize_angle

        assert -math.pi <= normalize_angle(raw) <= math.pi

    def test_bearing_of_each_cardinal(self) -> None:
        """bearing_to returns radians; due south is -pi or +pi depending on sign
        convention, so compare on the circle rather than as a raw number."""
        from anti_uav.utils.geometry import bearing_to

        def deg(a: float, b: float) -> float:
            return math.degrees(bearing_to((0.0, 0.0), (a, b))) % 360.0

        assert deg(0.0, 10.0) == pytest.approx(0.0, abs=1e-6)
        assert deg(10.0, 0.0) == pytest.approx(90.0, abs=1e-6)
        assert deg(0.0, -10.0) == pytest.approx(180.0, abs=1e-6)
        assert deg(-10.0, 0.0) == pytest.approx(270.0, abs=1e-6)


class TestLinearAssignment:
    def test_optimal_assignment_is_found(self) -> None:
        from anti_uav.utils.geometry import linear_assignment

        cost = np.array([[0.1, 0.9], [0.8, 0.2]])
        pairs = linear_assignment(cost, max_cost=1.0)
        mapping = {row: col for row, col, _ in pairs}
        assert mapping == {0: 0, 1: 1}

    def test_assignment_is_one_to_one(self) -> None:
        from anti_uav.utils.geometry import linear_assignment

        rng = np.random.default_rng(3)
        pairs = linear_assignment(rng.random((6, 6)), max_cost=1.0)
        rows = [r for r, _, _ in pairs]
        cols = [c for _, c, _ in pairs]
        assert len(set(rows)) == len(rows)
        assert len(set(cols)) == len(cols)

    def test_costs_above_the_gate_are_dropped(self) -> None:
        from anti_uav.utils.geometry import linear_assignment

        pairs = linear_assignment(np.array([[0.05, 0.99]]), max_cost=0.5)
        assert [c for _, c, _ in pairs] == [0]


class TestNms:
    def test_suppresses_the_overlapping_duplicate(self) -> None:
        from anti_uav.utils.geometry import greedy_nms

        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
        )
        result = greedy_nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
        assert len(result.keep) == 2

    def test_keeps_the_higher_scoring_box(self) -> None:
        from anti_uav.utils.geometry import greedy_nms

        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        result = greedy_nms(boxes, np.array([0.4, 0.9]), 0.5)
        assert list(result.keep) == [1]
        assert list(result.suppressed) == [0]

    def test_class_ids_are_not_suppressed_across_classes(self) -> None:
        """A drone and a bird at the same pixels are two objects, not a duplicate."""
        from anti_uav.utils.geometry import greedy_nms

        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        result = greedy_nms(
            boxes, np.array([0.9, 0.8]), 0.5, class_ids=np.array([0, 1])
        )
        assert len(result.keep) == 2

    def test_empty_input(self) -> None:
        from anti_uav.utils.geometry import greedy_nms

        result = greedy_nms(np.empty((0, 4)), np.empty((0,)), 0.5)
        assert len(result.keep) == 0


class TestCoverageModel:
    @pytest.fixture(scope="class")
    def model(self):
        return load_from_config()

    def test_fleet_shape(self, model) -> None:
        report = model.coverage_report()
        assert report["cameras"] == 100
        assert report["fixed"] == 80
        assert report["ptz"] == 20
        assert report["nodes"] == 8
        assert sorted(report["cameras_per_node"].values()) == [
            12, 12, 12, 12, 13, 13, 13, 13
        ]

    def test_every_protected_zone_is_watched(self, model) -> None:
        report = model.coverage_report()
        assert report["uncovered_zones"] == []
        assert report["all_zones_covered"] is True

    def test_fixed_and_ptz_coverage_account_for_every_zone(self, model) -> None:
        """A zone inside a fixed camera's near-field blind spot is still covered if
        a PTZ can be commanded there. Both lists must add up to the total."""
        report = model.coverage_report()
        total = len(report["fixed_covered_zones"]) + len(report["ptz_only_zones"])
        assert total == report["protected_zones"] == 5

    def test_every_camera_projects_a_sane_polygon(self, model) -> None:
        for camera in model.config.cameras:
            polygon = np.asarray(model.view(camera.id).fov_polygon, float).reshape(-1, 2)
            assert polygon.shape[0] >= 3, camera.id
            assert np.isfinite(polygon).all(), camera.id

    def test_overlap_zones_reference_two_distinct_cameras(self, model) -> None:
        for zone in model.overlap_zones:
            assert len(zone.camera_ids) == 2
            assert zone.camera_ids[0] != zone.camera_ids[1]
            assert zone.primary_camera in zone.camera_ids

    def test_overlap_pairs_are_unique(self, model) -> None:
        """An overlap is a pair. Counting it twice would double the handoff
        candidates the scheduler has to choose between."""
        keys = [tuple(sorted(zone.camera_ids)) for zone in model.overlap_zones]
        assert len(keys) == len(set(keys))

    def test_every_overlap_area_is_positive(self, model) -> None:
        for zone in model.overlap_zones[:200]:
            assert zone.area_m2 > 0.0

    def test_site_centre_is_seen_by_several_cameras(self, model) -> None:
        centre = (
            model.config.default_ground_plane_z_m * 0.0 + 410.0,
            320.0,
        )
        assert len(model.cameras_seeing(centre, height_m=25.0)) >= 3
