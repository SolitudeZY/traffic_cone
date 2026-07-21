#!/usr/bin/env python3
"""Regression tests for traffic-cone geometry rules."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ground_projection import GroundPlaneProjector
from traffic_rules import ConeDetection, analyze_cones
from infer_obb import (
    _events,
    _extract_detections,
    _parse_roi,
    _same_model_exclusion_boxes,
    build_parser,
)
from train_obb import create_resolved_data_config


def detection(x: float, y: float, angle: float = 90.0) -> ConeDetection:
    radians = math.radians(angle)
    long_axis = (math.cos(radians) * 5.0, math.sin(radians) * 5.0)
    short_axis = (-math.sin(radians) * 2.0, math.cos(radians) * 2.0)
    polygon = (
        (x - long_axis[0] - short_axis[0], y - long_axis[1] - short_axis[1]),
        (x - long_axis[0] + short_axis[0], y - long_axis[1] + short_axis[1]),
        (x + long_axis[0] + short_axis[0], y + long_axis[1] + short_axis[1]),
        (x + long_axis[0] - short_axis[0], y + long_axis[1] - short_axis[1]),
    )
    return ConeDetection(polygon, 0.9)


def ground_detection(
    projector: GroundPlaneProjector,
    ground_point: tuple[float, float],
    image_size: tuple[int, int],
) -> ConeDetection:
    ground_x, ground_y = projector.ground_to_image(ground_point, image_size)
    return detection(ground_x, ground_y - 5.0)


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeObb:
    def __init__(self, polygons, confidences, class_ids) -> None:
        self.xyxyxyxy = FakeTensor(polygons)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(class_ids)

    def __len__(self) -> int:
        return len(self.cls.values)


class TrafficRuleTests(unittest.TestCase):
    def test_roi_parser(self) -> None:
        self.assertEqual(_parse_roi("10,20,100,200"), (10.0, 20.0, 100.0, 200.0))
        with self.assertRaises(Exception):
            _parse_roi("100,20,10,200")

    def test_inference_defaults_to_45_degree_perspective_correction(self) -> None:
        args = build_parser().parse_args(
            ["--weights", "weights.pt", "--source", "video.mp4"]
        )

        self.assertEqual(args.camera_pitch_deg, 45.0)
        self.assertEqual(args.camera_hfov_deg, 84.0)
        self.assertFalse(args.no_perspective_correction)

    def test_dataset_path_is_resolved_relative_to_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            source = dataset / "data.yaml"
            source.write_text("path: .\ntrain: images/train\nnames: [cone]\n", encoding="utf-8")
            resolved = create_resolved_data_config(source, root / "runs")
            config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            self.assertEqual(config["path"], str(dataset.resolve()))

    def test_same_model_person_box_suppresses_overlapping_cone(self) -> None:
        result = SimpleNamespace(
            names={0: "cone", 2: "person"},
            obb=FakeObb(
                polygons=[
                    [[40, 40], [60, 40], [60, 60], [40, 60]],
                    [[20, 10], [80, 10], [80, 90], [20, 90]],
                ],
                confidences=[0.8, 0.9],
                class_ids=[0, 2],
            ),
        )
        boxes = _same_model_exclusion_boxes(result, {"person", "vest"})
        cones, detections, suppressed = _extract_detections(
            result, "cone", None, boxes
        )
        self.assertFalse(cones)
        self.assertEqual([item["label"] for item in detections], ["person"])
        self.assertEqual(suppressed, 1)

    def test_regular_row_has_no_alerts(self) -> None:
        result = analyze_cones([detection(x, 0) for x in range(0, 90, 10)])
        self.assertFalse(result.missing)
        self.assertFalse(result.displaced)
        self.assertFalse(result.fallen)

    def test_upright_cone_ground_contact_is_bottom_edge_midpoint(self) -> None:
        cone = detection(50.0, 50.0)

        self.assertEqual(cone.ground_contact, (50.0, 55.0))

    def test_distant_parallel_rows_do_not_create_cross_row_alerts(self) -> None:
        left = [detection(0, y) for y in range(0, 90, 10)]
        right = [detection(300, y) for y in range(0, 90, 10)]
        result = analyze_cones([*left, *right])
        self.assertFalse(result.missing)
        self.assertFalse(result.displaced)
        self.assertEqual(len(result.routes), 2)
        self.assertEqual([len(route) for route in result.routes], [9, 9])
        for route in result.routes:
            centers = [result.ordered[index].center for index in route]
            self.assertTrue(all(math.dist(first, second) <= 10.01 for first, second in zip(centers, centers[1:])))

    def test_sparse_groups_cannot_chain_through_inflated_nearest_distances(self) -> None:
        centers = [
            (166.5, 634.3), (55.8, 1008.1), (25.0, 1051.7),
            (1385.2, 607.1), (1088.9, 216.2), (1151.1, 126.0),
            (1036.9, 307.2), (646.2, 944.9), (1270.9, 959.7), (1243.0, 943.7),
        ]
        result = analyze_cones([detection(x, y) for x, y in centers])
        self.assertGreater(len(result.routes), 1)
        self.assertFalse(result.missing)
        self.assertFalse(result.displaced)

    def test_parallel_rows_keep_missing_detection_inside_its_route(self) -> None:
        left = [detection(0, y) for y in (0, 10, 20, 40, 50, 60)]
        right = [detection(300, y) for y in range(0, 70, 10)]
        result = analyze_cones([*left, *right])
        self.assertEqual(len(result.routes), 2)
        self.assertEqual(len(result.missing), 1)
        self.assertAlmostEqual(result.missing[0].positions[0][0], 0.0)
        self.assertAlmostEqual(result.missing[0].positions[0][1], 30.0)

    def test_missing_gap_estimates_one_cone(self) -> None:
        result = analyze_cones([detection(x, 0) for x in (0, 10, 20, 40, 50, 60)])
        self.assertEqual(len(result.missing), 1)
        self.assertEqual(result.missing[0].estimated_count, 1)
        self.assertAlmostEqual(result.missing[0].positions[0][0], 30.0)

    def test_ground_projection_prevents_perspective_gap_false_positive(self) -> None:
        image_size = (1920, 1080)
        projector = GroundPlaneProjector(45.0, 84.0, 10.0)
        ground_points = [
            (0.0, distance) for distance in (5.0, 10.0, 15.0, 20.0, 25.0)
        ]
        image_points = [
            projector.ground_to_image(point, image_size)
            for point in ground_points
        ]
        cones = [
            ground_detection(projector, point, image_size)
            for point in ground_points
        ]

        pixel_result = analyze_cones(
            cones,
            image_size=image_size,
            route_link_ratio=5.0,
        )
        corrected_result = analyze_cones(
            cones,
            image_size=image_size,
            route_link_ratio=5.0,
            ground_projector=projector,
        )

        self.assertTrue(pixel_result.missing)
        self.assertFalse(corrected_result.missing)
        self.assertEqual(corrected_result.distance_unit, "meters")
        self.assertAlmostEqual(
            corrected_result.median_gap_distance or 0.0, 5.0, places=6
        )

    def test_ground_projection_reports_metric_shift_at_any_image_depth(self) -> None:
        image_size = (1920, 1080)
        projector = GroundPlaneProjector(45.0, 84.0, 10.0)
        ground_points = [
            (1.0 if distance == 25.0 else 0.0, distance)
            for distance in range(5, 50, 5)
        ]
        cones = [
            ground_detection(projector, point, image_size)
            for point in ground_points
        ]

        result = analyze_cones(
            cones,
            image_size=image_size,
            route_link_ratio=6.0,
            ground_projector=projector,
            offset_threshold_meters=0.5,
        )

        self.assertTrue(result.displaced)
        self.assertGreater(
            max(alert.distance_meters or 0.0 for alert in result.displaced),
            0.5,
        )

    def test_events_record_pixels_corrected_distance_and_calibration(self) -> None:
        image_size = (1920, 1080)
        projector = GroundPlaneProjector(45.0, 84.0, 10.0)
        cones = [
            ground_detection(projector, (0.0, distance), image_size)
            for distance in (5.0, 10.0, 20.0, 25.0, 30.0)
        ]
        analysis = analyze_cones(
            cones,
            image_size=image_size,
            route_link_ratio=6.0,
            ground_projector=projector,
        )

        event = _events(0, analysis, [], None, 0)

        self.assertEqual(event["distance_unit"], "meters")
        self.assertEqual(
            event["perspective_calibration"]["pitch_degrees"], 45.0
        )
        self.assertIn("gap_pixels", event["missing"][0])
        self.assertAlmostEqual(event["missing"][0]["gap_corrected"], 10.0)

    def test_shift_uses_metric_calibration(self) -> None:
        cones = [detection(x, 8 if x == 40 else 0) for x in range(0, 90, 10)]
        result = analyze_cones(cones, meters_per_pixel=0.1, offset_threshold_meters=0.5)
        shifted = [alert for alert in result.displaced if alert.distance_meters is not None]
        self.assertTrue(shifted)
        self.assertGreater(max(alert.distance_meters for alert in shifted), 0.5)

    def test_edge_clipped_horizontal_cone_is_not_classified_as_fallen(self) -> None:
        result = analyze_cones(
            [detection(3, 50, angle=0)],
            image_size=(100, 100),
            fallen_edge_margin_ratio=0.05,
        )
        self.assertFalse(result.fallen)
        self.assertFalse(result.routes)
        self.assertEqual(len(result.edge_ignored), 1)

    def test_complete_cone_with_35_degree_tilt_is_fallen(self) -> None:
        result = analyze_cones(
            [detection(50, 50, angle=55)],
            image_size=(100, 100),
        )
        self.assertEqual(len(result.fallen), 1)

    def test_temporal_route_hints_prevent_close_rows_from_merging(self) -> None:
        first_frame = [
            *[detection(0, y) for y in range(0, 90, 10)],
            *[detection(50, y) for y in range(0, 90, 10)],
        ]
        first = analyze_cones(first_frame)
        hints = [[first.ordered[index].center for index in route] for route in first.routes]
        second_frame = [
            *[detection(10, y) for y in range(0, 90, 10)],
            *[detection(35, y) for y in range(0, 90, 10)],
        ]
        self.assertEqual(len(analyze_cones(second_frame).routes), 1)
        stabilized = analyze_cones(second_frame, route_hints=hints)
        self.assertEqual(len(stabilized.routes), 2)

    def test_temporal_hint_never_merges_current_spatial_groups(self) -> None:
        left = [detection(0, y) for y in range(0, 90, 10)]
        right = [detection(300, y) for y in range(0, 90, 10)]
        one_historical_hint = [[d.center for d in [*left, *right]]]
        result = analyze_cones(
            [*left, *right], route_hints=one_historical_hint
        )
        self.assertEqual(len(result.routes), 2)

    def test_horizontal_cone_is_fallen(self) -> None:
        result = analyze_cones([detection(0, 0, angle=0)])
        self.assertEqual(len(result.fallen), 1)
        self.assertAlmostEqual(result.fallen[0].difference_degrees, 90.0)


if __name__ == "__main__":
    unittest.main()
