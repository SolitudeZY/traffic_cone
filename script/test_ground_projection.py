#!/usr/bin/env python3
"""Regression tests for oblique-drone ground-plane projection."""

from __future__ import annotations

import math
import unittest

from ground_projection import GroundPlaneProjector


class GroundPlaneProjectorTests(unittest.TestCase):
    image_size = (1920, 1080)

    def test_ground_points_round_trip_at_45_degree_pitch(self) -> None:
        projector = GroundPlaneProjector(
            pitch_degrees=45.0,
            horizontal_fov_degrees=84.0,
            altitude_meters=10.0,
        )

        for ground_point in ((0.0, 5.0), (2.0, 15.0), (-3.0, 25.0)):
            image_point = projector.ground_to_image(
                ground_point, self.image_size
            )
            recovered = projector.image_to_ground(
                image_point, self.image_size
            )
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertAlmostEqual(recovered[0], ground_point[0], places=6)
            self.assertAlmostEqual(recovered[1], ground_point[1], places=6)

    def test_equal_ground_intervals_are_equal_after_projection(self) -> None:
        projector = GroundPlaneProjector(45.0, 84.0, 10.0)
        ground = [(0.0, distance) for distance in (5, 10, 15, 20, 25)]
        image = [
            projector.ground_to_image(point, self.image_size)
            for point in ground
        ]
        pixel_gaps = [math.dist(first, second) for first, second in zip(image, image[1:])]
        corrected = [
            projector.distance(first, second, self.image_size)
            for first, second in zip(image, image[1:])
        ]

        self.assertGreater(max(pixel_gaps) / min(pixel_gaps), 3.0)
        for gap in corrected:
            self.assertIsNotNone(gap)
            self.assertAlmostEqual(gap or 0.0, 5.0, places=6)

    def test_altitude_scales_metric_distance_but_not_image_geometry(self) -> None:
        normalized = GroundPlaneProjector(45.0, 84.0)
        at_ten_meters = GroundPlaneProjector(45.0, 84.0, 10.0)
        at_twenty_meters = GroundPlaneProjector(45.0, 84.0, 20.0)
        first = (960.0, 300.0)
        second = (960.0, 500.0)

        ratio_distance = normalized.distance(first, second, self.image_size)
        ten_meter_distance = at_ten_meters.distance(first, second, self.image_size)
        twenty_meter_distance = at_twenty_meters.distance(first, second, self.image_size)

        self.assertEqual(normalized.distance_unit, "camera_heights")
        self.assertEqual(at_ten_meters.distance_unit, "meters")
        self.assertAlmostEqual((ten_meter_distance or 0.0) / (ratio_distance or 1.0), 10.0)
        self.assertAlmostEqual((twenty_meter_distance or 0.0) / (ten_meter_distance or 1.0), 2.0)


if __name__ == "__main__":
    unittest.main()
