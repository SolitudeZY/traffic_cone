#!/usr/bin/env python3
"""Behavior tests for per-route temporal gap thresholds."""

from __future__ import annotations

import unittest

from gap_tracker import GapThresholdTracker
from ground_projection import GroundPlaneProjector
from infer_obb import _events
from route_tracker import RouteTracker, observations_from_analysis
from traffic_rules import AnalysisResult, ConeDetection, MissingAlert, analyze_cones


def detection(x: float, y: float = 100.0) -> ConeDetection:
    return ConeDetection(
        (
            (x - 3.0, y - 8.0),
            (x + 3.0, y - 8.0),
            (x + 3.0, y + 8.0),
            (x - 3.0, y + 8.0),
        ),
        0.9,
    )


def analysis_at(
    positions: list[float],
    *,
    missing_after: int | None = None,
    reference_gap: float | None = None,
) -> AnalysisResult:
    detections = [detection(position) for position in positions]
    missing = []
    if missing_after is not None and reference_gap is not None:
        first = positions[missing_after]
        second = positions[missing_after + 1]
        missing = [
            MissingAlert(
                after_index=missing_after,
                gap_pixels=second - first,
                reference_gap_pixels=reference_gap,
                estimated_count=max(1, round((second - first) / reference_gap) - 1),
                positions=(((first + second) / 2.0, 100.0),),
            )
        ]
    return AnalysisResult(
        ordered=detections,
        routes=[list(range(len(detections)))],
        missing=missing,
    )


class GapThresholdTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routes = RouteTracker(confirm_frames=1, recovery_frames=1)
        self.gaps = GapThresholdTracker(
            history_frames=5,
            min_history_frames=3,
            close_ratio=0.55,
            max_drop_ratio=0.05,
            vote_window=1,
            confirm_votes=1,
            clear_frames=1,
            membership_cooldown_frames=3,
            min_alert_cones=3,
        )

    def update(
        self,
        positions: list[float],
        *,
        missing_after: int | None = None,
        reference_gap: float | None = None,
    ):
        analysis = analysis_at(
            positions,
            missing_after=missing_after,
            reference_gap=reference_gap,
        )
        tracking = self.routes.update(
            observations_from_analysis(analysis), (1920, 1080)
        )
        return self.gaps.update(analysis, tracking, gap_ratio=1.8)

    def establish(self, gap: float = 75.0) -> None:
        for _ in range(3):
            self.update(
                [100.0, 100.0 + gap, 100.0 + gap * 3, 100.0 + gap * 4],
                missing_after=1,
                reference_gap=gap,
            )

    def test_close_pair_does_not_turn_normal_gap_into_missing_alert(self) -> None:
        self.establish(75.0)

        result = self.update(
            [100.0, 133.0, 211.0], missing_after=1, reference_gap=33.0
        )

        self.assertFalse(result.missing)
        self.assertEqual(len(result.routes), 1)
        self.assertAlmostEqual(result.routes[0].excluded_close_gaps[0], 33.0)
        self.assertGreater(result.routes[0].nominal_gap_pixels or 0.0, 70.0)

    def test_true_multiple_of_historical_gap_remains_missing_alert(self) -> None:
        self.establish(75.0)

        result = self.update(
            [100.0, 175.0, 325.0], missing_after=1, reference_gap=75.0
        )

        self.assertEqual(len(result.missing), 1)
        self.assertEqual(result.missing[0].estimated_count, 1)
        self.assertAlmostEqual(result.missing[0].reference_gap_pixels, 75.0)

    def test_repeated_close_frames_do_not_collapse_nominal_gap(self) -> None:
        self.establish(75.0)

        result = None
        for _ in range(6):
            result = self.update(
                [100.0, 133.0, 211.0], missing_after=1, reference_gap=33.0
            )

        assert result is not None
        route = result.routes[0]
        self.assertGreater(route.nominal_gap_pixels or 0.0, 70.0)
        self.assertAlmostEqual(route.frame_average_pixels or 0.0, 33.0)
        self.assertGreaterEqual(route.history_size, 3)

    def test_two_gaps_without_history_are_insufficient_for_alert(self) -> None:
        result = self.update(
            [100.0, 133.0, 211.0], missing_after=1, reference_gap=33.0
        )

        self.assertFalse(result.missing)
        self.assertEqual(result.routes[0].threshold_source, "warming_up")
        self.assertIsNone(result.routes[0].nominal_gap_pixels)

    def test_repeated_short_route_builds_history_before_alerting(self) -> None:
        first = self.update(
            [100.0, 157.0, 281.0], missing_after=1, reference_gap=57.0
        )
        second = self.update(
            [100.0, 157.0, 281.0], missing_after=1, reference_gap=57.0
        )
        third = self.update(
            [100.0, 157.0, 281.0], missing_after=1, reference_gap=57.0
        )

        self.assertFalse(first.missing)
        self.assertFalse(second.missing)
        self.assertEqual(len(third.missing), 1)
        self.assertAlmostEqual(third.routes[0].nominal_gap_pixels or 0.0, 57.0)

    def test_persistent_scale_change_can_lower_history_instead_of_locking_maximum(self) -> None:
        self.establish(100.0)

        result = None
        for _ in range(8):
            result = self.update(
                [100.0, 190.0, 370.0, 460.0],
                missing_after=1,
                reference_gap=90.0,
            )

        assert result is not None
        self.assertLess(result.routes[0].nominal_gap_pixels or 100.0, 95.0)
        self.assertGreater(result.routes[0].nominal_gap_pixels or 0.0, 85.0)

    def test_events_keep_raw_alert_and_publish_gap_history_diagnostics(self) -> None:
        self.establish(75.0)
        analysis = analysis_at(
            [100.0, 133.0, 211.0], missing_after=1, reference_gap=33.0
        )
        tracking = self.routes.update(
            observations_from_analysis(analysis), (1920, 1080)
        )
        gap_tracking = self.gaps.update(analysis, tracking, gap_ratio=1.8)

        event = _events(10, analysis, [], None, 0, tracking, gap_tracking)

        self.assertEqual(len(event["missing"]), 1)
        self.assertFalse(event["stable_missing"])
        self.assertGreater(event["gap_thresholds"][0]["nominal_gap_pixels"], 70.0)
        self.assertEqual(event["gap_thresholds"][0]["excluded_close_gaps"], (33.0,))


class StableGapAlertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routes = RouteTracker(confirm_frames=1, recovery_frames=1)
        self.gaps = GapThresholdTracker(
            history_frames=5,
            min_history_frames=3,
            close_ratio=0.55,
            max_drop_ratio=0.05,
            vote_window=5,
            confirm_votes=3,
            clear_frames=3,
            membership_cooldown_frames=3,
            min_alert_cones=3,
        )

    def update(
        self,
        positions: list[float],
        *,
        candidate_after: int | None = None,
        reference_gap: float = 75.0,
    ):
        analysis = analysis_at(
            positions,
            missing_after=candidate_after,
            reference_gap=(reference_gap if candidate_after is not None else None),
        )
        tracking = self.routes.update(
            observations_from_analysis(analysis), (1920, 1080)
        )
        return self.gaps.update(analysis, tracking, gap_ratio=1.8)

    def test_gap_ids_survive_frame_to_frame_camera_motion(self) -> None:
        first = self.update([100.0, 175.0, 325.0, 400.0])
        second = self.update([112.0, 187.0, 337.0, 412.0])

        self.assertEqual(
            [gap.stable_id for gap in first.gaps],
            [gap.stable_id for gap in second.gaps],
        )

    def test_missing_requires_three_positive_votes_in_latest_five(self) -> None:
        positions = [100.0, 175.0, 325.0, 400.0]
        first = self.update(positions, candidate_after=1)
        second = self.update(positions, candidate_after=1)
        normal = self.update(positions)
        confirmed = self.update(positions, candidate_after=1)

        self.assertFalse(first.missing)
        self.assertFalse(second.missing)
        self.assertFalse(normal.missing)
        self.assertEqual(len(confirmed.missing), 1)
        active_gap = next(gap for gap in confirmed.gaps if gap.active_missing)
        self.assertEqual(active_gap.positive_votes, 3)

    def test_active_missing_clears_after_three_consecutive_normal_frames(self) -> None:
        positions = [100.0, 175.0, 325.0, 400.0]
        for _ in range(3):
            active = self.update(positions, candidate_after=1)
        self.assertEqual(len(active.missing), 1)

        first = self.update(positions)
        second = self.update(positions)
        third = self.update(positions)

        self.assertEqual(len(first.missing), 1)
        self.assertEqual(len(second.missing), 1)
        self.assertFalse(third.missing)

    def test_member_removal_cools_new_adjacent_gap_before_voting(self) -> None:
        original = [100.0, 175.0, 325.0, 400.0]
        for _ in range(3):
            self.update(original, candidate_after=1)

        changed = [100.0, 325.0, 400.0]
        cooled = [
            self.update(changed, candidate_after=0)
            for _ in range(3)
        ]
        votes = [
            self.update(changed, candidate_after=0)
            for _ in range(3)
        ]

        self.assertTrue(all(not result.missing for result in cooled))
        self.assertTrue(all(not result.missing for result in votes[:2]))
        self.assertEqual(len(votes[2].missing), 1)

    def test_two_cone_route_never_produces_business_missing(self) -> None:
        result = None
        for _ in range(6):
            result = self.update(
                [100.0, 250.0], candidate_after=0, reference_gap=75.0
            )

        assert result is not None
        self.assertFalse(result.missing)
        self.assertTrue(result.gaps)
        self.assertFalse(result.gaps[0].business_eligible)

    def test_events_attach_stable_gap_id_to_stable_missing(self) -> None:
        positions = [100.0, 175.0, 325.0, 400.0]
        for _ in range(3):
            analysis = analysis_at(
                positions, missing_after=1, reference_gap=75.0
            )
            tracking = self.routes.update(
                observations_from_analysis(analysis), (1920, 1080)
            )
            gap_tracking = self.gaps.update(
                analysis, tracking, gap_ratio=1.8
            )

        event = _events(10, analysis, [], None, 0, tracking, gap_tracking)

        self.assertEqual(len(event["stable_missing"]), 1)
        self.assertIsInstance(
            event["stable_missing"][0]["stable_gap_id"], int
        )


class PerspectiveGapTrackerTests(unittest.TestCase):
    def test_stable_gap_voting_uses_corrected_ground_distance(self) -> None:
        image_size = (1920, 1080)
        projector = GroundPlaneProjector(45.0, 84.0, 10.0)
        routes = RouteTracker(confirm_frames=1, recovery_frames=1)
        gaps = GapThresholdTracker(
            history_frames=3,
            min_history_frames=3,
            vote_window=1,
            confirm_votes=1,
            clear_frames=1,
            membership_cooldown_frames=3,
            min_alert_cones=3,
        )
        result = None
        for _ in range(3):
            cones = [
                detection(
                    projector.ground_to_image((0.0, distance), image_size)[0],
                    projector.ground_to_image((0.0, distance), image_size)[1]
                    - 8.0,
                )
                for distance in (5.0, 10.0, 20.0, 25.0, 30.0)
            ]
            analysis = analyze_cones(
                cones,
                image_size=image_size,
                route_link_ratio=6.0,
                ground_projector=projector,
            )
            tracking = routes.update(
                observations_from_analysis(analysis), image_size
            )
            result = gaps.update(analysis, tracking, gap_ratio=1.8)

        assert result is not None
        self.assertEqual(len(result.missing), 1)
        self.assertAlmostEqual(result.missing[0].gap_corrected or 0.0, 10.0)
        self.assertAlmostEqual(
            result.missing[0].reference_gap_corrected or 0.0, 5.0
        )
        self.assertEqual(result.routes[0].distance_unit, "meters")


if __name__ == "__main__":
    unittest.main()
