#!/usr/bin/env python3
"""Behavior tests for persistent route tracking."""

from __future__ import annotations

import unittest

from infer_obb import _events
from route_tracker import RouteObservation, RouteTracker
from traffic_rules import (
    AnalysisResult,
    ConeDetection,
    DisplacementAlert,
    MissingAlert,
)


def observation(
    route_index: int,
    x: float,
    *,
    y_start: float = 100.0,
    count: int = 4,
    gap: float = 40.0,
    half_width: float = 8.0,
) -> RouteObservation:
    centers = tuple((x, y_start + index * gap) for index in range(count))
    return RouteObservation(
        route_index=route_index,
        detection_indices=tuple(range(route_index * 10, route_index * 10 + count)),
        centers=centers,
        bounds=(x - half_width, y_start - 12.0, x + half_width, y_start + (count - 1) * gap + 12.0),
    )


class RouteTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = RouteTracker(
            confirm_frames=2,
            recovery_frames=2,
            max_missed_frames=5,
            base_smoothing=0.35,
            max_smoothing=0.75,
        )

    def test_new_route_requires_two_hits_before_confirmation(self) -> None:
        first = self.tracker.update([observation(0, 200)], (1280, 720))
        self.assertEqual(first.tracks[0].state, "tentative")
        self.assertFalse(first.tracks[0].alert_ready)

        second = self.tracker.update([observation(0, 204)], (1280, 720))
        self.assertEqual(second.tracks[0].stable_id, first.tracks[0].stable_id)
        self.assertEqual(second.tracks[0].state, "confirmed")
        self.assertTrue(second.tracks[0].alert_ready)

    def test_one_route_is_held_while_another_route_remains_visible(self) -> None:
        self.tracker.update([observation(0, 200), observation(1, 600)], (1280, 720))
        confirmed = self.tracker.update(
            [observation(0, 204), observation(1, 604)], (1280, 720)
        )
        ids = {track.current_route_index: track.stable_id for track in confirmed.tracks}

        held = self.tracker.update([observation(0, 208)], (1280, 720))
        by_id = {track.stable_id: track for track in held.tracks}
        self.assertEqual(by_id[ids[1]].state, "held")
        self.assertEqual(by_id[ids[1]].missed_frames, 1)
        self.assertFalse(by_id[ids[1]].alert_ready)
        self.assertEqual(by_id[ids[0]].state, "confirmed")

    def test_held_route_recovers_same_id_after_two_hits(self) -> None:
        self.tracker.update([observation(0, 300)], (1280, 720))
        confirmed = self.tracker.update([observation(0, 304)], (1280, 720))
        stable_id = confirmed.tracks[0].stable_id
        self.tracker.update([], (1280, 720))
        self.tracker.update([], (1280, 720))

        first_return = self.tracker.update([observation(0, 316)], (1280, 720))
        self.assertEqual(first_return.tracks[0].stable_id, stable_id)
        self.assertEqual(first_return.tracks[0].state, "held")
        self.assertFalse(first_return.tracks[0].alert_ready)

        second_return = self.tracker.update([observation(0, 320)], (1280, 720))
        self.assertEqual(second_return.tracks[0].stable_id, stable_id)
        self.assertEqual(second_return.tracks[0].state, "confirmed")
        self.assertTrue(second_return.tracks[0].alert_ready)

    def test_route_expires_after_sixth_missed_frame(self) -> None:
        self.tracker.update([observation(0, 300)], (1280, 720))
        confirmed = self.tracker.update([observation(0, 304)], (1280, 720))
        stable_id = confirmed.tracks[0].stable_id
        for missed in range(1, 6):
            result = self.tracker.update([], (1280, 720))
            self.assertIn(stable_id, {track.stable_id for track in result.tracks})
            self.assertEqual(result.tracks[0].missed_frames, missed)
        expired = self.tracker.update([], (1280, 720))
        self.assertNotIn(stable_id, {track.stable_id for track in expired.tracks})

    def test_candidate_order_change_does_not_swap_parallel_route_ids(self) -> None:
        self.tracker.update([observation(0, 200), observation(1, 600)], (1280, 720))
        confirmed = self.tracker.update(
            [observation(0, 204), observation(1, 604)], (1280, 720)
        )
        left_id = min(confirmed.tracks, key=lambda track: track.centroid[0]).stable_id
        right_id = max(confirmed.tracks, key=lambda track: track.centroid[0]).stable_id

        reordered = self.tracker.update(
            [observation(0, 608), observation(1, 208)], (1280, 720)
        )
        left = min(reordered.tracks, key=lambda track: track.centroid[0])
        right = max(reordered.tracks, key=lambda track: track.centroid[0])
        self.assertEqual(left.stable_id, left_id)
        self.assertEqual(right.stable_id, right_id)

    def test_smoothing_reduces_static_jitter_but_follows_camera_motion(self) -> None:
        self.tracker.update([observation(0, 300)], (1280, 720))
        confirmed = self.tracker.update([observation(0, 304)], (1280, 720))
        before_x = confirmed.tracks[0].centroid[0]

        jittered = self.tracker.update([observation(0, 294)], (1280, 720))
        raw_x = jittered.tracks[0].raw_centroid[0]
        smooth_x = jittered.tracks[0].centroid[0]
        self.assertLess(abs(smooth_x - before_x), abs(raw_x - before_x))

        moving = jittered
        for x in (320, 345, 370, 395):
            moving = self.tracker.update([observation(0, x)], (1280, 720))
        self.assertGreater(moving.tracks[0].centroid[0], 370.0)
        self.assertLess(abs(moving.tracks[0].centroid[0] - 395.0), 25.0)

    def test_held_route_contributes_predicted_hints(self) -> None:
        self.tracker.update([observation(0, 300)], (1280, 720))
        self.tracker.update([observation(0, 305)], (1280, 720))
        self.tracker.update([], (1280, 720))
        hints = self.tracker.hints()
        self.assertEqual(len(hints), 1)
        self.assertEqual(len(hints[0]), 4)
        self.assertGreater(hints[0][0][0], 305.0)

    def test_events_keep_legacy_routes_and_suppress_recovering_alerts(self) -> None:
        tracker = RouteTracker(confirm_frames=1, recovery_frames=2)
        route = observation(0, 300)
        tracker.update([route], (1280, 720))
        tracker.update([], (1280, 720))
        recovering = tracker.update([observation(0, 308)], (1280, 720))
        detections = [
            ConeDetection(
                (
                    (center[0] - 5, center[1] - 10),
                    (center[0] + 5, center[1] - 10),
                    (center[0] + 5, center[1] + 10),
                    (center[0] - 5, center[1] + 10),
                ),
                0.9,
            )
            for center in observation(0, 308).centers
        ]
        analysis = AnalysisResult(
            ordered=detections,
            routes=[list(range(len(detections)))],
            missing=[
                MissingAlert(
                    after_index=1,
                    gap_pixels=80.0,
                    reference_gap_pixels=40.0,
                    estimated_count=1,
                    positions=((308.0, 160.0),),
                )
            ],
        )
        event = _events(10, analysis, [], None, 0, recovering)
        self.assertIn("adaptive_routes", event)
        self.assertEqual(event["adaptive_routes"][0]["stable_route_id"], 1)
        self.assertEqual(event["tracked_routes"][0]["state"], "held")
        self.assertFalse(event["tracked_routes"][0]["alert_ready"])
        self.assertEqual(len(event["missing"]), 1)
        self.assertFalse(event["stable_missing"])

    def test_two_cone_route_keeps_raw_shift_but_suppresses_business_alert(self) -> None:
        tracker = RouteTracker(confirm_frames=1, recovery_frames=1)
        route = observation(0, 300, count=2)
        tracking = tracker.update([route], (1280, 720))
        detections = [
            ConeDetection(
                (
                    (center[0] - 5, center[1] - 10),
                    (center[0] + 5, center[1] - 10),
                    (center[0] + 5, center[1] + 10),
                    (center[0] - 5, center[1] + 10),
                ),
                0.9,
            )
            for center in route.centers
        ]
        analysis = AnalysisResult(
            ordered=detections,
            routes=[[0, 1]],
            displaced=[DisplacementAlert(0, 20.0, None)],
        )

        event = _events(10, analysis, [], None, 0, tracking)

        self.assertEqual(len(event["displaced"]), 1)
        self.assertFalse(event["stable_displaced"])


if __name__ == "__main__":
    unittest.main()
