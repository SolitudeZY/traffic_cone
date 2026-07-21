#!/usr/bin/env python3
"""Behavior tests for concise alert transition logs."""

from __future__ import annotations

import unittest

from alert_logger import AlertJournal, format_review_alert


def event(*, missing: bool = False) -> dict:
    return {
        "stable_missing": (
            [{"stable_route_id": 4, "stable_gap_id": 12}] if missing else []
        ),
        "stable_displaced": [],
        "stable_fallen": [],
    }


class AlertJournalTests(unittest.TestCase):
    def test_persistent_alert_is_recorded_only_on_activation(self) -> None:
        journal = AlertJournal()

        first = journal.update(event(missing=True), 0.934)
        repeated = journal.update(event(missing=True), 0.968)

        self.assertEqual(len(first), 1)
        self.assertFalse(repeated)
        self.assertEqual(first[0]["alert_count"], 1)
        self.assertEqual(first[0]["time_seconds"], 0.934)
        self.assertEqual(first[0]["roi_id"], "ROI_4")
        self.assertEqual(first[0]["alert_type"], "MISSING")

    def test_cleared_alert_can_be_recorded_on_later_reactivation(self) -> None:
        journal = AlertJournal()
        journal.update(event(missing=True), 1.0)
        journal.update(event(), 1.1)

        second = journal.update(event(missing=True), 2.5)

        self.assertEqual(second[0]["alert_count"], 2)
        self.assertEqual(second[0]["time_seconds"], 2.5)

    def test_human_line_contains_only_reviewable_fields(self) -> None:
        record = AlertJournal().update(event(missing=True), 0.934)[0]

        line = format_review_alert(record)

        self.assertEqual(
            line,
            "1\ttime=0.934s\troi=ROI_4\ttype=MISSING",
        )
