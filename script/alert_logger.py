"""Concise transition-only logs for human-reviewable business alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AlertKey = tuple[str, int, int | None]


def _current_alerts(event: dict[str, Any]) -> dict[AlertKey, dict[str, Any]]:
    current: dict[AlertKey, dict[str, Any]] = {}
    fields = (
        ("stable_missing", "MISSING", "stable_gap_id"),
        ("stable_displaced", "SHIFT", None),
        ("stable_fallen", "FALLEN", None),
    )
    for field, alert_type, identity_field in fields:
        for alert in event.get(field, ()):
            stable_route_id = alert.get("stable_route_id")
            if stable_route_id is None:
                continue
            detail_id = (
                alert.get(identity_field) if identity_field is not None else None
            )
            current[(alert_type, int(stable_route_id), detail_id)] = alert
    return current


@dataclass
class AlertJournal:
    """Convert per-frame active alert sets into activation records."""

    _active: set[AlertKey] = field(default_factory=set, init=False)
    _count: int = field(default=0, init=False)

    def update(
        self, event: dict[str, Any], time_seconds: float
    ) -> list[dict[str, Any]]:
        current = _current_alerts(event)
        started = sorted(set(current) - self._active)
        records: list[dict[str, Any]] = []
        for alert_type, stable_route_id, detail_id in started:
            self._count += 1
            record: dict[str, Any] = {
                "alert_count": self._count,
                "time_seconds": round(float(time_seconds), 3),
                "roi_id": f"ROI_{stable_route_id}",
                "alert_type": alert_type,
            }
            if alert_type == "MISSING" and detail_id is not None:
                record["stable_gap_id"] = detail_id
            records.append(record)
        self._active = set(current)
        return records


def format_review_alert(record: dict[str, Any]) -> str:
    return (
        f"{record['alert_count']}\t"
        f"time={record['time_seconds']:.3f}s\t"
        f"roi={record['roi_id']}\t"
        f"type={record['alert_type']}"
    )
