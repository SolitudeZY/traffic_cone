"""Temporal per-route gap identity, thresholds, and alert voting."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from route_tracker import RouteTrackingResult, TrackedRoute
from traffic_rules import AnalysisResult, MissingAlert


Point = tuple[float, float]
GapKey = tuple[int, int]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _gap_key(first_id: int, second_id: int) -> GapKey:
    return tuple(sorted((first_id, second_id)))


def _route_gap_items(
    analysis: AnalysisResult, route: Sequence[int]
) -> list[tuple[int, int, float, float]]:
    items: list[tuple[int, int, float, float]] = []
    for first_index, second_index in zip(route, route[1:]):
        pixel_gap = _distance(
            analysis.ordered[first_index].center,
            analysis.ordered[second_index].center,
        )
        first_ground = (
            analysis.ground_centers[first_index]
            if first_index < len(analysis.ground_centers)
            else None
        )
        second_ground = (
            analysis.ground_centers[second_index]
            if second_index < len(analysis.ground_centers)
            else None
        )
        corrected_gap = (
            _distance(first_ground, second_ground)
            if first_ground is not None and second_ground is not None
            else pixel_gap
        )
        items.append((first_index, second_index, pixel_gap, corrected_gap))
    return items


@dataclass(frozen=True)
class GapRouteThreshold:
    stable_id: int
    current_route_index: int
    nominal_gap_pixels: float | None
    frame_average_pixels: float | None
    average_of_averages_pixels: float | None
    history_size: int
    confidence: float
    excluded_close_gaps: tuple[float, ...]
    threshold_source: str
    distance_unit: str = "pixels"


@dataclass(frozen=True)
class TrackedGap:
    stable_id: int
    stable_route_id: int
    after_index: int
    endpoint_ids: GapKey
    candidate_missing: bool
    active_missing: bool
    positive_votes: int
    vote_count: int
    normal_streak: int
    cooldown_remaining: int
    business_eligible: bool


@dataclass(frozen=True)
class GapTrackingResult:
    missing: tuple[MissingAlert, ...]
    routes: tuple[GapRouteThreshold, ...]
    gaps: tuple[TrackedGap, ...] = ()
    missing_gap_ids: tuple[int, ...] = ()


@dataclass
class _StableGapState:
    stable_id: int
    endpoint_ids: GapKey
    votes: deque[bool]
    active: bool = False
    normal_streak: int = 0
    cooldown_remaining: int = 0
    last_alert: MissingAlert | None = None


@dataclass
class _GapState:
    history: deque[float]
    nominal_gap: float | None = None
    cone_ids: tuple[int, ...] = ()
    centers: tuple[Point, ...] = ()
    gaps: dict[GapKey, _StableGapState] = field(default_factory=dict)


@dataclass
class GapThresholdTracker:
    history_frames: int = 30
    min_history_frames: int = 3
    close_ratio: float = 0.55
    max_drop_ratio: float = 0.05
    max_rise_ratio: float = 0.10
    minimum_gap_pixels: float = 0.0
    vote_window: int = 5
    confirm_votes: int = 3
    clear_frames: int = 3
    membership_cooldown_frames: int = 5
    min_alert_cones: int = 3
    _states: dict[int, _GapState] = field(default_factory=dict, init=False)
    _next_cone_id: int = field(default=1, init=False)
    _next_gap_id: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise ValueError("history_frames must be at least 1")
        if not 1 <= self.min_history_frames <= self.history_frames:
            raise ValueError("min_history_frames must be within history_frames")
        if not 0.0 < self.close_ratio < 1.0:
            raise ValueError("close_ratio must be in (0, 1)")
        if not 0.0 <= self.max_drop_ratio < 1.0:
            raise ValueError("max_drop_ratio must be in [0, 1)")
        if self.max_rise_ratio < 0.0:
            raise ValueError("max_rise_ratio must not be negative")
        if self.minimum_gap_pixels < 0.0:
            raise ValueError("minimum_gap_pixels must not be negative")
        if self.vote_window < 1:
            raise ValueError("vote_window must be at least 1")
        if not 1 <= self.confirm_votes <= self.vote_window:
            raise ValueError("confirm_votes must be within vote_window")
        if self.clear_frames < 1:
            raise ValueError("clear_frames must be at least 1")
        if not 3 <= self.membership_cooldown_frames <= 5:
            raise ValueError("membership_cooldown_frames must be between 3 and 5")
        if self.min_alert_cones < 3:
            raise ValueError("min_alert_cones must be at least 3")

    def _state(self, stable_id: int) -> _GapState:
        state = self._states.get(stable_id)
        if state is None:
            state = _GapState(deque(maxlen=self.history_frames))
            self._states[stable_id] = state
        return state

    def _new_cone_id(self) -> int:
        stable_id = self._next_cone_id
        self._next_cone_id += 1
        return stable_id

    def _new_gap_state(self, key: GapKey) -> _StableGapState:
        state = _StableGapState(
            stable_id=self._next_gap_id,
            endpoint_ids=key,
            votes=deque(maxlen=self.vote_window),
        )
        self._next_gap_id += 1
        return state

    def _update_state(
        self,
        state: _GapState,
        gaps: Sequence[float],
        raw_references: Sequence[float],
    ) -> tuple[float | None, tuple[float, ...]]:
        frame_average = (
            statistics.fmean(raw_references) if raw_references else None
        )
        excluded_close = (
            tuple(gap for gap in gaps if gap < state.nominal_gap * self.close_ratio)
            if state.nominal_gap is not None
            else ()
        )
        if frame_average is None:
            return None, excluded_close
        if (
            state.nominal_gap is not None
            and frame_average < state.nominal_gap * self.close_ratio
        ):
            return frame_average, excluded_close

        state.history.append(frame_average)
        if state.nominal_gap is None and len(state.history) < self.min_history_frames:
            return frame_average, excluded_close
        target = statistics.median(state.history)
        if state.nominal_gap is None:
            state.nominal_gap = target
        else:
            lower = state.nominal_gap * (1.0 - self.max_drop_ratio)
            upper = state.nominal_gap * (1.0 + self.max_rise_ratio)
            state.nominal_gap = min(upper, max(lower, target))
        return frame_average, excluded_close

    def _match_cones(
        self,
        state: _GapState,
        centers: Sequence[Point],
        track: TrackedRoute,
    ) -> tuple[tuple[int, ...], set[GapKey]]:
        if not state.cone_ids:
            return tuple(self._new_cone_id() for _ in centers), set()

        predicted = [
            (center[0] + track.velocity[0], center[1] + track.velocity[1])
            for center in state.centers
        ]
        maximum_distance = max(20.0, track.spacing * 0.75)
        candidates = sorted(
            (
                _distance(previous, current),
                previous_index,
                current_index,
            )
            for previous_index, previous in enumerate(predicted)
            for current_index, current in enumerate(centers)
        )
        previous_matches: set[int] = set()
        current_matches: set[int] = set()
        current_ids: list[int | None] = [None] * len(centers)
        for distance, previous_index, current_index in candidates:
            if distance > maximum_distance:
                break
            if previous_index in previous_matches or current_index in current_matches:
                continue
            previous_matches.add(previous_index)
            current_matches.add(current_index)
            current_ids[current_index] = state.cone_ids[previous_index]

        for current_index, stable_id in enumerate(current_ids):
            if stable_id is None:
                current_ids[current_index] = self._new_cone_id()

        resolved_ids = tuple(int(stable_id) for stable_id in current_ids)
        added_ids = set(resolved_ids) - set(state.cone_ids)
        removed_ids = set(state.cone_ids) - set(resolved_ids)
        affected = {
            _gap_key(first_id, second_id)
            for first_id, second_id in zip(resolved_ids, resolved_ids[1:])
            if first_id in added_ids or second_id in added_ids
        }
        current_positions = {
            stable_id: index for index, stable_id in enumerate(resolved_ids)
        }
        for removed_id in removed_ids:
            removed_index = state.cone_ids.index(removed_id)
            left = next(
                (
                    state.cone_ids[index]
                    for index in range(removed_index - 1, -1, -1)
                    if state.cone_ids[index] in current_positions
                ),
                None,
            )
            right = next(
                (
                    state.cone_ids[index]
                    for index in range(removed_index + 1, len(state.cone_ids))
                    if state.cone_ids[index] in current_positions
                ),
                None,
            )
            if (
                left is not None
                and right is not None
                and abs(current_positions[left] - current_positions[right]) == 1
            ):
                affected.add(_gap_key(left, right))
        return resolved_ids, affected

    def _candidate_alert(
        self,
        analysis: AnalysisResult,
        first_index: int,
        second_index: int,
        gap_pixels: float,
        gap_distance: float,
        raw_alert: MissingAlert | None,
        nominal_gap: float,
        gap_ratio: float,
    ) -> MissingAlert | None:
        if raw_alert is None:
            return None
        raw_reference = (
            raw_alert.reference_gap_corrected
            if raw_alert.reference_gap_corrected is not None
            else raw_alert.reference_gap_pixels
        )
        reference = max(
            nominal_gap,
            raw_reference,
            self.minimum_gap_pixels if analysis.distance_unit == "pixels" else 0.0,
        )
        if gap_distance <= reference * gap_ratio:
            return None
        return self._alert_for_gap(
            analysis,
            first_index,
            second_index,
            gap_pixels,
            gap_distance,
            reference,
            raw_alert.reference_gap_pixels,
            raw_alert.positions,
        )

    @staticmethod
    def _alert_for_gap(
        analysis: AnalysisResult,
        first_index: int,
        second_index: int,
        gap_pixels: float,
        gap_distance: float,
        reference: float,
        reference_pixels: float,
        positions: tuple[Point, ...] | None = None,
    ) -> MissingAlert:
        estimated_count = max(1, round(gap_distance / reference) - 1)
        first = analysis.ordered[first_index].center
        second = analysis.ordered[second_index].center
        if positions is None or len(positions) != estimated_count:
            positions = tuple(
                (
                    first[0]
                    + (second[0] - first[0]) * step / (estimated_count + 1),
                    first[1]
                    + (second[1] - first[1]) * step / (estimated_count + 1),
                )
                for step in range(1, estimated_count + 1)
            )
        return MissingAlert(
            first_index,
            gap_pixels,
            reference_pixels,
            estimated_count,
            positions,
            gap_corrected=(
                gap_distance if analysis.distance_unit != "pixels" else None
            ),
            reference_gap_corrected=(
                reference if analysis.distance_unit != "pixels" else None
            ),
            distance_unit=analysis.distance_unit,
        )

    def _vote(
        self,
        gap: _StableGapState,
        candidate: MissingAlert | None,
        *,
        allowed: bool,
        can_activate: bool,
    ) -> None:
        if not allowed:
            gap.votes.clear()
            gap.active = False
            gap.normal_streak = 0
            gap.last_alert = None
            return
        if gap.cooldown_remaining > 0:
            gap.cooldown_remaining -= 1
            gap.votes.clear()
            gap.active = False
            gap.normal_streak = 0
            gap.last_alert = None
            return

        positive = candidate is not None
        gap.votes.append(positive)
        if positive:
            gap.last_alert = candidate
            gap.normal_streak = 0
        elif gap.active:
            gap.normal_streak += 1

        if (
            not gap.active
            and can_activate
            and sum(gap.votes) >= self.confirm_votes
        ):
            gap.active = True
            gap.normal_streak = 0
        elif gap.active and gap.normal_streak >= self.clear_frames:
            gap.active = False
            gap.votes.clear()
            gap.last_alert = None
            gap.normal_streak = 0

    def update(
        self,
        analysis: AnalysisResult,
        tracking: RouteTrackingResult,
        *,
        gap_ratio: float,
    ) -> GapTrackingResult:
        if gap_ratio <= 1.0:
            raise ValueError("gap_ratio must be greater than 1")
        active_ids = {track.stable_id for track in tracking.tracks}
        self._states = {
            stable_id: state
            for stable_id, state in self._states.items()
            if stable_id in active_ids
        }
        tracks_by_route = {
            track.current_route_index: track
            for track in tracking.tracks
            if track.current_route_index is not None
        }
        alerts: list[MissingAlert] = []
        alert_gap_ids: list[int] = []
        diagnostics: list[GapRouteThreshold] = []
        gap_diagnostics: list[TrackedGap] = []

        for route_index, route in enumerate(analysis.routes):
            track = tracks_by_route.get(route_index)
            if track is None:
                continue
            gap_items = _route_gap_items(analysis, route)
            state = self._state(track.stable_id)
            raw_by_after = {
                alert.after_index: alert
                for alert in analysis.missing
                if alert.after_index in route
            }
            frame_average, excluded_close = self._update_state(
                state,
                [item[3] for item in gap_items],
                [
                    alert.reference_gap_corrected
                    if alert.reference_gap_corrected is not None
                    else alert.reference_gap_pixels
                    for alert in raw_by_after.values()
                ],
            )
            enough_history = len(state.history) >= self.min_history_frames
            if state.nominal_gap is None and state.history:
                source = "warming_up"
            elif state.nominal_gap is None:
                source = "insufficient_history"
            elif enough_history:
                source = "history"
            else:
                source = "warming_up"

            centers = tuple(analysis.ordered[index].center for index in route)
            cone_ids, affected_keys = self._match_cones(state, centers, track)
            current_keys = {
                _gap_key(first_id, second_id)
                for first_id, second_id in zip(cone_ids, cone_ids[1:])
            }
            for key in affected_keys:
                gap_state = state.gaps.get(key)
                if gap_state is None:
                    gap_state = self._new_gap_state(key)
                    state.gaps[key] = gap_state
                gap_state.cooldown_remaining = self.membership_cooldown_frames
                gap_state.votes.clear()
                gap_state.active = False
                gap_state.normal_streak = 0
                gap_state.last_alert = None

            eligible = len(route) >= self.min_alert_cones
            for gap_index, (
                first_index,
                second_index,
                gap_pixels,
                gap_distance,
            ) in enumerate(gap_items):
                key = _gap_key(cone_ids[gap_index], cone_ids[gap_index + 1])
                gap_state = state.gaps.get(key)
                if gap_state is None:
                    gap_state = self._new_gap_state(key)
                    state.gaps[key] = gap_state
                raw_alert = raw_by_after.get(first_index)
                provisional_reference = (
                    state.nominal_gap
                    or (
                        raw_alert.reference_gap_corrected
                        if raw_alert is not None
                        and raw_alert.reference_gap_corrected is not None
                        else raw_alert.reference_gap_pixels
                        if raw_alert is not None
                        else None
                    )
                )
                candidate = (
                    self._candidate_alert(
                        analysis,
                        first_index,
                        second_index,
                        gap_pixels,
                        gap_distance,
                        raw_alert,
                        provisional_reference,
                        gap_ratio,
                    )
                    if provisional_reference is not None
                    else None
                )
                self._vote(
                    gap_state,
                    candidate,
                    allowed=track.alert_ready and eligible,
                    can_activate=enough_history and state.nominal_gap is not None,
                )
                if gap_state.active:
                    reference = max(
                        state.nominal_gap or 0.0,
                        self.minimum_gap_pixels
                        if analysis.distance_unit == "pixels"
                        else 0.0,
                    )
                    reference_pixels = (
                        gap_state.last_alert.reference_gap_pixels
                        if gap_state.last_alert is not None
                        else gap_pixels
                    )
                    alert = (
                        candidate
                        or (
                            self._alert_for_gap(
                                analysis,
                                first_index,
                                second_index,
                                gap_pixels,
                                gap_distance,
                                reference,
                                reference_pixels,
                            )
                            if reference > 0.0
                            else gap_state.last_alert
                        )
                    )
                    if alert is not None:
                        alerts.append(alert)
                        alert_gap_ids.append(gap_state.stable_id)
                gap_diagnostics.append(
                    TrackedGap(
                        stable_id=gap_state.stable_id,
                        stable_route_id=track.stable_id,
                        after_index=first_index,
                        endpoint_ids=key,
                        candidate_missing=candidate is not None,
                        active_missing=gap_state.active,
                        positive_votes=sum(gap_state.votes),
                        vote_count=len(gap_state.votes),
                        normal_streak=gap_state.normal_streak,
                        cooldown_remaining=gap_state.cooldown_remaining,
                        business_eligible=eligible,
                    )
                )

            state.cone_ids = cone_ids
            state.centers = centers
            state.gaps = {
                key: gap_state
                for key, gap_state in state.gaps.items()
                if key in current_keys
            }
            diagnostics.append(
                GapRouteThreshold(
                    stable_id=track.stable_id,
                    current_route_index=route_index,
                    nominal_gap_pixels=state.nominal_gap,
                    frame_average_pixels=frame_average,
                    average_of_averages_pixels=(
                        statistics.fmean(state.history) if state.history else None
                    ),
                    history_size=len(state.history),
                    confidence=min(1.0, len(state.history) / self.min_history_frames),
                    excluded_close_gaps=excluded_close,
                    threshold_source=source,
                    distance_unit=analysis.distance_unit,
                )
            )

        return GapTrackingResult(
            tuple(alerts),
            tuple(diagnostics),
            tuple(gap_diagnostics),
            tuple(alert_gap_ids),
        )
