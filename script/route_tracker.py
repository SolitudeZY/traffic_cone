"""Persistent route tracking for temporally stable traffic-cone ROIs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from scipy.optimize import linear_sum_assignment


Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _bounds_center(bounds: Bounds) -> Point:
    return ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)


def _translate_points(points: Sequence[Point], velocity: Point) -> tuple[Point, ...]:
    return tuple((point[0] + velocity[0], point[1] + velocity[1]) for point in points)


def _translate_bounds(bounds: Bounds, velocity: Point) -> Bounds:
    return (
        bounds[0] + velocity[0],
        bounds[1] + velocity[1],
        bounds[2] + velocity[0],
        bounds[3] + velocity[1],
    )


def _clip_bounds(bounds: Bounds, image_size: tuple[int, int]) -> Bounds:
    width, height = image_size
    x1 = min(float(width), max(0.0, bounds[0]))
    y1 = min(float(height), max(0.0, bounds[1]))
    x2 = min(float(width), max(x1, bounds[2]))
    y2 = min(float(height), max(y1, bounds[3]))
    return (x1, y1, x2, y2)


def _clip_points(points: Sequence[Point], image_size: tuple[int, int]) -> tuple[Point, ...]:
    width, height = image_size
    return tuple(
        (min(float(width), max(0.0, point[0])), min(float(height), max(0.0, point[1])))
        for point in points
    )


def _blend_bounds(predicted: Bounds, observed: Bounds, alpha: float) -> Bounds:
    return tuple(
        predicted[index] * (1.0 - alpha) + observed[index] * alpha
        for index in range(4)
    )  # type: ignore[return-value]


def _bounds_iou(first: Bounds, second: Bounds) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def _principal_angle(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    covariance_xx = statistics.fmean((point[0] - mean_x) ** 2 for point in points)
    covariance_yy = statistics.fmean((point[1] - mean_y) ** 2 for point in points)
    covariance_xy = statistics.fmean(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    )
    return math.degrees(
        0.5 * math.atan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
    ) % 180.0


def _axis_difference(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _blend_axis(first: float, second: float, alpha: float) -> float:
    first_radians = math.radians(first * 2.0)
    second_radians = math.radians(second * 2.0)
    x = math.cos(first_radians) * (1.0 - alpha) + math.cos(second_radians) * alpha
    y = math.sin(first_radians) * (1.0 - alpha) + math.sin(second_radians) * alpha
    return (math.degrees(math.atan2(y, x)) / 2.0) % 180.0


def _typical_spacing(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 20.0
    gaps = [
        _distance(first, second)
        for first, second in zip(points, points[1:])
        if _distance(first, second) > 1e-6
    ]
    return statistics.median(gaps) if gaps else 20.0


@dataclass(frozen=True)
class RouteObservation:
    route_index: int
    detection_indices: tuple[int, ...]
    centers: tuple[Point, ...]
    bounds: Bounds

    def __post_init__(self) -> None:
        if not self.centers:
            raise ValueError("RouteObservation requires at least one center")
        if self.bounds[0] > self.bounds[2] or self.bounds[1] > self.bounds[3]:
            raise ValueError("RouteObservation bounds are invalid")

    @property
    def centroid(self) -> Point:
        return _bounds_center(self.bounds)

    @property
    def direction(self) -> float:
        return _principal_angle(self.centers)

    @property
    def spacing(self) -> float:
        return _typical_spacing(self.centers)


@dataclass(frozen=True)
class TrackedRoute:
    stable_id: int
    state: str
    roi: Bounds
    raw_roi: Bounds | None
    centers: tuple[Point, ...]
    detection_indices: tuple[int, ...]
    current_route_index: int | None
    hits: int
    hit_streak: int
    age: int
    missed_frames: int
    matched: bool
    alert_ready: bool
    direction: float
    spacing: float
    velocity: Point

    @property
    def centroid(self) -> Point:
        return _bounds_center(self.roi)

    @property
    def raw_centroid(self) -> Point | None:
        return _bounds_center(self.raw_roi) if self.raw_roi is not None else None


@dataclass(frozen=True)
class RouteTrackingResult:
    tracks: tuple[TrackedRoute, ...]
    route_to_stable_id: dict[int, int]
    alertable_stable_ids: frozenset[int]


@dataclass
class _TrackState:
    stable_id: int
    state: str
    roi: Bounds
    raw_roi: Bounds | None
    centers: tuple[Point, ...]
    detection_indices: tuple[int, ...]
    current_route_index: int | None
    hits: int
    hit_streak: int
    recovery_hits: int
    age: int
    missed_frames: int
    matched: bool
    direction: float
    spacing: float
    velocity: Point
    last_observed_centroid: Point
    frames_since_observation: int


def observations_from_analysis(analysis: Any) -> list[RouteObservation]:
    observations: list[RouteObservation] = []
    for route_index, route in enumerate(analysis.routes):
        detections = [analysis.ordered[index] for index in route]
        points = [point for detection in detections for point in detection.polygon]
        observations.append(
            RouteObservation(
                route_index=route_index,
                detection_indices=tuple(route),
                centers=tuple(detection.center for detection in detections),
                bounds=(
                    min(point[0] for point in points),
                    min(point[1] for point in points),
                    max(point[0] for point in points),
                    max(point[1] for point in points),
                ),
            )
        )
    return observations


class RouteTracker:
    """Track route groups with stable IDs and independent missed-frame state."""

    def __init__(
        self,
        *,
        confirm_frames: int = 3,
        recovery_frames: int = 2,
        max_missed_frames: int = 5,
        match_distance_ratio: float = 2.5,
        max_direction_difference: float = 45.0,
        base_smoothing: float = 0.35,
        max_smoothing: float = 0.75,
        velocity_momentum: float = 0.6,
    ) -> None:
        if confirm_frames < 1 or recovery_frames < 1:
            raise ValueError("confirmation frame counts must be at least 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must not be negative")
        if match_distance_ratio <= 0.0:
            raise ValueError("match_distance_ratio must be positive")
        if not 0.0 <= base_smoothing <= max_smoothing <= 1.0:
            raise ValueError("smoothing values must satisfy 0 <= base <= max <= 1")
        if not 0.0 <= velocity_momentum < 1.0:
            raise ValueError("velocity_momentum must be in [0, 1)")
        self.confirm_frames = confirm_frames
        self.recovery_frames = recovery_frames
        self.max_missed_frames = max_missed_frames
        self.match_distance_ratio = match_distance_ratio
        self.max_direction_difference = max_direction_difference
        self.base_smoothing = base_smoothing
        self.max_smoothing = max_smoothing
        self.velocity_momentum = velocity_momentum
        self._tracks: list[_TrackState] = []
        self._next_id = 1
        self._last_image_size: tuple[int, int] | None = None

    def hints(self) -> list[list[Point]]:
        hints: list[list[Point]] = []
        for track in self._tracks:
            if track.state == "tentative":
                continue
            predicted = _translate_points(track.centers, track.velocity)
            if self._last_image_size is not None:
                predicted = _clip_points(predicted, self._last_image_size)
            hints.append(list(predicted))
        return hints

    def _predicted(self, track: _TrackState) -> tuple[Bounds, tuple[Point, ...]]:
        return (
            _translate_bounds(track.roi, track.velocity),
            _translate_points(track.centers, track.velocity),
        )

    def _match_cost(self, track: _TrackState, observation: RouteObservation) -> float:
        predicted_bounds, predicted_centers = self._predicted(track)
        scale = max(10.0, track.spacing, observation.spacing)
        candidate_to_track = statistics.median(
            min(_distance(center, predicted) for predicted in predicted_centers)
            for center in observation.centers
        )
        track_to_candidate = statistics.median(
            min(_distance(predicted, center) for center in observation.centers)
            for predicted in predicted_centers
        )
        nearest_distance = max(candidate_to_track, track_to_candidate)
        direction_difference = _axis_difference(track.direction, observation.direction)
        if nearest_distance > self.match_distance_ratio * scale:
            return math.inf
        if direction_difference > self.max_direction_difference:
            return math.inf
        centroid_distance = _distance(
            _bounds_center(predicted_bounds), observation.centroid
        )
        return (
            0.50 * nearest_distance / scale
            + 0.25 * centroid_distance / (scale * 2.0)
            + 0.15 * direction_difference / max(1.0, self.max_direction_difference)
            + 0.10 * (1.0 - _bounds_iou(predicted_bounds, observation.bounds))
        )

    def _assign(self, observations: Sequence[RouteObservation]) -> dict[int, int]:
        if not self._tracks or not observations:
            return {}
        costs = [
            [self._match_cost(track, observation) for observation in observations]
            for track in self._tracks
        ]
        finite_costs = [cost for row in costs for cost in row if math.isfinite(cost)]
        if not finite_costs:
            return {}
        blocked = max(finite_costs) + 1_000_000.0
        matrix = [
            [cost if math.isfinite(cost) else blocked for cost in row]
            for row in costs
        ]
        track_indices, observation_indices = linear_sum_assignment(matrix)
        return {
            int(track_index): int(observation_index)
            for track_index, observation_index in zip(track_indices, observation_indices)
            if math.isfinite(costs[int(track_index)][int(observation_index)])
        }

    def _update_matched(
        self,
        track: _TrackState,
        observation: RouteObservation,
        image_size: tuple[int, int],
    ) -> None:
        predicted_bounds, _ = self._predicted(track)
        observed_bounds = _clip_bounds(observation.bounds, image_size)
        prediction_error = _distance(
            _bounds_center(predicted_bounds), _bounds_center(observed_bounds)
        )
        motion = min(1.0, prediction_error / max(20.0, track.spacing * 2.0))
        alpha = self.base_smoothing + (
            self.max_smoothing - self.base_smoothing
        ) * motion
        smoothed_bounds = _clip_bounds(
            _blend_bounds(predicted_bounds, observed_bounds, alpha), image_size
        )
        raw_centroid = observation.centroid
        elapsed = track.frames_since_observation + 1
        measured_velocity = (
            (raw_centroid[0] - track.last_observed_centroid[0]) / elapsed,
            (raw_centroid[1] - track.last_observed_centroid[1]) / elapsed,
        )
        track.velocity = (
            track.velocity[0] * self.velocity_momentum
            + measured_velocity[0] * (1.0 - self.velocity_momentum),
            track.velocity[1] * self.velocity_momentum
            + measured_velocity[1] * (1.0 - self.velocity_momentum),
        )
        smooth_centroid = _bounds_center(smoothed_bounds)
        center_offset = (
            smooth_centroid[0] - raw_centroid[0],
            smooth_centroid[1] - raw_centroid[1],
        )
        track.centers = _clip_points(
            _translate_points(observation.centers, center_offset), image_size
        )
        track.roi = smoothed_bounds
        track.raw_roi = observed_bounds
        track.detection_indices = observation.detection_indices
        track.current_route_index = observation.route_index
        track.direction = _blend_axis(track.direction, observation.direction, alpha)
        track.spacing = track.spacing * (1.0 - alpha) + observation.spacing * alpha
        track.last_observed_centroid = raw_centroid
        track.frames_since_observation = 0
        track.missed_frames = 0
        track.matched = True
        track.hits += 1
        track.hit_streak += 1
        if track.state == "tentative":
            if track.hit_streak >= self.confirm_frames:
                track.state = "confirmed"
        elif track.state == "held":
            track.recovery_hits += 1
            if track.recovery_hits >= self.recovery_frames:
                track.state = "confirmed"
                track.recovery_hits = 0
        else:
            track.recovery_hits = 0

    def _update_missed(
        self, track: _TrackState, image_size: tuple[int, int]
    ) -> bool:
        if track.state == "tentative":
            return False
        predicted_bounds, predicted_centers = self._predicted(track)
        track.roi = _clip_bounds(predicted_bounds, image_size)
        track.centers = _clip_points(predicted_centers, image_size)
        track.raw_roi = None
        track.detection_indices = ()
        track.current_route_index = None
        track.frames_since_observation += 1
        track.missed_frames += 1
        track.hit_streak = 0
        track.recovery_hits = 0
        track.matched = False
        track.state = "held"
        return track.missed_frames <= self.max_missed_frames

    def _new_track(
        self,
        observation: RouteObservation,
        image_size: tuple[int, int],
    ) -> _TrackState:
        bounds = _clip_bounds(observation.bounds, image_size)
        state = "confirmed" if self.confirm_frames == 1 else "tentative"
        track = _TrackState(
            stable_id=self._next_id,
            state=state,
            roi=bounds,
            raw_roi=bounds,
            centers=_clip_points(observation.centers, image_size),
            detection_indices=observation.detection_indices,
            current_route_index=observation.route_index,
            hits=1,
            hit_streak=1,
            recovery_hits=0,
            age=1,
            missed_frames=0,
            matched=True,
            direction=observation.direction,
            spacing=observation.spacing,
            velocity=(0.0, 0.0),
            last_observed_centroid=observation.centroid,
            frames_since_observation=0,
        )
        self._next_id += 1
        return track

    def update(
        self,
        observations: Sequence[RouteObservation],
        image_size: tuple[int, int],
    ) -> RouteTrackingResult:
        self._last_image_size = image_size
        assignments = self._assign(observations)
        matched_observations = set(assignments.values())
        retained: list[_TrackState] = []
        for track_index, track in enumerate(self._tracks):
            track.age += 1
            observation_index = assignments.get(track_index)
            if observation_index is None:
                if self._update_missed(track, image_size):
                    retained.append(track)
                continue
            self._update_matched(track, observations[observation_index], image_size)
            retained.append(track)
        for observation_index, observation in enumerate(observations):
            if observation_index not in matched_observations:
                retained.append(self._new_track(observation, image_size))
        retained.sort(key=lambda track: track.stable_id)
        self._tracks = retained
        return self._snapshot()

    def _snapshot(self) -> RouteTrackingResult:
        public_tracks = tuple(
            TrackedRoute(
                stable_id=track.stable_id,
                state=track.state,
                roi=track.roi,
                raw_roi=track.raw_roi,
                centers=track.centers,
                detection_indices=track.detection_indices,
                current_route_index=track.current_route_index,
                hits=track.hits,
                hit_streak=track.hit_streak,
                age=track.age,
                missed_frames=track.missed_frames,
                matched=track.matched,
                alert_ready=track.state == "confirmed" and track.matched,
                direction=track.direction,
                spacing=track.spacing,
                velocity=track.velocity,
            )
            for track in self._tracks
        )
        route_to_stable_id = {
            track.current_route_index: track.stable_id
            for track in public_tracks
            if track.current_route_index is not None
        }
        return RouteTrackingResult(
            tracks=public_tracks,
            route_to_stable_id=route_to_stable_id,
            alertable_stable_ids=frozenset(
                track.stable_id for track in public_tracks if track.alert_ready
            ),
        )
