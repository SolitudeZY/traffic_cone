#!/usr/bin/env python3
"""Run OBB inference and traffic-cone spatial rules on images or video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from alert_logger import AlertJournal, format_review_alert
from gap_tracker import GapThresholdTracker, GapTrackingResult
from ground_projection import GroundPlaneProjector
from route_tracker import (
    RouteTracker,
    RouteTrackingResult,
    observations_from_analysis,
)
from traffic_rules import AnalysisResult, ConeDetection, MissingAlert, analyze_cones


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def _parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        coordinates = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain four numbers: x1,y1,x2,y2") from exc
    if len(coordinates) != 4 or coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        raise argparse.ArgumentTypeError("ROI must satisfy x1 < x2 and y1 < y2")
    return coordinates


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--source", required=True, help="Image, directory, video, stream URL, or camera index")
    parser.add_argument("--output", type=Path, default=project_root / "script" / "runs" / "infer")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=0, help="For example 0, cpu, or mps")
    parser.add_argument("--cone-class", default="cone")
    parser.add_argument(
        "--roi",
        type=_parse_roi,
        default=None,
        help="Analyze one cone sequence inside x1,y1,x2,y2; detections outside remain visible",
    )
    parser.add_argument("--gap-ratio", type=float, default=1.8)
    parser.add_argument("--gap-window", type=int, default=5)
    parser.add_argument("--gap-history-frames", type=int, default=30)
    parser.add_argument("--gap-history-min-frames", type=int, default=3)
    parser.add_argument(
        "--gap-close-ratio",
        type=float,
        default=0.55,
        help="Exclude gaps below this ratio of the per-route historical nominal gap",
    )
    parser.add_argument(
        "--gap-max-drop-ratio",
        type=float,
        default=0.05,
        help="Maximum per-frame decrease of a route nominal gap",
    )
    parser.add_argument(
        "--gap-min-pixels",
        type=float,
        default=0.0,
        help="Legacy pixel-mode gap floor; ignored by perspective-corrected distances",
    )
    parser.add_argument("--gap-vote-window", type=int, default=5)
    parser.add_argument("--gap-confirm-votes", type=int, default=3)
    parser.add_argument("--gap-clear-frames", type=int, default=3)
    parser.add_argument(
        "--gap-membership-cooldown",
        type=int,
        default=5,
        help="Frames (3-5) to suppress gaps adjacent to route member changes",
    )
    parser.add_argument(
        "--route-link-ratio",
        type=float,
        default=3.0,
        help="Adaptive ROI link distance as a multiple of local nearest-neighbor spacing",
    )
    parser.add_argument("--route-hint-ratio", type=float, default=2.0)
    parser.add_argument("--min-route-cones", type=int, default=2)
    parser.add_argument(
        "--min-alert-cones",
        type=int,
        default=3,
        help="Minimum current cones in a route before business alerts are allowed",
    )
    parser.add_argument(
        "--roi-memory-frames",
        type=int,
        default=5,
        help="Frames to retain each independently missing confirmed route",
    )
    parser.add_argument("--route-confirm-frames", type=int, default=3)
    parser.add_argument("--route-recovery-frames", type=int, default=2)
    parser.add_argument("--route-match-ratio", type=float, default=2.5)
    parser.add_argument("--route-max-angle", type=float, default=45.0)
    parser.add_argument("--roi-smoothing", type=float, default=0.35)
    parser.add_argument("--roi-max-smoothing", type=float, default=0.75)
    parser.add_argument("--no-temporal-roi", action="store_true")
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=45.0,
        help="Camera optical-axis depression below horizontal",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=84.0,
        help="Camera horizontal field of view used for ground rectification",
    )
    parser.add_argument(
        "--drone-altitude-m",
        type=float,
        default=None,
        help="Camera height above the cone ground plane; enables metric distances",
    )
    parser.add_argument(
        "--no-perspective-correction",
        action="store_true",
        help="Disable oblique flat-ground rectification and use legacy pixel distances",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="Ground calibration at this view; enables metric displacement threshold",
    )
    parser.add_argument("--offset-threshold-m", type=float, default=0.5)
    parser.add_argument(
        "--offset-ratio",
        type=float,
        default=0.25,
        help="Uncalibrated fallback: displacement / median cone spacing",
    )
    parser.add_argument(
        "--upright-angle",
        type=float,
        default=90.0,
        help="Expected upright cone long-axis angle in image coordinates",
    )
    parser.add_argument("--fallen-angle", type=float, default=30.0)
    parser.add_argument("--fallen-edge-margin", type=float, default=0.05)
    parser.add_argument(
        "--exclude-classes",
        default="person,vest",
        help="Comma-separated classes from the same OBB model whose boxes suppress cone centers",
    )
    parser.add_argument(
        "--person-weights",
        default=None,
        help="Optional COCO person detector weights used to suppress cones inside person boxes",
    )
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument(
        "--person-classes",
        default="0",
        help="Comma-separated exclusion class IDs; use 0,2 for the project PPE model",
    )
    parser.add_argument("--person-imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def require_runtime():
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependency. Run: pip install -r script/requirements.txt"
        ) from exc
    return cv2, np, YOLO


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _inside_any_box(
    point: tuple[float, float], boxes: Sequence[Sequence[float]]
) -> bool:
    return any(
        box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        for box in boxes
    )


def _same_model_exclusion_boxes(
    result: Any, excluded_classes: set[str]
) -> list[tuple[float, float, float, float]]:
    if not excluded_classes or result.obb is None or len(result.obb) == 0:
        return []
    polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
    class_ids = result.obb.cls.detach().cpu().numpy().astype(int)
    boxes: list[tuple[float, float, float, float]] = []
    for polygon, class_id in zip(polygons, class_ids):
        if _class_name(result.names, int(class_id)) not in excluded_classes:
            continue
        boxes.append(
            (
                float(polygon[:, 0].min()),
                float(polygon[:, 1].min()),
                float(polygon[:, 0].max()),
                float(polygon[:, 1].max()),
            )
        )
    return boxes


def _extract_detections(
    result: Any,
    cone_class: str,
    roi: tuple[float, float, float, float] | None,
    exclusion_boxes: Sequence[Sequence[float]] = (),
) -> tuple[list[ConeDetection], list[dict[str, Any]], int]:
    cones: list[ConeDetection] = []
    all_detections: list[dict[str, Any]] = []
    suppressed = 0
    obb = result.obb
    if obb is None or len(obb) == 0:
        return cones, all_detections, suppressed
    polygons = obb.xyxyxyxy.detach().cpu().numpy()
    confidences = obb.conf.detach().cpu().numpy()
    class_ids = obb.cls.detach().cpu().numpy().astype(int)
    for polygon_array, confidence, class_id in zip(polygons, confidences, class_ids):
        polygon = tuple((float(point[0]), float(point[1])) for point in polygon_array)
        if len(polygon) != 4:
            continue
        label = _class_name(result.names, int(class_id))
        item = {
            "polygon": polygon,
            "confidence": float(confidence),
            "class_id": int(class_id),
            "label": label,
        }
        if label == cone_class:
            cone = ConeDetection(**item)
            center_x, center_y = cone.center
            if _inside_any_box((center_x, center_y), exclusion_boxes):
                suppressed += 1
                continue
            all_detections.append(item)
            if roi is None or (roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]):
                cones.append(cone)
        else:
            all_detections.append(item)
    return cones, all_detections, suppressed


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _route_bounds(analysis: AnalysisResult, route: list[int]) -> tuple[float, float, float, float]:
    points = [
        point
        for index in route
        for point in analysis.ordered[index].polygon
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _route_id(analysis: AnalysisResult, detection_index: int) -> int | None:
    return next(
        (route_id for route_id, route in enumerate(analysis.routes) if detection_index in route),
        None,
    )


def _stable_route_id(
    analysis: AnalysisResult,
    detection_index: int,
    tracking: RouteTrackingResult | None,
) -> int | None:
    if tracking is None:
        return None
    route_id = _route_id(analysis, detection_index)
    if route_id is not None:
        return tracking.route_to_stable_id.get(route_id)
    if not 0 <= detection_index < len(analysis.ordered):
        return None
    point = analysis.ordered[detection_index].center
    candidates: list[tuple[float, int]] = []
    for track in tracking.tracks:
        if track.state == "tentative" or not track.centers:
            continue
        distance = min(
            ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
            for center in track.centers
        )
        if distance <= max(20.0, track.spacing * 2.5):
            candidates.append((distance, track.stable_id))
    return min(candidates)[1] if candidates else None


def _alert_allowed(
    analysis: AnalysisResult,
    detection_index: int,
    tracking: RouteTrackingResult | None,
    min_alert_cones: int = 3,
) -> bool:
    if tracking is None:
        route_id = _route_id(analysis, detection_index)
        return route_id is None or len(analysis.routes[route_id]) >= min_alert_cones
    stable_id = _stable_route_id(analysis, detection_index, tracking)
    if stable_id not in tracking.alertable_stable_ids:
        return False
    track = next(
        (item for item in tracking.tracks if item.stable_id == stable_id),
        None,
    )
    if track is None or track.current_route_index is None:
        return False
    return len(analysis.routes[track.current_route_index]) >= min_alert_cones


def _draw_frame(
    cv2: Any,
    np: Any,
    frame: Any,
    detections: list[dict[str, Any]],
    analysis: AnalysisResult,
    roi: tuple[float, float, float, float] | None,
    tracking: RouteTrackingResult | None = None,
    gap_tracking: GapTrackingResult | None = None,
    min_alert_cones: int = 3,
) -> Any:
    fallen_indices = {
        alert.index
        for alert in analysis.fallen
        if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
    }
    displaced_indices = {
        alert.index
        for alert in analysis.displaced
        if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
    }
    edge_ignored_indices = set(analysis.edge_ignored)

    for item in detections:
        polygon = np.asarray(item["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        color = (255, 180, 0) if item["label"] != "cone" else (50, 210, 50)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        first = tuple(int(value) for value in polygon[0, 0])
        cv2.putText(
            frame,
            f"{item['label']} {item['confidence']:.2f}",
            (first[0], max(18, first[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    route_colors = ((0, 220, 255), (255, 80, 80), (80, 255, 120), (255, 120, 255))
    if tracking is None:
        drawable_routes = [
            (route_id + 1, "confirmed", _route_bounds(analysis, route), route)
            for route_id, route in enumerate(analysis.routes)
        ]
    else:
        drawable_routes = [
            (
                track.stable_id,
                track.state,
                track.roi,
                (
                    analysis.routes[track.current_route_index]
                    if track.matched
                    and track.current_route_index is not None
                    and track.current_route_index < len(analysis.routes)
                    else []
                ),
            )
            for track in tracking.tracks
            if track.state != "tentative"
        ]
    for stable_id, state, bounds, route in drawable_routes:
        color = route_colors[(stable_id - 1) % len(route_colors)]
        centers = [analysis.ordered[index].center for index in route]
        for first, second in zip(centers, centers[1:]):
            cv2.line(frame, _point(first), _point(second), color, 2, cv2.LINE_AA)
        if route or tracking is not None:
            x1, y1, x2, y2 = bounds
            padding = 8
            top_left = _point((max(0, x1 - padding), max(0, y1 - padding)))
            bottom_right = _point((x2 + padding, y2 + padding))
            cv2.rectangle(frame, top_left, bottom_right, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"AUTO ROI {stable_id}{' HOLD' if state == 'held' else ''}",
                (top_left[0], max(16, top_left[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for index, detection in enumerate(analysis.ordered):
        center = _point(detection.center)
        color = (
            (160, 160, 160)
            if index in edge_ignored_indices
            else (0, 0, 255)
            if index in fallen_indices or index in displaced_indices
            else (0, 255, 255)
        )
        cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
        if index in edge_ignored_indices:
            cv2.putText(frame, "EDGE SKIP", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif index in fallen_indices:
            cv2.putText(frame, "FALLEN", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        elif index in displaced_indices:
            cv2.putText(frame, "SHIFT", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    missing_alerts = (
        gap_tracking.missing if gap_tracking is not None else analysis.missing
    )
    for alert in missing_alerts:
        if gap_tracking is None and not _alert_allowed(
            analysis, alert.after_index, tracking, min_alert_cones
        ):
            continue
        for position in alert.positions:
            center = _point(position)
            cv2.circle(frame, center, 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "MISSING", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    if roi is not None:
        cv2.rectangle(frame, _point((roi[0], roi[1])), _point((roi[2], roi[3])), (255, 255, 0), 2)
    return frame


def _events(
    frame_index: int,
    analysis: AnalysisResult,
    detections: list[dict[str, Any]],
    roi: tuple[float, float, float, float] | None,
    person_suppressed: int,
    tracking: RouteTrackingResult | None = None,
    gap_tracking: GapTrackingResult | None = None,
    video_time_seconds: float | None = None,
    min_alert_cones: int = 3,
) -> dict[str, Any]:
    def missing_event(
        alert: MissingAlert, stable_gap_id: int | None = None
    ) -> dict[str, Any]:
        event = {
            "after_index": alert.after_index,
            "route_id": _route_id(analysis, alert.after_index),
            "stable_route_id": _stable_route_id(
                analysis, alert.after_index, tracking
            ),
            "gap_pixels": alert.gap_pixels,
            "reference_gap_pixels": alert.reference_gap_pixels,
            "estimated_count": alert.estimated_count,
            "positions": alert.positions,
            "gap_corrected": alert.gap_corrected,
            "reference_gap_corrected": alert.reference_gap_corrected,
            "distance_unit": alert.distance_unit,
        }
        if stable_gap_id is not None:
            event["stable_gap_id"] = stable_gap_id
        return event

    missing_events = [missing_event(alert) for alert in analysis.missing]
    stable_missing_events = (
        [
            missing_event(alert, stable_gap_id)
            for alert, stable_gap_id in zip(
                gap_tracking.missing, gap_tracking.missing_gap_ids
            )
        ]
        if gap_tracking is not None
        else [
            event
            for alert, event in zip(analysis.missing, missing_events)
            if _alert_allowed(
                analysis, alert.after_index, tracking, min_alert_cones
            )
        ]
    )
    displaced_events = [
        {
            "index": alert.index,
            "route_id": _route_id(analysis, alert.index),
            "stable_route_id": _stable_route_id(analysis, alert.index, tracking),
            "distance_pixels": alert.distance_pixels,
            "distance_meters": alert.distance_meters,
            "distance_corrected": alert.distance_corrected,
            "distance_unit": alert.distance_unit,
        }
        for alert in analysis.displaced
    ]
    fallen_events = [
        {
            "index": alert.index,
            "stable_route_id": _stable_route_id(analysis, alert.index, tracking),
            "obb_axis_angle_degrees": alert.angle_degrees,
            "upright_difference_degrees": alert.difference_degrees,
        }
        for alert in analysis.fallen
    ]
    return {
        "frame_index": frame_index,
        "timestamp_seconds": time.time(),
        "video_time_seconds": video_time_seconds,
        "distance_unit": analysis.distance_unit,
        "perspective_calibration": analysis.perspective_calibration,
        "cone_count": len(analysis.ordered),
        "roi": roi,
        "person_suppressed": person_suppressed,
        "detections": [
            {
                **item,
                "polygon": [list(point) for point in item["polygon"]],
                "center": [
                    sum(point[0] for point in item["polygon"]) / 4.0,
                    sum(point[1] for point in item["polygon"]) / 4.0,
                ],
            }
            for item in detections
        ],
        "ordered_cone_centers": [detection.center for detection in analysis.ordered],
        "ordered_cone_ground_contacts": [
            detection.ground_contact for detection in analysis.ordered
        ],
        "ordered_cone_ground_centers": analysis.ground_centers,
        "adaptive_routes": [
            {
                "route_id": route_id,
                "stable_route_id": (
                    tracking.route_to_stable_id.get(route_id)
                    if tracking is not None
                    else None
                ),
                "indices": route,
                "roi": _route_bounds(analysis, route),
                "centers": [analysis.ordered[index].center for index in route],
            }
            for route_id, route in enumerate(analysis.routes)
        ],
        "tracked_routes": (
            [
                {
                    "stable_route_id": track.stable_id,
                    "state": track.state,
                    "matched": track.matched,
                    "alert_ready": track.alert_ready,
                    "current_route_id": track.current_route_index,
                    "roi": track.roi,
                    "raw_roi": track.raw_roi,
                    "centers": track.centers,
                    "hits": track.hits,
                    "hit_streak": track.hit_streak,
                    "age": track.age,
                    "missed_frames": track.missed_frames,
                    "velocity": track.velocity,
                }
                for track in tracking.tracks
            ]
            if tracking is not None
            else []
        ),
        "median_gap_pixels": analysis.median_gap_pixels,
        "median_gap_distance": analysis.median_gap_distance,
        "gap_thresholds": (
            [
                {
                    "stable_route_id": route.stable_id,
                    "current_route_id": route.current_route_index,
                    "nominal_gap_pixels": (
                        route.nominal_gap_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "frame_average_pixels": (
                        route.frame_average_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "average_of_averages_pixels": (
                        route.average_of_averages_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "history_size": route.history_size,
                    "confidence": route.confidence,
                    "excluded_close_gaps": route.excluded_close_gaps,
                    "threshold_source": route.threshold_source,
                    "distance_unit": route.distance_unit,
                    "nominal_gap_corrected": (
                        route.nominal_gap_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                    "frame_average_corrected": (
                        route.frame_average_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                    "average_of_averages_corrected": (
                        route.average_of_averages_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                }
                for route in gap_tracking.routes
            ]
            if gap_tracking is not None
            else []
        ),
        "tracked_gaps": (
            [
                {
                    "stable_gap_id": gap.stable_id,
                    "stable_route_id": gap.stable_route_id,
                    "after_index": gap.after_index,
                    "endpoint_ids": gap.endpoint_ids,
                    "candidate_missing": gap.candidate_missing,
                    "active_missing": gap.active_missing,
                    "positive_votes": gap.positive_votes,
                    "vote_count": gap.vote_count,
                    "normal_streak": gap.normal_streak,
                    "cooldown_remaining": gap.cooldown_remaining,
                    "business_eligible": gap.business_eligible,
                }
                for gap in gap_tracking.gaps
            ]
            if gap_tracking is not None
            else []
        ),
        "missing": missing_events,
        "stable_missing": stable_missing_events,
        "displaced": displaced_events,
        "stable_displaced": [
            event
            for alert, event in zip(analysis.displaced, displaced_events)
            if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
        ],
        "edge_ignored": analysis.edge_ignored,
        "fallen": fallen_events,
        "stable_fallen": [
            event
            for alert, event in zip(analysis.fallen, fallen_events)
            if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
        ],
    }


def _video_fps(cv2: Any, source: str) -> float:
    if source.isdigit():
        return 25.0
    capture = cv2.VideoCapture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps > 0 else 25.0


def main() -> int:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    cv2, np, YOLO = require_runtime()
    if args.drone_altitude_m is not None and args.meters_per_pixel is not None:
        raise SystemExit(
            "--drone-altitude-m and --meters-per-pixel are mutually exclusive"
        )
    ground_projector = (
        GroundPlaneProjector(
            pitch_degrees=args.camera_pitch_deg,
            horizontal_fov_degrees=args.camera_hfov_deg,
            altitude_meters=args.drone_altitude_m,
        )
        if not args.no_perspective_correction and args.meters_per_pixel is None
        else None
    )
    model = YOLO(str(args.weights.resolve()))
    person_model = YOLO(args.person_weights) if args.person_weights else None
    excluded_classes = {
        value.strip() for value in args.exclude_classes.split(",") if value.strip()
    }
    try:
        person_classes = [int(value) for value in args.person_classes.split(",")]
    except ValueError as exc:
        raise SystemExit("--person-classes must contain comma-separated integers") from exc
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    predict_args = {
        "source": source,
        "stream": True,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / "events.jsonl"
    structured_alert_path = args.output / "alert.jsonl"
    review_alert_path = args.output / "alert.log"
    source_path = Path(args.source)
    is_video = (
        args.source.isdigit()
        or source_path.suffix.lower() in VIDEO_SUFFIXES
        or args.source.startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )
    video_fps = _video_fps(cv2, args.source) if is_video else 0.0
    writer = None
    route_tracker = (
        RouteTracker(
            confirm_frames=args.route_confirm_frames,
            recovery_frames=args.route_recovery_frames,
            max_missed_frames=args.roi_memory_frames,
            match_distance_ratio=args.route_match_ratio,
            max_direction_difference=args.route_max_angle,
            base_smoothing=args.roi_smoothing,
            max_smoothing=args.roi_max_smoothing,
        )
        if is_video and not args.no_temporal_roi
        else None
    )
    gap_tracker = (
        GapThresholdTracker(
            history_frames=args.gap_history_frames,
            min_history_frames=args.gap_history_min_frames,
            close_ratio=args.gap_close_ratio,
            max_drop_ratio=args.gap_max_drop_ratio,
            minimum_gap_pixels=args.gap_min_pixels,
            vote_window=args.gap_vote_window,
            confirm_votes=args.gap_confirm_votes,
            clear_frames=args.gap_clear_frames,
            membership_cooldown_frames=args.gap_membership_cooldown,
            min_alert_cones=args.min_alert_cones,
        )
        if route_tracker is not None
        else None
    )
    events_file = event_path.open("w", encoding="utf-8")
    structured_alert_file = structured_alert_path.open("w", encoding="utf-8")
    review_alert_file = review_alert_path.open("w", encoding="utf-8")
    alert_journal = AlertJournal()
    try:
        for frame_index, result in enumerate(model.predict(**predict_args)):
            video_time_seconds = (
                frame_index / video_fps if video_fps > 0.0 else 0.0
            )
            frame = result.orig_img.copy()
            height, width = frame.shape[:2]
            person_boxes = _same_model_exclusion_boxes(result, excluded_classes)
            if person_model is not None:
                person_result = person_model.predict(
                    frame,
                    classes=person_classes,
                    conf=args.person_conf,
                    imgsz=args.person_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                if person_result.boxes is not None:
                    person_boxes.extend(person_result.boxes.xyxy.detach().cpu().tolist())
            cones, detections, person_suppressed = _extract_detections(
                result, args.cone_class, args.roi, person_boxes
            )
            analysis = analyze_cones(
                cones,
                gap_ratio=args.gap_ratio,
                gap_window=args.gap_window,
                route_link_ratio=args.route_link_ratio,
                route_hints=route_tracker.hints() if route_tracker is not None else None,
                route_hint_ratio=args.route_hint_ratio,
                min_route_cones=args.min_route_cones,
                image_size=(width, height),
                fallen_edge_margin_ratio=args.fallen_edge_margin,
                offset_ratio=args.offset_ratio,
                offset_threshold_meters=args.offset_threshold_m,
                meters_per_pixel=args.meters_per_pixel,
                ground_projector=ground_projector,
                upright_angle=args.upright_angle,
                fallen_angle=args.fallen_angle,
            )
            tracking = (
                route_tracker.update(
                    observations_from_analysis(analysis), (width, height)
                )
                if route_tracker is not None
                else None
            )
            gap_tracking = (
                gap_tracker.update(analysis, tracking, gap_ratio=args.gap_ratio)
                if gap_tracker is not None and tracking is not None
                else None
            )
            frame = _draw_frame(
                cv2,
                np,
                frame,
                detections,
                analysis,
                args.roi,
                tracking,
                gap_tracking,
                args.min_alert_cones,
            )
            event = _events(
                frame_index,
                analysis,
                detections,
                args.roi,
                person_suppressed,
                tracking,
                gap_tracking,
                video_time_seconds,
                args.min_alert_cones,
            )
            events_file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )
            events_file.flush()
            for alert_record in alert_journal.update(
                event, video_time_seconds
            ):
                structured_alert_file.write(
                    json.dumps(alert_record, ensure_ascii=False) + "\n"
                )
                review_alert_file.write(
                    format_review_alert(alert_record) + "\n"
                )
            structured_alert_file.flush()
            review_alert_file.flush()

            if not args.no_save:
                if is_video:
                    if writer is None:
                        height, width = frame.shape[:2]
                        output_video = args.output / f"{source_path.stem or 'camera'}_result.mp4"
                        writer = cv2.VideoWriter(
                            str(output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {output_video}")
                    writer.write(frame)
                else:
                    image_name = f"{Path(result.path).stem}_result.jpg"
                    cv2.imwrite(str(args.output / image_name), frame)
            if args.display:
                cv2.imshow("Traffic cone OBB", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        events_file.close()
        structured_alert_file.close()
        review_alert_file.close()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    print(f"Inference events: {event_path}")
    print(f"Structured alerts: {structured_alert_path}")
    print(f"Review alerts: {review_alert_path}")
    if not args.no_save:
        print(f"Annotated output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
