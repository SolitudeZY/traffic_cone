"""Pinhole-camera projection between an oblique image and a flat ground plane."""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class GroundPlaneProjector:
    """Rectify image points onto level ground below a roll-stabilized camera.

    Pitch is the optical-axis depression below horizontal. When altitude is
    omitted, ground coordinates are normalized by camera height; ratios remain
    valid but values are not meters.
    """

    pitch_degrees: float = 45.0
    horizontal_fov_degrees: float = 84.0
    altitude_meters: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.pitch_degrees < 90.0:
            raise ValueError("pitch_degrees must be between 0 and 90")
        if not 0.0 < self.horizontal_fov_degrees < 179.0:
            raise ValueError("horizontal_fov_degrees must be between 0 and 179")
        if self.altitude_meters is not None and self.altitude_meters <= 0.0:
            raise ValueError("altitude_meters must be positive")

    @property
    def distance_unit(self) -> str:
        return "meters" if self.altitude_meters is not None else "camera_heights"

    @property
    def scale(self) -> float:
        return self.altitude_meters if self.altitude_meters is not None else 1.0

    def _intrinsics(
        self, image_size: tuple[int, int]
    ) -> tuple[float, float, float]:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size dimensions must be positive")
        focal = (width / 2.0) / math.tan(
            math.radians(self.horizontal_fov_degrees) / 2.0
        )
        return focal, width / 2.0, height / 2.0

    def image_to_ground(
        self, point: Point, image_size: tuple[int, int]
    ) -> Point | None:
        focal, center_x, center_y = self._intrinsics(image_size)
        image_x = (point[0] - center_x) / focal
        image_y = (point[1] - center_y) / focal
        pitch = math.radians(self.pitch_degrees)
        sine = math.sin(pitch)
        cosine = math.cos(pitch)
        downward = sine + image_y * cosine
        if downward <= 1e-9:
            return None
        scale = self.scale
        return (
            scale * image_x / downward,
            scale * (cosine - image_y * sine) / downward,
        )

    def ground_to_image(
        self, point: Point, image_size: tuple[int, int]
    ) -> Point:
        focal, center_x, center_y = self._intrinsics(image_size)
        lateral = point[0] / self.scale
        forward = point[1] / self.scale
        pitch = math.radians(self.pitch_degrees)
        sine = math.sin(pitch)
        cosine = math.cos(pitch)
        denominator = forward * cosine + sine
        if abs(denominator) <= 1e-9:
            raise ValueError("ground point projects to the camera horizon")
        image_y = (cosine - forward * sine) / denominator
        downward = sine + image_y * cosine
        image_x = lateral * downward
        return (
            center_x + focal * image_x,
            center_y + focal * image_y,
        )

    def distance(
        self,
        first: Point,
        second: Point,
        image_size: tuple[int, int],
    ) -> float | None:
        first_ground = self.image_to_ground(first, image_size)
        second_ground = self.image_to_ground(second, image_size)
        if first_ground is None or second_ground is None:
            return None
        return math.dist(first_ground, second_ground)

    def metadata(self, image_size: tuple[int, int]) -> dict[str, float | str | None]:
        focal, _, _ = self._intrinsics(image_size)
        return {
            "model": "flat_ground_pinhole",
            "pitch_degrees": self.pitch_degrees,
            "horizontal_fov_degrees": self.horizontal_fov_degrees,
            "altitude_meters": self.altitude_meters,
            "focal_length_pixels": focal,
            "distance_unit": self.distance_unit,
        }
