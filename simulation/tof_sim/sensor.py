"""Noise-free model of the TMF8829 as a grid of independent ranging zones.

Each zone is sampled by a single ray through its centre, so the only error the sensor
itself contributes is range quantisation. Reflectance, multipath, ambient light and
per-zone footprint are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Intrinsics, directions_to_world, position_of
from .geometry import Scene

# The device reports distance as uint16 in 0.25 mm steps.
DEVICE_LSB_MM = 0.25
DEVICE_MAX_DISTANCE_MM = (2**16 - 1) * DEVICE_LSB_MM


@dataclass
class Frame:
    """One captured grid. ``distance_mm`` and ``valid`` are all a reconstruction may use."""

    distance_mm: np.ndarray
    valid: np.ndarray
    intrinsics: Intrinsics
    pose: np.ndarray
    origin: np.ndarray
    directions: np.ndarray
    # Ground truth, so only a rendered frame has these. A measured one leaves them None, which
    # makes the consumers that need them fail rather than score a reconstruction against nothing.
    surface_id: np.ndarray | None = None
    incidence_deg: np.ndarray | None = None
    hit_points: np.ndarray | None = None
    true_distance_mm: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.distance_mm.shape

    def valid_points(self) -> np.ndarray:
        return self.hit_points[self.valid]


class IdealTofSensor:
    def __init__(self, intrinsics: Intrinsics, pose: np.ndarray, min_range_mm: float = 0.0,
                 max_range_mm: float = 10_000.0, quantization_mm: float = DEVICE_LSB_MM):
        self.intrinsics = intrinsics
        self.pose = np.asarray(pose, dtype=float)
        self.min_range_mm = float(min_range_mm)
        self.max_range_mm = min(float(max_range_mm), DEVICE_MAX_DISTANCE_MM)
        self.quantization_mm = float(quantization_mm)

    def capture(self, scene: Scene) -> Frame:
        height, width = self.intrinsics.height, self.intrinsics.width
        dirs_world = directions_to_world(self.pose, self.intrinsics.ray_directions())
        flat_dirs = dirs_world.reshape(-1, 3)
        origin = position_of(self.pose)
        origins = np.broadcast_to(origin, flat_dirs.shape)

        t, hit, normal, surface_id = scene.intersect(origins, flat_dirs)
        in_range = hit & (t >= self.min_range_mm) & (t <= self.max_range_mm)

        distance = np.where(in_range, t, np.nan)
        if self.quantization_mm > 0.0:
            distance = np.round(distance / self.quantization_mm) * self.quantization_mm

        cos_incidence = np.clip(np.abs(np.sum(flat_dirs * normal, axis=1)), 0.0, 1.0)
        incidence = np.where(in_range, np.degrees(np.arccos(cos_incidence)), np.nan)

        points = np.where(in_range[:, None], origins + np.where(hit, t, 0.0)[:, None] * flat_dirs, np.nan)

        grid = (height, width)
        return Frame(
            distance_mm=distance.reshape(grid),
            valid=in_range.reshape(grid),
            intrinsics=self.intrinsics,
            pose=self.pose,
            origin=origin,
            directions=dirs_world,
            surface_id=np.where(in_range, surface_id, -1).reshape(grid),
            incidence_deg=incidence.reshape(grid),
            hit_points=points.reshape(*grid, 3),
            true_distance_mm=np.where(in_range, t, np.nan).reshape(grid),
        )
