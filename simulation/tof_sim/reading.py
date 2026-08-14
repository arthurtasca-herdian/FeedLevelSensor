"""Measured zone grids exported from a capture log, as frames the reconstruction can consume.

The `.npz` this reads is written by ``python-poc/export_reading.py``. Keeping the handoff a file
rather than an import is deliberate: nothing here knows about the TMF8829 register set, the log
format or the vendor driver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import Intrinsics, directions_to_world, position_of
from .sensor import DEVICE_MAX_DISTANCE_MM, Frame


@dataclass
class Reading:
    """One measurement set. ``distances_mm`` is radial range per zone, NaN where nothing returned."""

    distances_mm: np.ndarray
    signal: np.ndarray
    snr: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.distances_mm.shape

    @property
    def label(self) -> str:
        return self.meta.get("label") or Path(self.meta.get("source", "reading")).name

    def describe(self) -> str:
        rows, cols = self.shape
        finite = self.distances_mm[np.isfinite(self.distances_mm)]
        span = "no targets" if not finite.size else "{:.0f}..{:.0f} mm".format(
            finite.min(), finite.max())
        return "{} set {} - {}x{} zones, {}/{} targets, {}".format(
            self.label, self.meta.get("seq", "?"), rows, cols, finite.size,
            self.distances_mm.size, span)


def load_reading(path: Path | str) -> Reading:
    with np.load(Path(path), allow_pickle=False) as data:
        distances = np.asarray(data["distances_mm"], dtype=float)
        reading = Reading(
            distances_mm=distances,
            signal=np.asarray(data["signal"], dtype=float),
            snr=np.asarray(data["snr"], dtype=float),
            meta=json.loads(str(data["meta"])),
        )

    if distances.ndim != 2:
        raise ValueError(f"distances_mm must be a 2-D zone grid, got shape {distances.shape}")
    if reading.meta.get("unit") != "mm":
        raise ValueError(f"reading is in {reading.meta.get('unit')!r}, not millimetres")

    over = np.isfinite(distances) & (distances > DEVICE_MAX_DISTANCE_MM)
    if over.any():
        raise ValueError(
            f"{over.sum()} zones exceed the device's {DEVICE_MAX_DISTANCE_MM:.0f} mm range; "
            "the file is not a millimetre-scaled reading")

    return reading


def frame_from_reading(reading: Reading, intrinsics: Intrinsics, pose: np.ndarray,
                       min_range_mm: float = 0.0,
                       max_range_mm: float = DEVICE_MAX_DISTANCE_MM) -> Frame:
    """Place a measured grid in the world frame. Ground-truth fields stay None.

    The range gate is the same one ``IdealTofSensor`` applies, so a config describes one sensor
    whether the frame was rendered or measured.
    """
    rows, cols = reading.shape
    if (intrinsics.height, intrinsics.width) != (rows, cols):
        raise ValueError(
            f"reading is {rows}x{cols} zones but the camera is configured for "
            f"{intrinsics.height}x{intrinsics.width}; every point would be misplaced")

    distances = reading.distances_mm
    in_range = np.isfinite(distances) & (distances >= min_range_mm) & (distances <= max_range_mm)

    pose = np.asarray(pose, dtype=float)
    return Frame(
        distance_mm=np.where(in_range, distances, np.nan),
        valid=in_range,
        intrinsics=intrinsics,
        pose=pose,
        origin=position_of(pose),
        directions=directions_to_world(pose, intrinsics.ray_directions()),
    )
