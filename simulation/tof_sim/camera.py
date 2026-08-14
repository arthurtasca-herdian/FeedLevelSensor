"""Frame conventions, pinhole intrinsics and pose handling for the simulated sensor.

World frame: +Z up, origin at the centre of the silo floor. All lengths in millimetres.
Camera frame: OpenCV convention - +Z along the optical axis, +X right, +Y down.
A pose is a 4x4 homogeneous ``world_T_cam``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Beyond this alignment between the view direction and the up hint the cross product
# that builds the camera basis loses precision, so an alternate hint is substituted.
_UP_DEGENERACY_LIMIT = 0.999

_UP_FALLBACKS = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


@dataclass(frozen=True)
class Intrinsics:
    # Pinhole intrinsics for a rectangular grid of ToF zones.
    # Zone (row, col) is sampled at pixel coordinate (col + 0.5, row + 0.5), so the
    # grid spans [0, width] x [0, height] and the field of view is measured to that edge.

    width: int
    height: int
    fx: float # focal length measured in zone-widths
    fy: float # focal length measured in zone-widths
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, vfov_deg: float) -> "Intrinsics":
        return cls(
            width=int(width),
            height=int(height),
            fx=(width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0),
            fy=(height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0),
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @classmethod
    def from_vendor_zcorrection(cls, width: int, height: int) -> "Intrinsics":
        # Intrinsics equivalent to the ams driver's ``zCorrection``.
        return cls(
            width=int(width),
            height=int(height),
            fx=0.75 * width,
            fy=float(height),
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan((self.width / 2.0) / self.fx)))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan((self.height / 2.0) / self.fy)))

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def ray_directions(self) -> np.ndarray:
        cols, rows = np.meshgrid(
            np.arange(self.width, dtype=float),
            np.arange(self.height, dtype=float),
            indexing="xy",
        )
        x = (cols + 0.5 - self.cx) / self.fx
        y = (rows + 0.5 - self.cy) / self.fy
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        points_cam = np.asarray(points_cam, dtype=float)
        z = points_cam[..., 2]
        u = self.fx * points_cam[..., 0] / z + self.cx
        v = self.fy * points_cam[..., 1] / z + self.cy
        return np.stack([u, v], axis=-1)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def from_euler(position, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
               roll_deg: float = 0.0) -> np.ndarray:
    """Build ``world_T_cam`` from mounting angles against a world-aligned reference.

    position: sensor focus point in world coordinates.
    yaw_deg: azimuth the optical axis leans towards, about world +Z, CCW from +X.
    pitch_deg: angle of the optical axis away from world +Z. 0 looks straight up,
        180 straight down, so a roof-mounted sensor sits near 180.
    roll_deg: rotation about the optical axis.

    At yaw = pitch = roll = 0 the camera axes coincide with the world axes,
    i.e. ordinary spherical coordinates.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    pose = np.eye(4)
    pose[:3, :3] = _rot_z(yaw) @ _rot_y(pitch) @ _rot_z(roll)
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


def look_at(position, target, up=(0.0, 0.0, 1.0), roll_deg: float = 0.0) -> np.ndarray:
    """Build ``world_T_cam`` for a sensor at ``position`` whose optical axis points at ``target``."""
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = target - position
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        raise ValueError("look_at target coincides with the camera position")
    forward /= norm

    hints = (np.asarray(up, dtype=float),) + tuple(np.asarray(h) for h in _UP_FALLBACKS)
    for hint in hints:
        hint_norm = np.linalg.norm(hint)
        if hint_norm == 0.0:
            continue
        hint = hint / hint_norm
        if abs(float(np.dot(forward, hint))) < _UP_DEGENERACY_LIMIT:
            break
    else:
        raise ValueError("no usable up hint for the requested view direction")

    right = np.cross(forward, hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward]) @ _rot_z(np.radians(roll_deg))
    pose[:3, 3] = position
    return pose


def rotation_of(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float)[:3, :3]


def position_of(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float)[:3, 3]


def to_world(pose: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=float)
    return points_cam @ rotation_of(pose).T + position_of(pose)


def directions_to_world(pose: np.ndarray, dirs_cam: np.ndarray) -> np.ndarray:
    return np.asarray(dirs_cam, dtype=float) @ rotation_of(pose).T


def unproject(distance_mm: np.ndarray, intrinsics: Intrinsics, pose: np.ndarray) -> np.ndarray:
    """World-frame points for a grid of radial distances, shaped (height, width, 3).

    ``distance_mm`` is the Euclidean focus-point-to-target range the device reports, so it
    scales the unit ray directly. ``intrinsics`` is the model *assumed* by the reconstruction,
    which need not be the one used to render the frame.
    """
    dirs = intrinsics.ray_directions()
    points_cam = dirs * np.asarray(distance_mm, dtype=float)[..., None]
    return to_world(pose, points_cam)
