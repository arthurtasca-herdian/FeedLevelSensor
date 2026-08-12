"""Turn a captured frame into a feed-surface estimate.

Only ``distance_mm``, ``valid`` and an assumed camera model are consumed here - the same
information firmware has. Ground-truth fields on the frame are used solely by the ``oracle``
classifier, which exists to separate classification error from surface-fitting error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

from .camera import Intrinsics, unproject
from .geometry import Scene, SiloProfile
from .sensor import Frame


class SurfaceFit(ABC):
    @abstractmethod
    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class MeanLevelFit(SurfaceFit):
    """Single scalar level - the naive product algorithm, and the baseline to beat."""

    def __init__(self, points: np.ndarray):
        self.level = float(np.mean(points[:, 2]))

    def height(self, x, y):
        return np.full(np.broadcast(x, y).shape, self.level, dtype=float)


class PlaneFit(SurfaceFit):
    def __init__(self, points: np.ndarray):
        if len(points) < 3:
            raise ValueError("plane fit needs at least 3 points")
        a = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
        self.coeffs, *_ = np.linalg.lstsq(a, points[:, 2], rcond=None)

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.coeffs[0] * x + self.coeffs[1] * y + self.coeffs[2]


class LinearInterpFit(SurfaceFit):
    """Delaunay-linear inside the sample hull, nearest-neighbour outside it."""

    def __init__(self, points: np.ndarray):
        if len(points) < 4:
            raise ValueError("linear interpolation needs at least 4 points")
        self._linear = LinearNDInterpolator(points[:, :2], points[:, 2])
        self._nearest = NearestNDInterpolator(points[:, :2], points[:, 2])

    def height(self, x, y):
        xy = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
        z = self._linear(xy)
        gaps = ~np.isfinite(z)
        if gaps.any():
            z[gaps] = self._nearest(xy[gaps])
        return z.reshape(np.asarray(x).shape)


class RbfFit(SurfaceFit):
    def __init__(self, points: np.ndarray, kernel: str = "thin_plate_spline", smoothing: float = 0.0):
        if len(points) < 3:
            raise ValueError("RBF fit needs at least 3 points")
        self._rbf = RBFInterpolator(points[:, :2], points[:, 2], kernel=kernel, smoothing=smoothing)

    def height(self, x, y):
        shape = np.asarray(x).shape
        xy = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
        return self._rbf(xy).reshape(shape)


FITTERS = {
    "mean_level": MeanLevelFit,
    "plane_fit": PlaneFit,
    "linear_interp": LinearInterpFit,
    "rbf": RbfFit,
}


@dataclass
class Reconstruction:
    points_world: np.ndarray
    feed_points: np.ndarray
    feed_mask: np.ndarray
    fit: SurfaceFit
    intrinsics_used: Intrinsics
    method: str


def classify_geometric(points: np.ndarray, valid: np.ndarray, profile: SiloProfile,
                       centre=(0.0, 0.0), wall_margin: float = 0.0) -> np.ndarray:
    """Keep zones landing clear of the silo shell.

    The test is against the wall radius *at the point's own height*: in a hopper a genuine
    feed return can sit well inside the body radius, so a single radius threshold would
    accept the sloping wall as feed.
    """
    r = np.hypot(points[..., 0] - centre[0], points[..., 1] - centre[1])
    return valid & (r <= profile.radius_at(points[..., 2]) - wall_margin)


def classify_oracle(frame: Frame, scene: Scene) -> np.ndarray:
    feed_ids = scene.index_of_role("feed")
    return frame.valid & np.isin(frame.surface_id, feed_ids)


def reconstruct(frame: Frame, method: str = "linear_interp",
                intrinsics: Intrinsics | None = None, feed_mask: np.ndarray | None = None,
                **fit_kwargs) -> Reconstruction:
    """Unproject a frame and fit a feed surface to the points classified as feed."""
    if method not in FITTERS:
        raise ValueError(f"unknown reconstruction method {method!r}; choose from {sorted(FITTERS)}")

    assumed = intrinsics or frame.intrinsics
    distances = np.nan_to_num(frame.distance_mm, nan=0.0)
    points = unproject(distances, assumed, frame.pose)

    mask = frame.valid if feed_mask is None else (feed_mask & frame.valid)
    feed_points = points[mask]
    if len(feed_points) == 0:
        raise ValueError("no zones classified as feed surface")

    return Reconstruction(
        points_world=points,
        feed_points=feed_points,
        feed_mask=mask,
        fit=FITTERS[method](feed_points, **fit_kwargs),
        intrinsics_used=assumed,
        method=method,
    )


def surface_mesh(fit: SurfaceFit, profile: SiloProfile, centre=(0.0, 0.0), rings: int = 64,
                 sectors: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Polar tessellation of a fitted surface, clipped to the silo interior, for visualisation."""
    from .geometry import _polar_faces

    r = np.linspace(0.0, profile.body_radius * (1.0 - 1e-9), rings)
    theta = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x = centre[0] + rr * np.cos(tt)
    y = centre[1] + rr * np.sin(tt)
    z = np.asarray(fit.height(x, y), dtype=float)

    vertices = np.stack([x.ravel(), y.ravel(), np.nan_to_num(z).ravel()], axis=1)
    usable = np.isfinite(z).ravel() & profile.contains(rr.ravel(), vertices[:, 2])
    faces = _polar_faces(rings, sectors)
    return vertices, faces[usable[faces].all(axis=1)]
