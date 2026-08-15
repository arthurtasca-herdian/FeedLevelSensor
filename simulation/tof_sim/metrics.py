"""Scoring of a reconstruction against the analytic ground-truth surface.

Both volumes are integrated on the same grid so that quadrature error cancels in the
difference and what remains is attributable to the reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import Delaunay

from .geometry import HeightField, SiloProfile
from .reconstruct import Reconstruction, SurfaceFit
from .sensor import Frame


@dataclass
class ResidualField:
    x: np.ndarray
    y: np.ndarray
    z_true: np.ndarray
    z_recon: np.ndarray
    residual: np.ndarray
    inside: np.ndarray


@dataclass
class Metrics:
    volume_true_mm3: float
    volume_recon_mm3: float
    volume_error_mm3: float
    volume_error_frac: float
    level_error_mm: float
    residual_rms_mm: float
    residual_bias_mm: float
    residual_max_abs_mm: float
    residual_p95_mm: float
    sample_offset_rms_mm: float
    sample_offset_max_mm: float
    feed_zone_count: int
    valid_zone_count: int
    zone_count: int
    field: ResidualField | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("field")
        return data


def _integration_grid(profile: SiloProfile, centre=(0.0, 0.0), grid: int = 512) -> tuple:
    """Columns every volume and residual number is summed over: xx, yy, rho, in_silo, cell_area."""
    radius = profile.body_radius
    axis = np.linspace(-radius, radius, grid)
    xx, yy = np.meshgrid(centre[0] + axis, centre[1] + axis, indexing="xy")
    rho = np.hypot(xx - centre[0], yy - centre[1])
    return xx, yy, rho, rho <= radius, ((2.0 * radius) / (grid - 1)) ** 2


def volume_under(surface, profile: SiloProfile, centre=(0.0, 0.0), grid: int = 512) -> float:
    # A ``HeightField`` and a ``SurfaceFit`` both satisfy that, so ground truth and reconstruction
    # are integrated by the same code on the same columns and quadrature error cancels between them.
    # """
    xx, yy, rho, in_silo, cell_area = _integration_grid(profile, centre, grid)
    z = np.asarray(surface.height(xx, yy), dtype=float)
    # no unit conversion, uses the same provided by input data
    return float(np.sum(_column_heights(z, rho, profile)[in_silo]) * cell_area)


def sampled_fraction(points: np.ndarray, profile: SiloProfile, centre=(0.0, 0.0),
                     grid: int = 512) -> float:
    # Every fitter extrapolates silently past its samples, so a volume alone cannot say how much of
    # itself was measured. Measured on the same columns as ``volume_under`` so the two agree.
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0

    xx, yy, _, in_silo, _ = _integration_grid(profile, centre, grid)
    hull = Delaunay(points[:, :2])
    covered = hull.find_simplex(np.column_stack([xx.ravel(), yy.ravel()])) >= 0
    return float(np.count_nonzero(covered.reshape(xx.shape) & in_silo) / np.count_nonzero(in_silo))


def residual_field(feed: HeightField, fit: SurfaceFit, profile: SiloProfile, centre=(0.0, 0.0),
                   grid: int = 512) -> ResidualField:
    radius = profile.body_radius
    xx, yy, rho, _, _ = _integration_grid(profile, centre, grid)

    z_true = feed.height(xx, yy)
    z_recon = np.asarray(fit.height(xx, yy), dtype=float)

    # The feed surface only exists where it is inside the silo: below a given radius the
    # hopper wall, not the surface, is what a ray would meet.
    inside = np.isfinite(z_true) & (rho <= radius) & profile.contains(rho, np.nan_to_num(z_true))

    residual = np.where(inside, z_recon - z_true, np.nan)
    return ResidualField(x=xx, y=yy, z_true=z_true, z_recon=z_recon, residual=residual, inside=inside)


def _column_heights(z_surface: np.ndarray, rho: np.ndarray, profile: SiloProfile) -> np.ndarray:
    """Depth of feed in each vertical column, clamped to the silo's interior at that radius."""
    z_lo = profile.z_lower(rho)
    z_hi = profile.z_upper(rho)
    with np.errstate(invalid="ignore"):
        filled = np.clip(np.nan_to_num(z_surface, nan=-np.inf), z_lo, z_hi) - z_lo
    return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate(frame: Frame, reconstruction: Reconstruction, feed: HeightField,
             profile: SiloProfile, centre=(0.0, 0.0), grid: int = 512,
             keep_field: bool = True) -> Metrics:
    """Volume, level and residual error of a reconstruction against the true feed surface."""
    rf = residual_field(feed, reconstruction.fit, profile=profile, centre=centre, grid=grid)
    _, _, _, _, cell_area = _integration_grid(profile, centre, grid)

    volume_true = volume_under(feed, profile, centre=centre, grid=grid)
    volume_recon = volume_under(reconstruction.fit, profile, centre=centre, grid=grid)
    volume_error = volume_recon - volume_true

    inside = rf.inside
    cross_section = float(np.count_nonzero(inside) * cell_area)
    residuals = rf.residual[inside]
    residuals = residuals[np.isfinite(residuals)]

    offsets = _sample_offsets(reconstruction, feed)

    return Metrics(
        volume_true_mm3=volume_true,
        volume_recon_mm3=volume_recon,
        volume_error_mm3=volume_error,
        volume_error_frac=volume_error / volume_true if volume_true else float("nan"),
        level_error_mm=volume_error / cross_section if cross_section else float("nan"),
        residual_rms_mm=float(np.sqrt(np.mean(residuals**2))) if residuals.size else float("nan"),
        residual_bias_mm=float(np.mean(residuals)) if residuals.size else float("nan"),
        residual_max_abs_mm=float(np.max(np.abs(residuals))) if residuals.size else float("nan"),
        residual_p95_mm=float(np.percentile(np.abs(residuals), 95)) if residuals.size else float("nan"),
        sample_offset_rms_mm=float(np.sqrt(np.mean(offsets**2))) if offsets.size else float("nan"),
        sample_offset_max_mm=float(np.max(np.abs(offsets))) if offsets.size else float("nan"),
        feed_zone_count=int(np.count_nonzero(reconstruction.feed_mask)),
        valid_zone_count=int(np.count_nonzero(frame.valid)),
        zone_count=int(frame.valid.size),
        field=rf if keep_field else None,
    )


def _sample_offsets(reconstruction: Reconstruction, feed: HeightField) -> np.ndarray:
    """Vertical gap between each unprojected sample and the true surface below it.

    With matching intrinsics this collapses to the range-quantisation floor, so a large value
    means the assumed camera model is wrong rather than the surface fit.
    """
    points = reconstruction.points_world[reconstruction.feed_mask]
    if len(points) == 0:
        return np.array([])
    z_true = feed.height(points[:, 0], points[:, 1])
    offsets = points[:, 2] - z_true
    return offsets[np.isfinite(offsets)]
