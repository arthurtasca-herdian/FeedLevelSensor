"""Analytic scene primitives with vectorised ray intersection.

Every surface exposes ``intersect(origins, directions) -> Hit`` over a bundle of rays and
``triangulate() -> (vertices, faces)`` for visualisation only. Silo shells are treated as
vertical: the axis is parallel to world +Z, which is what the primitives assume.

Closed-form intersection is used wherever the shape allows it so that the ground truth
carries no tessellation error, leaving reconstruction as the only source of error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12

# Rays are nudged off their own origin so a surface cannot self-intersect at t=0.
_T_MIN = 1e-6

# A ray grazing a cone apex or cylinder wall makes b**2 and 4ac cancel, so the
# discriminant lands slightly negative and the tangent hit would be dropped.
_DISCRIMINANT_REL_TOL = 1e-12


def _discriminant(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    disc = b * b - 4.0 * a * c
    tol = _DISCRIMINANT_REL_TOL * (b * b + np.abs(4.0 * a * c))
    return disc, disc >= -tol


@dataclass
class Hit:
    t: np.ndarray
    hit: np.ndarray
    normal: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "Hit":
        return cls(
            t=np.full(n, np.inf),
            hit=np.zeros(n, dtype=bool),
            normal=np.zeros((n, 3)),
        )


def _orient_against(normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    facing = np.sum(normals * directions, axis=-1, keepdims=True)
    return np.where(facing > 0.0, -normals, normals)


def _select_nearest(candidates: list[tuple[np.ndarray, np.ndarray]], n: int) -> tuple[np.ndarray, np.ndarray]:
    best_t = np.full(n, np.inf)
    best_hit = np.zeros(n, dtype=bool)
    for t, valid in candidates:
        take = valid & (t < best_t)
        best_t = np.where(take, t, best_t)
        best_hit |= take
    return best_t, best_hit


class Surface(ABC):
    @abstractmethod
    def intersect(self, origins: np.ndarray, directions: np.ndarray) -> Hit: ...

    @abstractmethod
    def triangulate(self) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class Plane(Surface):
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    viz_extent: float = 1000.0

    def __post_init__(self):
        self._p0 = np.asarray(self.point, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        self._n = n / np.linalg.norm(n)

    def intersect(self, origins, directions):
        denom = directions @ self._n
        parallel = np.abs(denom) < _EPS
        safe = np.where(parallel, 1.0, denom)
        t = ((self._p0 - origins) @ self._n) / safe
        valid = ~parallel & (t > _T_MIN)
        normal = _orient_against(np.broadcast_to(self._n, directions.shape), directions)
        return Hit(t=np.where(valid, t, np.inf), hit=valid, normal=normal)

    def triangulate(self):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(self._n @ axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        u = np.cross(self._n, axis)
        u /= np.linalg.norm(u)
        v = np.cross(self._n, u)
        h = self.viz_extent / 2.0
        verts = np.array([
            self._p0 - h * u - h * v,
            self._p0 + h * u - h * v,
            self._p0 + h * u + h * v,
            self._p0 - h * u + h * v,
        ])
        return verts, np.array([[0, 1, 2], [0, 2, 3]])


@dataclass
class Disc(Surface):
    """Flat annulus normal to +Z; ``inner_radius`` > 0 makes it a ring."""

    z: float = 0.0
    radius: float = 1000.0
    centre: tuple[float, float] = (0.0, 0.0)
    inner_radius: float = 0.0
    segments: int = 96

    def __post_init__(self):
        self._c = np.asarray(self.centre, dtype=float)

    def intersect(self, origins, directions):
        dz = directions[:, 2]
        parallel = np.abs(dz) < _EPS
        safe = np.where(parallel, 1.0, dz)
        t = (self.z - origins[:, 2]) / safe
        xy = origins[:, :2] + t[:, None] * directions[:, :2] - self._c
        r = np.linalg.norm(xy, axis=1)
        valid = ~parallel & (t > _T_MIN) & (r <= self.radius) & (r >= self.inner_radius)
        normal = _orient_against(
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), directions.shape), directions
        )
        return Hit(t=np.where(valid, t, np.inf), hit=valid, normal=normal)

    def triangulate(self):
        theta = np.linspace(0.0, 2.0 * np.pi, self.segments, endpoint=False)
        outer = np.stack(
            [self._c[0] + self.radius * np.cos(theta),
             self._c[1] + self.radius * np.sin(theta),
             np.full(self.segments, self.z)], axis=1)
        if self.inner_radius <= 0.0:
            verts = np.vstack([np.array([[self._c[0], self._c[1], self.z]]), outer])
            idx = np.arange(self.segments)
            faces = np.stack([np.zeros(self.segments, dtype=int), idx + 1,
                              (idx + 1) % self.segments + 1], axis=1)
            return verts, faces
        inner = np.stack(
            [self._c[0] + self.inner_radius * np.cos(theta),
             self._c[1] + self.inner_radius * np.sin(theta),
             np.full(self.segments, self.z)], axis=1)
        return _ring_mesh(inner, outer)


@dataclass
class Cylinder(Surface):
    """Vertical tube, open at both ends; rays from inside hit its wall."""

    radius: float = 1000.0
    z_min: float = 0.0
    z_max: float = 3000.0
    centre: tuple[float, float] = (0.0, 0.0)
    segments: int = 96

    def __post_init__(self):
        self._c = np.asarray(self.centre, dtype=float)

    def intersect(self, origins, directions):
        n = origins.shape[0]
        oxy = origins[:, :2] - self._c
        dxy = directions[:, :2]
        a = np.sum(dxy * dxy, axis=1)
        b = 2.0 * np.sum(oxy * dxy, axis=1)
        c = np.sum(oxy * oxy, axis=1) - self.radius**2

        disc, resolvable = _discriminant(a, b, c)
        solvable = (a > _EPS) & resolvable
        sq = np.sqrt(np.maximum(np.where(solvable, disc, 0.0), 0.0))
        a_safe = np.where(solvable, a, 1.0)

        candidates = []
        for root in ((-b - sq) / (2.0 * a_safe), (-b + sq) / (2.0 * a_safe)):
            z = origins[:, 2] + root * directions[:, 2]
            candidates.append(
                (root, solvable & (root > _T_MIN) & (z >= self.z_min) & (z <= self.z_max))
            )
        t, hit = _select_nearest(candidates, n)

        point_xy = origins[:, :2] + np.where(hit, t, 0.0)[:, None] * dxy - self._c
        radial = np.zeros((n, 3))
        radial[:, :2] = point_xy / np.maximum(
            np.linalg.norm(point_xy, axis=1, keepdims=True), _EPS
        )
        return Hit(t=t, hit=hit, normal=_orient_against(radial, directions))

    def triangulate(self):
        theta = np.linspace(0.0, 2.0 * np.pi, self.segments, endpoint=False)
        lower = np.stack([self._c[0] + self.radius * np.cos(theta),
                          self._c[1] + self.radius * np.sin(theta),
                          np.full(self.segments, self.z_min)], axis=1)
        upper = lower.copy()
        upper[:, 2] = self.z_max
        return _ring_mesh(lower, upper)


@dataclass
class Frustum(Surface):
    """Vertical truncated cone; ``radius_top`` = 0 gives a cone with its apex on top."""

    radius_bottom: float = 1000.0
    radius_top: float = 0.0
    z_min: float = 0.0
    z_max: float = 1000.0
    centre: tuple[float, float] = (0.0, 0.0)
    segments: int = 96

    def __post_init__(self):
        self._c = np.asarray(self.centre, dtype=float)
        self._k = (self.radius_top - self.radius_bottom) / (self.z_max - self.z_min)

    def intersect(self, origins, directions):
        n = origins.shape[0]
        k = self._k
        oxy = origins[:, :2] - self._c
        dxy = directions[:, :2]
        dz = directions[:, 2]
        w0 = self.radius_bottom + k * (origins[:, 2] - self.z_min)

        a = np.sum(dxy * dxy, axis=1) - (k * dz) ** 2
        b = 2.0 * (np.sum(oxy * dxy, axis=1) - k * dz * w0)
        c = np.sum(oxy * oxy, axis=1) - w0**2

        quadratic = np.abs(a) > _EPS
        disc, resolvable = _discriminant(a, b, c)
        solvable = quadratic & resolvable
        sq = np.sqrt(np.maximum(np.where(solvable, disc, 0.0), 0.0))
        a_safe = np.where(quadratic, a, 1.0)
        b_safe = np.where(np.abs(b) > _EPS, b, 1.0)

        roots = [
            ((-b - sq) / (2.0 * a_safe), solvable),
            ((-b + sq) / (2.0 * a_safe), solvable),
            (-c / b_safe, ~quadratic & (np.abs(b) > _EPS)),
        ]
        candidates = []
        for root, ok in roots:
            z = origins[:, 2] + root * dz
            radius_at_z = w0 + k * root * dz
            candidates.append(
                (root, ok & (root > _T_MIN) & (z >= self.z_min) & (z <= self.z_max)
                 & (radius_at_z >= 0.0))
            )
        t, hit = _select_nearest(candidates, n)

        point_xy = origins[:, :2] + np.where(hit, t, 0.0)[:, None] * dxy - self._c
        u = point_xy / np.maximum(np.linalg.norm(point_xy, axis=1, keepdims=True), _EPS)
        normal = np.concatenate([u, np.full((n, 1), -k)], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        return Hit(t=t, hit=hit, normal=_orient_against(normal, directions))

    def triangulate(self):
        theta = np.linspace(0.0, 2.0 * np.pi, self.segments, endpoint=False)
        lower = np.stack([self._c[0] + self.radius_bottom * np.cos(theta),
                          self._c[1] + self.radius_bottom * np.sin(theta),
                          np.full(self.segments, self.z_min)], axis=1)
        upper = np.stack([self._c[0] + self.radius_top * np.cos(theta),
                          self._c[1] + self.radius_top * np.sin(theta),
                          np.full(self.segments, self.z_max)], axis=1)
        return _ring_mesh(lower, upper)


class HeightField(Surface):
    """Feed surface expressed as z = height(x, y) over a circular domain.

    The generic intersection marches along each ray for a sign change in
    ``z_ray - height`` and then bisects, so any callable height works. Subclasses whose
    shape has a closed form override ``intersect`` and keep the ground truth exact.
    """

    def __init__(self, radius: float, centre=(0.0, 0.0), max_range: float = 100_000.0,
                 march_steps: int = 512, bisect_steps: int = 50, rings: int = 64,
                 sectors: int = 96, clip_profile: "SiloProfile | None" = None):
        self.radius = float(radius)
        self.centre = np.asarray(centre, dtype=float)
        self.max_range = float(max_range)
        self.march_steps = int(march_steps)
        self.bisect_steps = int(bisect_steps)
        self.rings = int(rings)
        self.sectors = int(sectors)
        # Ray casting needs no clipping - a surface extending past the wall is occluded by it -
        # but a mesh of that surface would visibly poke through the silo.
        self.clip_profile = clip_profile

    @abstractmethod
    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Surface elevation at (x, y); NaN outside the domain."""

    def in_domain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.hypot(x - self.centre[0], y - self.centre[1]) <= self.radius

    def _gap(self, origins, directions, t):
        p = origins + t[..., None] * directions
        h = self.height(p[..., 0], p[..., 1])
        return p[..., 2] - h

    def intersect(self, origins, directions):
        n = origins.shape[0]
        ts = np.linspace(_T_MIN, self.max_range, self.march_steps)
        gaps = self._gap(origins[:, None, :], directions[:, None, :],
                         np.broadcast_to(ts, (n, self.march_steps)))

        above = gaps > 0.0
        crossing = above[:, :-1] & ~above[:, 1:] & np.isfinite(gaps[:, :-1]) & np.isfinite(gaps[:, 1:])
        found = crossing.any(axis=1)
        first = np.argmax(crossing, axis=1)

        lo = np.where(found, ts[first], np.inf)
        hi = np.where(found, ts[np.minimum(first + 1, self.march_steps - 1)], np.inf)
        lo_safe = np.where(found, lo, 0.0)
        hi_safe = np.where(found, hi, 1.0)
        for _ in range(self.bisect_steps):
            mid = 0.5 * (lo_safe + hi_safe)
            g = self._gap(origins, directions, mid)
            still_above = np.nan_to_num(g, nan=-1.0) > 0.0
            lo_safe = np.where(still_above, mid, lo_safe)
            hi_safe = np.where(still_above, hi_safe, mid)
        t = np.where(found, 0.5 * (lo_safe + hi_safe), np.inf)

        return Hit(t=t, hit=found, normal=_orient_against(self._normals(origins, directions, t, found), directions))

    def _normals(self, origins, directions, t, found):
        step = max(self.radius * 1e-4, 1e-3)
        p = origins + np.where(found, t, 0.0)[:, None] * directions
        x, y = p[:, 0], p[:, 1]
        dzdx = (self.height(x + step, y) - self.height(x - step, y)) / (2.0 * step)
        dzdy = (self.height(x, y + step) - self.height(x, y - step)) / (2.0 * step)
        normal = np.stack([-np.nan_to_num(dzdx), -np.nan_to_num(dzdy), np.ones_like(x)], axis=1)
        return normal / np.linalg.norm(normal, axis=1, keepdims=True)

    def triangulate(self):
        # The outermost ring would otherwise round to just outside the domain and evaluate
        # to NaN, dropping those vertices to z=0 as spikes.
        r = np.linspace(0.0, self.radius * (1.0 - 1e-9), self.rings)
        theta = np.linspace(0.0, 2.0 * np.pi, self.sectors, endpoint=False)
        rr, tt = np.meshgrid(r, theta, indexing="ij")
        x = self.centre[0] + rr * np.cos(tt)
        y = self.centre[1] + rr * np.sin(tt)
        z = self.height(x, y)
        verts = np.stack([x.ravel(), y.ravel(), np.nan_to_num(z).ravel()], axis=1)

        usable = np.isfinite(z).ravel()
        if self.clip_profile is not None:
            usable &= self.clip_profile.contains(np.hypot(x, y).ravel(), verts[:, 2])

        faces = _polar_faces(self.rings, self.sectors)
        return verts, faces[usable[faces].all(axis=1)]


class FlatFeed(HeightField):
    def __init__(self, level: float, radius: float, centre=(0.0, 0.0), **kwargs):
        super().__init__(radius=radius, centre=centre, **kwargs)
        self.level = float(level)
        self._disc = Disc(z=self.level, radius=self.radius, centre=tuple(self.centre))

    def height(self, x, y):
        z = np.full(np.broadcast(x, y).shape, self.level, dtype=float)
        return np.where(self.in_domain(np.asarray(x), np.asarray(y)), z, np.nan)

    def intersect(self, origins, directions):
        return self._disc.intersect(origins, directions)


class ConicalPileFeed(HeightField):
    """Right circular pile of half-angle set by ``peak_height``/``pile_radius``, on a flat base."""

    def __init__(self, base_level: float, peak_height: float, pile_radius: float,
                 radius: float, centre=(0.0, 0.0), pile_centre=None, **kwargs):
        super().__init__(radius=radius, centre=centre, **kwargs)
        self.base_level = float(base_level)
        self.peak_height = float(peak_height)
        self.pile_radius = float(pile_radius)
        self.pile_centre = np.asarray(
            self.centre if pile_centre is None else pile_centre, dtype=float
        )
        self._cone = Frustum(
            radius_bottom=self.pile_radius, radius_top=0.0,
            z_min=self.base_level, z_max=self.base_level + self.peak_height,
            centre=tuple(self.pile_centre),
        )
        self._apron = Disc(
            z=self.base_level, radius=self.radius, centre=tuple(self.centre),
            inner_radius=0.0,
        )

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.hypot(x - self.pile_centre[0], y - self.pile_centre[1])
        cone = self.base_level + self.peak_height * (1.0 - r / self.pile_radius)
        z = np.where(r <= self.pile_radius, cone, self.base_level)
        return np.where(self.in_domain(x, y), z, np.nan)

    def intersect(self, origins, directions):
        n = origins.shape[0]
        cone = self._cone.intersect(origins, directions)
        apron = self._apron.intersect(origins, directions)

        # The apron plane extends under the pile, so hits inside the footprint are discarded.
        p = origins + np.where(apron.hit, apron.t, 0.0)[:, None] * directions
        under_pile = np.hypot(p[:, 0] - self.pile_centre[0], p[:, 1] - self.pile_centre[1]) < self.pile_radius
        apron_ok = apron.hit & ~under_pile

        t, hit = _select_nearest(
            [(cone.t, cone.hit), (np.where(apron_ok, apron.t, np.inf), apron_ok)], n
        )
        take_cone = cone.hit & (cone.t <= np.where(apron_ok, apron.t, np.inf))
        normal = np.where(take_cone[:, None], cone.normal, apron.normal)
        return Hit(t=t, hit=hit, normal=normal)

    def pile_volume(self) -> float:
        """Volume of the cone above ``base_level``; the silo below it is the profile's business."""
        return float(np.pi * self.pile_radius**2 * self.peak_height / 3.0)


class CallableHeightField(HeightField):
    def __init__(self, func, radius: float, centre=(0.0, 0.0), **kwargs):
        super().__init__(radius=radius, centre=centre, **kwargs)
        self.func = func

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.where(self.in_domain(x, y), self.func(x, y), np.nan)


class TriMeshSurface(Surface):
    """Fallback for imported geometry; wraps trimesh's ray engine."""

    def __init__(self, mesh):
        import trimesh

        self.mesh = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.load(mesh, force="mesh")

    def intersect(self, origins, directions):
        n = origins.shape[0]
        result = Hit.empty(n)
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False
        )
        if len(index_ray) == 0:
            return result
        t = np.linalg.norm(locations - origins[index_ray], axis=1)
        result.t[index_ray] = t
        result.hit[index_ray] = True
        result.normal[index_ray] = self.mesh.face_normals[index_tri]
        result.normal = _orient_against(result.normal, directions)
        return result

    def triangulate(self):
        return np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces)


@dataclass
class SiloProfile:
    """Silo body as a solid of revolution: hopper, cylindrical shell, roof cone.

    Radius varies with height, so feed volume is not ``pi r^2 h`` and a wall return is not
    identified by radius alone. Everything that needs the silo's shape goes through this:
    surface construction, volume integration, feed/wall classification and mesh clipping.
    The lid is not modelled - the roof is open at ``z_roof_top``.
    """

    outlet_radius: float
    body_radius: float
    roof_radius: float
    z_outlet: float
    z_hopper_top: float
    z_cylinder_top: float
    z_roof_top: float

    def __post_init__(self):
        heights = (self.z_outlet, self.z_hopper_top, self.z_cylinder_top, self.z_roof_top)
        if list(heights) != sorted(heights):
            raise ValueError("silo section heights must be non-decreasing from outlet to roof")
        if self.outlet_radius > self.body_radius or self.roof_radius > self.body_radius:
            raise ValueError("outlet and roof radii cannot exceed the body radius")
        if min(self.outlet_radius, self.roof_radius) < 0.0 or self.body_radius <= 0.0:
            raise ValueError("silo radii must be positive")

    @classmethod
    def cylindrical(cls, radius: float, height: float, floor_z: float = 0.0) -> "SiloProfile":
        return cls(
            outlet_radius=radius, body_radius=radius, roof_radius=radius,
            z_outlet=floor_z, z_hopper_top=floor_z,
            z_cylinder_top=floor_z + height, z_roof_top=floor_z + height,
        )

    def radius_at(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        hopper = _lerp(z, self.z_outlet, self.z_hopper_top, self.outlet_radius, self.body_radius)
        roof = _lerp(z, self.z_cylinder_top, self.z_roof_top, self.body_radius, self.roof_radius)

        r = np.full(z.shape, self.body_radius, dtype=float)
        r = np.where(z <= self.z_hopper_top, hopper, r)
        r = np.where(z >= self.z_cylinder_top, roof, r)
        return np.where((z < self.z_outlet) | (z > self.z_roof_top), 0.0, r)

    def z_lower(self, rho) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        on_hopper = _lerp(rho, self.outlet_radius, self.body_radius, self.z_outlet, self.z_hopper_top)
        z = np.where(rho <= self.outlet_radius, self.z_outlet, on_hopper)
        return np.where(rho > self.body_radius, np.nan, z)

    def z_upper(self, rho) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        on_roof = _lerp(rho, self.body_radius, self.roof_radius, self.z_cylinder_top, self.z_roof_top)
        z = np.where(rho <= self.roof_radius, self.z_roof_top, on_roof)
        return np.where(rho > self.body_radius, np.nan, z)

    def contains(self, rho, z, tolerance: float = 1e-6) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)
        with np.errstate(invalid="ignore"):
            inside = (z >= self.z_lower(rho) - tolerance) & (z <= self.z_upper(rho) + tolerance)
        return np.nan_to_num(inside, nan=0.0).astype(bool) & (rho <= self.body_radius)

    def volume_below_level(self, level: float) -> float:
        """Exact volume of the silo interior below a horizontal plane."""
        level = float(np.clip(level, self.z_outlet, self.z_roof_top))
        total = 0.0

        top = min(level, self.z_hopper_top)
        if top > self.z_outlet:
            total += _frustum_volume(
                self.outlet_radius, float(self.radius_at(top)), top - self.z_outlet
            )

        top = min(level, self.z_cylinder_top)
        if top > self.z_hopper_top:
            total += np.pi * self.body_radius**2 * (top - self.z_hopper_top)

        if level > self.z_cylinder_top:
            total += _frustum_volume(
                self.body_radius, float(self.radius_at(level)), level - self.z_cylinder_top
            )
        return float(total)

    def capacity(self) -> float:
        return self.volume_below_level(self.z_roof_top)

    def surfaces(self) -> list[tuple[str, Surface, str]]:
        built: list[tuple[str, Surface, str]] = [
            ("outlet", Disc(z=self.z_outlet, radius=self.outlet_radius), "floor")
        ]
        if self.z_hopper_top > self.z_outlet:
            built.append(("hopper", Frustum(
                radius_bottom=self.outlet_radius, radius_top=self.body_radius,
                z_min=self.z_outlet, z_max=self.z_hopper_top), "wall"))
        if self.z_cylinder_top > self.z_hopper_top:
            built.append(("wall", Cylinder(
                radius=self.body_radius, z_min=self.z_hopper_top,
                z_max=self.z_cylinder_top), "wall"))
        if self.z_roof_top > self.z_cylinder_top:
            built.append(("roof", Frustum(
                radius_bottom=self.body_radius, radius_top=self.roof_radius,
                z_min=self.z_cylinder_top, z_max=self.z_roof_top), "wall"))
        return built


def _lerp(t, t0, t1, v0, v1) -> np.ndarray:
    span = t1 - t0
    # A zero span is a collapsed section; a negative one is legitimate wherever the mapped
    # quantity decreases, as the roof radius does with height.
    if span == 0.0:
        return np.full(np.asarray(t, dtype=float).shape, v1, dtype=float)
    return v0 + (v1 - v0) * (np.asarray(t, dtype=float) - t0) / span


def _frustum_volume(r0: float, r1: float, height: float) -> float:
    return np.pi * height * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0


@dataclass
class SceneElement:
    name: str
    surface: Surface
    role: str = "other"


@dataclass
class Scene:
    """Collection of named surfaces; intersection keeps the nearest hit and its element index."""

    elements: list[SceneElement] = field(default_factory=list)

    def add(self, name: str, surface: Surface, role: str = "other") -> "Scene":
        self.elements.append(SceneElement(name=name, surface=surface, role=role))
        return self

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.elements]

    def index_of_role(self, role: str) -> list[int]:
        return [i for i, e in enumerate(self.elements) if e.role == role]

    def intersect(self, origins, directions):
        origins = np.asarray(origins, dtype=float)
        directions = np.asarray(directions, dtype=float)
        n = origins.shape[0]

        best_t = np.full(n, np.inf)
        best_hit = np.zeros(n, dtype=bool)
        best_normal = np.zeros((n, 3))
        best_id = np.full(n, -1, dtype=int)

        for i, element in enumerate(self.elements):
            h = element.surface.intersect(origins, directions)
            take = h.hit & (h.t < best_t)
            best_t = np.where(take, h.t, best_t)
            best_normal = np.where(take[:, None], h.normal, best_normal)
            best_id = np.where(take, i, best_id)
            best_hit |= take

        return best_t, best_hit, best_normal, best_id


def _polar_faces(rings: int, sectors: int) -> np.ndarray:
    faces = []
    for i in range(rings - 1):
        j = np.arange(sectors)
        jn = (j + 1) % sectors
        a = i * sectors + j
        b = i * sectors + jn
        c = (i + 1) * sectors + jn
        d = (i + 1) * sectors + j
        faces.append(np.stack([a, b, c], axis=1))
        faces.append(np.stack([a, c, d], axis=1))
    return np.vstack(faces)


def _ring_mesh(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(lower)
    verts = np.vstack([lower, upper])
    j = np.arange(n)
    jn = (j + 1) % n
    faces = np.vstack([
        np.stack([j, jn, jn + n], axis=1),
        np.stack([j, jn + n, j + n], axis=1),
    ])
    return verts, faces
