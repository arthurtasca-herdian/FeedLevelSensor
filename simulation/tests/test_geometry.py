import numpy as np
import pytest

from tof_sim.geometry import (
    CallableHeightField,
    ConicalPileFeed,
    Cylinder,
    Disc,
    FlatFeed,
    Frustum,
    Plane,
    Scene,
    TriMeshSurface,
)


def _rays(origins, directions):
    o = np.atleast_2d(np.asarray(origins, dtype=float))
    d = np.atleast_2d(np.asarray(directions, dtype=float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.broadcast_to(o, d.shape).copy(), d


def test_plane_hit_distance_and_normal():
    o, d = _rays([[0.0, 0.0, 1000.0]], [[0.0, 0.0, -1.0]])
    h = Plane(point=(0, 0, 250.0), normal=(0, 0, 1)).intersect(o, d)
    assert h.hit[0]
    assert h.t[0] == pytest.approx(750.0)
    assert h.normal[0] == pytest.approx([0.0, 0.0, 1.0])


def test_plane_misses_when_parallel():
    o, d = _rays([[0.0, 0.0, 1000.0]], [[1.0, 0.0, 0.0]])
    assert not Plane(point=(0, 0, 250.0)).intersect(o, d).hit[0]


def test_disc_radius_and_annulus_limits():
    disc = Disc(z=0.0, radius=500.0)
    o, d = _rays([[0.0, 0.0, 1000.0]], [[0.0, 0.0, -1.0]])
    assert disc.intersect(o, d).hit[0]

    o, d = _rays([[600.0, 0.0, 1000.0]], [[0.0, 0.0, -1.0]])
    assert not disc.intersect(o, d).hit[0]

    annulus = Disc(z=0.0, radius=500.0, inner_radius=200.0)
    o, d = _rays([[100.0, 0.0, 1000.0]], [[0.0, 0.0, -1.0]])
    assert not annulus.intersect(o, d).hit[0]


def test_cylinder_hit_from_inside():
    cyl = Cylinder(radius=1000.0, z_min=0.0, z_max=3000.0)
    o, d = _rays([[0.0, 0.0, 1500.0]], [[1.0, 0.0, 0.0]])
    h = cyl.intersect(o, d)
    assert h.hit[0]
    assert h.t[0] == pytest.approx(1000.0)
    assert h.normal[0] == pytest.approx([-1.0, 0.0, 0.0])


def test_cylinder_respects_height_limits():
    cyl = Cylinder(radius=1000.0, z_min=0.0, z_max=3000.0)
    o, d = _rays([[0.0, 0.0, 3500.0]], [[1.0, 0.0, 0.0]])
    assert not cyl.intersect(o, d).hit[0]


def test_frustum_cone_apex_and_slope():
    cone = Frustum(radius_bottom=1000.0, radius_top=0.0, z_min=0.0, z_max=1000.0)
    o, d = _rays([[0.0, 0.0, 2000.0]], [[0.0, 0.0, -1.0]])
    h = cone.intersect(o, d)
    assert h.hit[0]
    assert h.t[0] == pytest.approx(1000.0)

    o, d = _rays([[500.0, 0.0, 2000.0]], [[0.0, 0.0, -1.0]])
    h = cone.intersect(o, d)
    assert h.t[0] == pytest.approx(1500.0)
    assert h.normal[0] == pytest.approx([np.sqrt(0.5), 0.0, np.sqrt(0.5)])


def test_flat_feed_volume_and_height():
    feed = FlatFeed(level=1200.0, radius=1000.0)
    assert feed.height(np.array(0.0), np.array(0.0)) == pytest.approx(1200.0)
    assert np.isnan(feed.height(np.array(2000.0), np.array(0.0)))


def test_generic_marching_matches_closed_form_plane():
    level, radius = 1200.0, 1000.0
    exact = FlatFeed(level=level, radius=radius)
    generic = CallableHeightField(
        lambda x, y: np.full(np.broadcast(x, y).shape, level), radius=radius, max_range=6000.0
    )

    rng = np.random.default_rng(0)
    origins = np.tile([0.0, 0.0, 4000.0], (64, 1))
    dirs = np.stack(
        [rng.uniform(-0.2, 0.2, 64), rng.uniform(-0.2, 0.2, 64), -np.ones(64)], axis=1
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    h_exact = exact.intersect(origins, dirs)
    h_generic = generic.intersect(origins, dirs)
    assert np.array_equal(h_exact.hit, h_generic.hit)
    assert h_generic.t[h_exact.hit] == pytest.approx(h_exact.t[h_exact.hit], abs=1e-3)


def test_conical_pile_volume_and_apex():
    feed = ConicalPileFeed(
        base_level=1000.0, peak_height=600.0, pile_radius=800.0, radius=1000.0
    )
    assert feed.height(np.array(0.0), np.array(0.0)) == pytest.approx(1600.0)
    assert feed.height(np.array(900.0), np.array(0.0)) == pytest.approx(1000.0)

    assert feed.pile_volume() == pytest.approx(np.pi * 800.0**2 * 600.0 / 3.0)


def test_conical_pile_ray_hits_apex_not_apron():
    feed = ConicalPileFeed(
        base_level=1000.0, peak_height=600.0, pile_radius=800.0, radius=1000.0
    )
    o, d = _rays([[0.0, 0.0, 4000.0]], [[0.0, 0.0, -1.0]])
    h = feed.intersect(o, d)
    assert h.hit[0]
    assert h.t[0] == pytest.approx(2400.0)

    o, d = _rays([[900.0, 0.0, 4000.0]], [[0.0, 0.0, -1.0]])
    assert feed.intersect(o, d).t[0] == pytest.approx(3000.0)


def test_conical_pile_generic_intersect_matches_closed_form():
    kwargs = dict(base_level=1000.0, peak_height=600.0, pile_radius=800.0, radius=1000.0)
    exact = ConicalPileFeed(**kwargs)
    generic = CallableHeightField(exact.height, radius=1000.0, max_range=5000.0, march_steps=2048)

    rng = np.random.default_rng(1)
    origins = np.tile([0.0, 0.0, 4000.0], (64, 1))
    dirs = np.stack(
        [rng.uniform(-0.2, 0.2, 64), rng.uniform(-0.2, 0.2, 64), -np.ones(64)], axis=1
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    h_exact = exact.intersect(origins, dirs)
    h_generic = generic.intersect(origins, dirs)
    assert h_generic.t[h_exact.hit] == pytest.approx(h_exact.t[h_exact.hit], abs=1e-2)


def test_scene_returns_nearest_surface():
    scene = (
        Scene()
        .add("floor", Disc(z=0.0, radius=1000.0), role="floor")
        .add("wall", Cylinder(radius=1000.0, z_min=0.0, z_max=3000.0), role="wall")
        .add("feed", FlatFeed(level=1200.0, radius=1000.0), role="feed")
    )
    origins = np.array([[0.0, 0.0, 2500.0], [0.0, 0.0, 2500.0]])
    dirs = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

    t, hit, _, ids = scene.intersect(origins, dirs)
    assert hit.all()
    assert t[0] == pytest.approx(1300.0)
    assert scene.names[ids[0]] == "feed"
    assert t[1] == pytest.approx(1000.0)
    assert scene.names[ids[1]] == "wall"
    assert scene.index_of_role("feed") == [2]


def test_trimesh_surface_matches_analytic_plane():
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.Trimesh(
        vertices=np.array([[-1e3, -1e3, 500.0], [1e3, -1e3, 500.0], [1e3, 1e3, 500.0], [-1e3, 1e3, 500.0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    surface = TriMeshSurface(mesh)
    o, d = _rays([[0.0, 0.0, 2000.0]], [[0.0, 0.0, -1.0]])
    h = surface.intersect(o, d)
    assert h.hit[0]
    assert h.t[0] == pytest.approx(1500.0)
