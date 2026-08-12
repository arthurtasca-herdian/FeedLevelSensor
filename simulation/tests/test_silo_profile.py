import numpy as np
import pytest

from tof_sim.camera import Intrinsics, look_at
from tof_sim.geometry import FlatFeed, Scene, SiloProfile
from tof_sim.metrics import evaluate
from tof_sim.reconstruct import classify_geometric, classify_oracle, reconstruct
from tof_sim.sensor import IdealTofSensor

HOPPER = SiloProfile(
    outlet_radius=250.0, body_radius=1500.0, roof_radius=350.0,
    z_outlet=0.0, z_hopper_top=1400.0, z_cylinder_top=4400.0, z_roof_top=5200.0,
)


def test_radius_follows_the_three_sections():
    assert HOPPER.radius_at(0.0) == pytest.approx(250.0)
    assert HOPPER.radius_at(700.0) == pytest.approx(875.0)
    assert HOPPER.radius_at(1400.0) == pytest.approx(1500.0)
    assert HOPPER.radius_at(3000.0) == pytest.approx(1500.0)
    assert HOPPER.radius_at(4400.0) == pytest.approx(1500.0)
    assert HOPPER.radius_at(4800.0) == pytest.approx(925.0)
    assert HOPPER.radius_at(5200.0) == pytest.approx(350.0)
    assert HOPPER.radius_at(-10.0) == pytest.approx(0.0)
    assert HOPPER.radius_at(5300.0) == pytest.approx(0.0)


def test_height_bounds_invert_the_radius():
    for z in (0.0, 700.0, 1400.0, 3000.0, 4400.0, 4800.0, 5200.0):
        rho = float(HOPPER.radius_at(z))
        assert HOPPER.z_lower(rho) <= z + 1e-6
        assert HOPPER.z_upper(rho) >= z - 1e-6

    assert HOPPER.z_lower(875.0) == pytest.approx(700.0)
    assert HOPPER.z_upper(925.0) == pytest.approx(4800.0)
    assert np.isnan(HOPPER.z_lower(1600.0))


def test_contains_rejects_points_outside_the_shell():
    assert HOPPER.contains(np.array(100.0), np.array(50.0))
    assert not HOPPER.contains(np.array(1400.0), np.array(50.0))
    assert HOPPER.contains(np.array(1400.0), np.array(3000.0))
    assert not HOPPER.contains(np.array(1400.0), np.array(5000.0))


@pytest.mark.parametrize("level", [0.0, 700.0, 1400.0, 2500.0, 4400.0, 4800.0, 5200.0])
def test_analytic_volume_matches_numeric_column_integration(level):
    feed = FlatFeed(level=level, radius=HOPPER.body_radius, clip_profile=HOPPER)
    scene = Scene()
    for name, surface, role in HOPPER.surfaces():
        scene.add(name, surface, role=role)
    scene.add("feed", feed, role="feed")

    intr = Intrinsics.from_fov(8, 8, 67.38, 53.13)
    frame = IdealTofSensor(intr, look_at((0.0, 0.0, 5100.0), (0.0, 0.0, 0.0))).capture(scene)
    recon = reconstruct(frame, feed_mask=frame.valid)
    m = evaluate(frame, recon, feed, profile=HOPPER, grid=1024)

    assert m.volume_true_mm3 == pytest.approx(HOPPER.volume_below_level(level), rel=3e-3, abs=1e6)


def test_capacity_is_the_sum_of_the_three_sections():
    hopper = np.pi * 1400.0 * (250.0**2 + 250.0 * 1500.0 + 1500.0**2) / 3.0
    barrel = np.pi * 1500.0**2 * 3000.0
    roof = np.pi * 800.0 * (1500.0**2 + 1500.0 * 350.0 + 350.0**2) / 3.0
    assert HOPPER.capacity() == pytest.approx(hopper + barrel + roof)


def test_cylindrical_profile_reduces_to_a_plain_tube():
    profile = SiloProfile.cylindrical(radius=1500.0, height=4000.0)
    assert profile.radius_at(2000.0) == pytest.approx(1500.0)
    assert profile.capacity() == pytest.approx(np.pi * 1500.0**2 * 4000.0)
    assert [name for name, _, _ in profile.surfaces()] == ["outlet", "wall"]


def test_profiled_scene_builds_all_sections():
    scene = Scene()
    for name, surface, role in HOPPER.surfaces():
        scene.add(name, surface, role=role)
    assert scene.names == ["outlet", "hopper", "wall", "roof"]


def test_a_single_radius_threshold_would_accept_the_hopper_wall():
    """The classifier must test against the wall radius at the point's own height."""
    feed = FlatFeed(level=600.0, radius=HOPPER.body_radius, clip_profile=HOPPER)
    scene = Scene()
    for name, surface, role in HOPPER.surfaces():
        scene.add(name, surface, role=role)
    scene.add("feed", feed, role="feed")

    intr = Intrinsics.from_fov(8, 8, 67.38, 53.13)
    frame = IdealTofSensor(intr, look_at((0.0, 0.0, 5100.0), (0.0, 0.0, 0.0))).capture(scene)
    points = reconstruct(frame, feed_mask=frame.valid).points_world

    oracle = classify_oracle(frame, scene)
    profiled = classify_geometric(points, frame.valid, profile=HOPPER, wall_margin=25.0)
    flat_threshold = frame.valid & (
        np.hypot(points[..., 0], points[..., 1]) <= HOPPER.body_radius - 25.0
    )

    assert oracle.sum() > 0
    assert np.array_equal(profiled, profiled & oracle)
    assert flat_threshold.sum() > profiled.sum()


def test_feed_mesh_is_clipped_to_the_silo():
    feed = FlatFeed(level=600.0, radius=HOPPER.body_radius, clip_profile=HOPPER)
    vertices, faces = feed.triangulate()
    used = np.unique(faces)
    rho = np.hypot(vertices[used, 0], vertices[used, 1])

    assert len(faces) > 0
    assert rho.max() <= float(HOPPER.radius_at(600.0)) + 1e-6
