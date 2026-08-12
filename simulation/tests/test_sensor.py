import numpy as np
import pytest

from tof_sim.camera import Intrinsics, look_at
from tof_sim.geometry import FlatFeed, Scene, SiloProfile
from tof_sim.sensor import DEVICE_LSB_MM, IdealTofSensor

SILO_RADIUS = 1500.0
SILO_HEIGHT = 4000.0
FEED_LEVEL = 2000.0


PROFILE = SiloProfile.cylindrical(radius=SILO_RADIUS, height=SILO_HEIGHT)


def _scene(feed_level=FEED_LEVEL):
    scene = Scene()
    for name, surface, role in PROFILE.surfaces():
        scene.add(name, surface, role=role)
    return scene.add("feed", FlatFeed(level=feed_level, radius=SILO_RADIUS), role="feed")


def _sensor(**kwargs):
    intr = Intrinsics.from_fov(8, 8, 67.38, 53.13)
    pose = look_at(position=(0.0, 0.0, SILO_HEIGHT), target=(0.0, 0.0, 0.0))
    return IdealTofSensor(intr, pose, **kwargs)


def test_flat_surface_ranges_follow_secant_law():
    sensor = _sensor(quantization_mm=0.0)
    frame = sensor.capture(_scene())

    drop = SILO_HEIGHT - FEED_LEVEL
    dirs_cam = sensor.intrinsics.ray_directions()
    expected = drop / np.abs(dirs_cam[..., 2])

    assert frame.valid.all()
    assert frame.distance_mm == pytest.approx(expected)


def test_all_zones_land_on_the_feed_surface():
    scene = _scene()
    frame = _sensor().capture(scene)
    feed_id = scene.index_of_role("feed")[0]
    assert (frame.surface_id == feed_id).all()


def test_zones_beyond_the_feed_hit_the_wall():
    scene = _scene(feed_level=200.0)
    frame = _sensor().capture(scene)
    names = np.array(scene.names)[frame.surface_id]
    assert set(np.unique(names)) == {"wall", "feed"}


def test_quantisation_snaps_to_device_lsb():
    frame = _sensor(quantization_mm=DEVICE_LSB_MM).capture(_scene())
    steps = frame.distance_mm[frame.valid] / DEVICE_LSB_MM
    assert steps == pytest.approx(np.round(steps))

    exact = _sensor(quantization_mm=0.0).capture(_scene())
    error = frame.distance_mm[frame.valid] - exact.distance_mm[exact.valid]
    assert np.max(np.abs(error)) <= DEVICE_LSB_MM / 2 + 1e-9


def test_range_limits_invalidate_zones():
    frame = _sensor(max_range_mm=2100.0).capture(_scene())
    assert not frame.valid.all()
    assert frame.valid[3:5, 3:5].all()
    assert np.isnan(frame.distance_mm[~frame.valid]).all()


def test_incidence_angle_is_zero_at_nadir_on_a_flat_surface():
    intr = Intrinsics.from_fov(1, 1, 40.0, 40.0)
    pose = look_at(position=(0.0, 0.0, SILO_HEIGHT), target=(0.0, 0.0, 0.0))
    frame = IdealTofSensor(intr, pose).capture(_scene())
    assert frame.incidence_deg[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert frame.distance_mm[0, 0] == pytest.approx(SILO_HEIGHT - FEED_LEVEL)
