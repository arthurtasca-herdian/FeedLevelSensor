import numpy as np
import pytest

from tof_sim import config as config_module
from tof_sim.camera import Intrinsics, from_euler, look_at, position_of, rotation_of

AXIS_CASES = [
    (0.0, 0.0), (0.0, 12.0), (90.0, 12.0), (180.0, 7.5), (-45.0, 30.0), (135.0, 3.0),
]


def _optical_axis(pose):
    return rotation_of(pose)[:, 2]


def test_zero_angles_point_straight_down():
    pose = from_euler((400.0, -250.0, 5050.0))
    assert _optical_axis(pose) == pytest.approx([0.0, 0.0, -1.0])
    assert position_of(pose) == pytest.approx([400.0, -250.0, 5050.0])


def test_zero_angles_match_a_straight_down_look_at():
    euler = from_euler((0.0, 0.0, 5100.0))
    aimed = look_at((0.0, 0.0, 5100.0), (0.0, 0.0, 0.0))
    assert rotation_of(euler) == pytest.approx(rotation_of(aimed))


@pytest.mark.parametrize("yaw,pitch", AXIS_CASES)
def test_optical_axis_follows_the_documented_closed_form(yaw, pitch):
    pose = from_euler((0.0, 0.0, 5000.0), yaw_deg=yaw, pitch_deg=pitch)
    y, p = np.radians([yaw, pitch])
    expected = [np.sin(p) * np.cos(y), np.sin(p) * np.sin(y), -np.cos(p)]
    assert _optical_axis(pose) == pytest.approx(expected)


@pytest.mark.parametrize("yaw,pitch", AXIS_CASES)
def test_pitch_is_the_angle_away_from_vertical(yaw, pitch):
    pose = from_euler((0.0, 0.0, 5000.0), yaw_deg=yaw, pitch_deg=pitch)
    from_nadir = np.degrees(np.arccos(np.clip(-_optical_axis(pose)[2], -1.0, 1.0)))
    assert from_nadir == pytest.approx(abs(pitch))


@pytest.mark.parametrize("roll", [0.0, 30.0, 90.0, -125.0])
def test_roll_spins_the_grid_without_moving_the_axis(roll):
    upright = from_euler((0.0, 0.0, 5000.0), pitch_deg=10.0)
    rolled = from_euler((0.0, 0.0, 5000.0), pitch_deg=10.0, roll_deg=roll)

    assert _optical_axis(rolled) == pytest.approx(_optical_axis(upright))
    # arccos loses precision next to 1.0, so the angle is recovered from both components.
    x_upright, x_rolled = rotation_of(upright)[:, 0], rotation_of(rolled)[:, 0]
    angle = np.degrees(np.arctan2(
        np.cross(x_upright, x_rolled) @ _optical_axis(upright), x_upright @ x_rolled
    ))
    assert abs(angle) == pytest.approx(abs(roll), abs=1e-9)


@pytest.mark.parametrize("pose_builder", [
    lambda: from_euler((300.0, -200.0, 4800.0), yaw_deg=35.0, pitch_deg=14.0, roll_deg=22.0),
    lambda: look_at((300.0, -200.0, 4800.0), (0.0, 100.0, 1000.0), roll_deg=17.0),
])
def test_poses_stay_right_handed_and_orthonormal(pose_builder):
    r = rotation_of(pose_builder())
    assert r.T @ r == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_roll_rotates_the_zone_grid_on_the_ground():
    intr = Intrinsics.from_fov(8, 8, 67.38, 53.13)
    upright = intr.ray_directions() @ rotation_of(from_euler((0.0, 0.0, 4000.0))).T
    rolled = intr.ray_directions() @ rotation_of(from_euler((0.0, 0.0, 4000.0), roll_deg=90.0)).T

    footprint = lambda d: (d[..., :2] / -d[..., 2:3] * 4000.0).reshape(-1, 2)
    a, b = footprint(upright), footprint(rolled)

    # A 90 degree roll swaps the grid's extents, since the field of view is not square.
    assert np.ptp(a[:, 0]) == pytest.approx(np.ptp(b[:, 1]))
    assert np.ptp(a[:, 1]) == pytest.approx(np.ptp(b[:, 0]))
    assert np.ptp(a[:, 0]) != pytest.approx(np.ptp(a[:, 1]))


def test_look_at_roll_matches_euler_roll_when_pointing_down():
    aimed = look_at((0.0, 0.0, 5000.0), (0.0, 0.0, 0.0), roll_deg=25.0)
    euler = from_euler((0.0, 0.0, 5000.0), roll_deg=25.0)
    assert rotation_of(aimed) == pytest.approx(rotation_of(euler))


def test_config_placement_forms_agree_on_an_equivalent_setup():
    tilted = config_module.SensorConfig(
        placement="euler", position_mm=(0.0, 0.0, 5000.0), yaw_deg=0.0, pitch_deg=20.0
    ).pose()

    reach = 1000.0
    aim = (reach * np.sin(np.radians(20.0)), 0.0, 5000.0 - reach * np.cos(np.radians(20.0)))
    aimed = config_module.SensorConfig(placement="look_at", position_mm=(0.0, 0.0, 5000.0),
                                       target_mm=aim).pose()

    assert _optical_axis(tilted) == pytest.approx(_optical_axis(aimed))


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError):
        config_module.SensorConfig(placement="nonsense").pose()


def test_offset_tilted_config_produces_a_usable_frame():
    from pathlib import Path

    cfg = config_module.load(Path(__file__).resolve().parents[1] / "configs" / "offset_tilted.yaml")
    profile = cfg.build_profile()
    frame = cfg.build_sensor().capture(cfg.build_scene(cfg.build_feed(profile), profile))

    assert frame.valid.any()
    axis = _optical_axis(cfg.sensor.pose())
    assert axis[0] < 0.0
    assert np.degrees(np.arccos(-axis[2])) == pytest.approx(12.0)
