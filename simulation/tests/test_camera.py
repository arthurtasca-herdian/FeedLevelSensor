import sys
from pathlib import Path

import numpy as np
import pytest

from tof_sim.camera import Intrinsics, look_at, position_of, rotation_of, to_world, unproject

_DRIVER = Path(__file__).resolve().parents[2] / "python-poc" / "driver" / "tmf8829"


def test_fov_round_trip():
    intr = Intrinsics.from_fov(8, 8, hfov_deg=60.0, vfov_deg=45.0)
    assert intr.hfov_deg == pytest.approx(60.0)
    assert intr.vfov_deg == pytest.approx(45.0)


def test_vendor_intrinsics_have_resolution_independent_fov():
    for width, height in ((8, 8), (16, 16), (32, 32), (48, 32)):
        intr = Intrinsics.from_vendor_zcorrection(width, height)
        assert intr.hfov_deg == pytest.approx(np.degrees(2 * np.arctan(2 / 3)))
        assert intr.vfov_deg == pytest.approx(np.degrees(2 * np.arctan(1 / 2)))


@pytest.mark.parametrize("fp_mode,width,height", [(0, 8, 8), (2, 16, 16), (3, 32, 32), (5, 48, 32)])
def test_vendor_intrinsics_match_driver_zcorrection(fp_mode, width, height):
    if not _DRIVER.exists():
        pytest.skip("vendor driver not available")
    sys.path.insert(0, str(_DRIVER))
    from tmf8829_application_common import Tmf8829AppCommon

    intr = Intrinsics.from_vendor_zcorrection(width, height)
    dirs = intr.ray_directions()

    for row in range(height):
        for col in range(width):
            z_corr, x, y = Tmf8829AppCommon.zCorrection(
                pixel_x=col, pixel_y=row, fp_mode=fp_mode, getxy=True
            )
            d = dirs[row, col]
            assert d[0] / d[2] == pytest.approx(x)
            assert d[1] / d[2] == pytest.approx(y)
            assert 1.0 / d[2] == pytest.approx(z_corr)


def test_ray_directions_are_unit_and_symmetric():
    intr = Intrinsics.from_fov(8, 8, 60.0, 45.0)
    dirs = intr.ray_directions()
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)
    assert dirs[:, ::-1, 0] == pytest.approx(-dirs[:, :, 0])
    assert dirs[::-1, :, 1] == pytest.approx(-dirs[:, :, 1])


def test_look_at_straight_down():
    pose = look_at(position=(0.0, 0.0, 5000.0), target=(0.0, 0.0, 0.0))
    r = rotation_of(pose)
    assert r[:, 2] == pytest.approx([0.0, 0.0, -1.0])
    assert np.linalg.det(r) == pytest.approx(1.0)
    assert r.T @ r == pytest.approx(np.eye(3))
    assert position_of(pose) == pytest.approx([0.0, 0.0, 5000.0])


def test_unproject_inverts_projection():
    intr = Intrinsics.from_fov(8, 8, 60.0, 45.0)
    pose = look_at((100.0, -200.0, 4000.0), (0.0, 0.0, 0.0))

    dirs = intr.ray_directions()
    distances = np.full((8, 8), 3210.0)
    world = unproject(distances, intr, pose)

    cam = (world - position_of(pose)) @ rotation_of(pose)
    assert np.linalg.norm(cam, axis=-1) == pytest.approx(distances)

    pixels = intr.project(cam)
    cols, rows = np.meshgrid(np.arange(8), np.arange(8), indexing="xy")
    assert pixels[..., 0] == pytest.approx(cols + 0.5)
    assert pixels[..., 1] == pytest.approx(rows + 0.5)


def test_to_world_matches_manual_transform():
    pose = look_at((0.0, 0.0, 1000.0), (500.0, 0.0, 0.0))
    p_cam = np.array([[0.0, 0.0, 250.0]])
    assert to_world(pose, p_cam)[0] == pytest.approx(
        position_of(pose) + 250.0 * rotation_of(pose)[:, 2]
    )
