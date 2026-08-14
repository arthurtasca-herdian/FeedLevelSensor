import json
from pathlib import Path

import numpy as np
import pytest

from tof_sim.camera import Intrinsics, from_euler, position_of
from tof_sim.geometry import Scene, SiloProfile
from tof_sim.reading import Reading, frame_from_reading, load_reading
from tof_sim.reconstruct import check_is_contained_in_silo, classify_oracle, reconstruct

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

INTRINSICS = Intrinsics.from_vendor_zcorrection(8, 8)
MOUNT_Z = 5000.0
POSE = from_euler((0.0, 0.0, MOUNT_Z), pitch_deg=180.0)
# Wide enough that every zone of a flat 3 m drop lands clear of the shell, so a rejection in the
# classification test can only come from the ray the test lengthened.
PROFILE = SiloProfile.cylindrical(radius=3000.0, height=6000.0)


def _radial_for_flat_surface(level_mm: float) -> np.ndarray:
    # The device reports slant range, so a flat surface is not a constant reading: each zone's
    # value is the drop divided by the z-component of its own unit ray.
    dirs = INTRINSICS.ray_directions()
    return (MOUNT_Z - level_mm) / dirs[..., 2]


def _reading(distances: np.ndarray, **meta) -> Reading:
    base = {"unit": "mm", "label": "bench", "seq": 1, "integrity_ok": True}
    return Reading(
        distances_mm=distances,
        signal=np.full(distances.shape, 500.0),
        snr=np.full(distances.shape, 30.0),
        meta={**base, **meta},
    )


def _write_npz(path: Path, distances: np.ndarray, **meta) -> Path:
    reading = _reading(distances, **meta)
    np.savez(path, distances_mm=reading.distances_mm, signal=reading.signal, snr=reading.snr,
             meta=json.dumps(reading.meta))
    return path


def test_npz_round_trips_through_load_reading(tmp_path):
    distances = _radial_for_flat_surface(2000.0)
    loaded = load_reading(_write_npz(tmp_path / "r.npz", distances, seq=7))

    assert loaded.shape == (8, 8)
    assert loaded.distances_mm == pytest.approx(distances)
    assert loaded.meta["seq"] == 7
    assert loaded.label == "bench"


def test_a_flat_surface_unprojects_to_one_world_height():
    frame = frame_from_reading(_reading(_radial_for_flat_surface(2000.0)), INTRINSICS, POSE)
    recon = reconstruct(frame, method="linear_interp")

    assert frame.origin == pytest.approx(position_of(POSE))
    assert recon.points_world[..., 2] == pytest.approx(2000.0)


def test_missing_targets_stay_out_of_the_fit():
    distances = _radial_for_flat_surface(2000.0)
    distances[3, 3] = np.nan
    distances[0, 7] = np.nan

    frame = frame_from_reading(_reading(distances), INTRINSICS, POSE)
    assert not frame.valid[3, 3]
    assert not frame.valid[0, 7]
    assert frame.valid.sum() == 62
    assert len(reconstruct(frame, method="linear_interp").feed_points) == 62


def test_range_gate_drops_zones_outside_the_configured_band():
    distances = _radial_for_flat_surface(2000.0)
    distances[0, 0] = 20.0
    distances[7, 7] = 40_000.0

    frame = frame_from_reading(_reading(distances), INTRINSICS, POSE,
                               min_range_mm=200.0, max_range_mm=10_000.0)

    assert not frame.valid[0, 0]
    assert not frame.valid[7, 7]
    assert np.isnan(frame.distance_mm[0, 0])
    assert frame.valid.sum() == 62


def test_a_grid_the_camera_is_not_configured_for_is_refused():
    with pytest.raises(ValueError, match="misplaced"):
        frame_from_reading(_reading(np.full((16, 16), 3000.0)), INTRINSICS, POSE)


def test_a_reading_in_device_units_is_refused(tmp_path):
    path = _write_npz(tmp_path / "bins.npz", np.full((8, 8), 3000.0), unit="bins")
    with pytest.raises(ValueError, match="not millimetres"):
        load_reading(path)


def test_distances_beyond_the_device_range_are_refused(tmp_path):
    path = _write_npz(tmp_path / "huge.npz", np.full((8, 8), 90_000.0))
    with pytest.raises(ValueError, match="exceed the device"):
        load_reading(path)


def test_oracle_classification_refuses_a_measured_frame():
    frame = frame_from_reading(_reading(_radial_for_flat_surface(2000.0)), INTRINSICS, POSE)
    scene = Scene()
    for name, surface, role in PROFILE.surfaces():
        scene.add(name, surface, role=role)

    with pytest.raises(ValueError, match="no surface labels"):
        classify_oracle(frame, scene)


def test_geometric_classification_rejects_returns_off_the_silo_wall():
    distances = _radial_for_flat_surface(2000.0)
    # A ray that ran past the feed and struck the shell reads much longer than its neighbours.
    distances[0, 0] *= 1.6

    frame = frame_from_reading(_reading(distances), INTRINSICS, POSE)
    points = reconstruct(frame, method="linear_interp").points_world
    mask = check_is_contained_in_silo(points, frame.valid, profile=PROFILE, wall_margin=25.0)

    assert not mask[0, 0]
    assert mask.sum() == 63


def test_cli_writes_the_reading_views(tmp_path):
    from scripts.run_reading import main

    reading = _write_npz(tmp_path / "r.npz", _radial_for_flat_surface(2000.0))
    out = tmp_path / "run"
    assert main([
        "--config", str(CONFIGS / "field_capture.template.yaml"),
        "--reading", str(reading), "--out", str(out),
    ]) == 0

    for name in ("distance_map.png", "reading.png", "reading.html"):
        assert (out / name).exists(), name
    # A capture has no ground truth, so scoring it would be inventing a reference.
    assert not (out / "metrics.json").exists()
