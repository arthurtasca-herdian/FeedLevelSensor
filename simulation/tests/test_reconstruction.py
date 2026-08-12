import numpy as np
import pytest

from tof_sim.camera import Intrinsics, look_at
from tof_sim.geometry import ConicalPileFeed, FlatFeed, Scene, SiloProfile
from tof_sim.metrics import evaluate
from tof_sim.reconstruct import classify_geometric, classify_oracle, reconstruct, surface_mesh
from tof_sim.sensor import IdealTofSensor

SILO_RADIUS = 1500.0
SILO_HEIGHT = 4000.0
PROFILE = SiloProfile.cylindrical(radius=SILO_RADIUS, height=SILO_HEIGHT)


def _scene(feed):
    scene = Scene()
    for name, surface, role in PROFILE.surfaces():
        scene.add(name, surface, role=role)
    return scene.add("feed", feed, role="feed")


def _flat(level):
    return FlatFeed(level=level, radius=SILO_RADIUS, clip_profile=PROFILE)


def _pile(**kwargs):
    return ConicalPileFeed(radius=SILO_RADIUS, clip_profile=PROFILE, **kwargs)


def _capture(feed, width=8, height=8, hfov=67.38, vfov=53.13, quantization_mm=0.25):
    intr = Intrinsics.from_fov(width, height, hfov, vfov)
    pose = look_at(position=(0.0, 0.0, SILO_HEIGHT), target=(0.0, 0.0, 0.0))
    scene = _scene(feed)
    frame = IdealTofSensor(intr, pose, quantization_mm=quantization_mm).capture(scene)
    return scene, frame


def test_flat_feed_reconstructs_almost_exactly():
    feed = _flat(2000.0)
    scene, frame = _capture(feed)
    recon = reconstruct(frame, method="linear_interp", feed_mask=classify_oracle(frame, scene))
    m = evaluate(frame, recon, feed, profile=PROFILE)

    assert m.volume_true_mm3 == pytest.approx(np.pi * SILO_RADIUS**2 * 2000.0, rel=1e-3)
    assert abs(m.volume_error_frac) < 1e-4
    assert abs(m.level_error_mm) < 0.2
    assert m.residual_rms_mm < 0.2
    assert m.sample_offset_max_mm < 0.3


def test_sample_offsets_expose_a_wrong_camera_model():
    feed = _flat(2000.0)
    scene, frame = _capture(feed, hfov=50.0, vfov=40.0)

    matched = reconstruct(frame, feed_mask=classify_oracle(frame, scene))
    mismatched = reconstruct(
        frame, intrinsics=Intrinsics.from_vendor_zcorrection(8, 8),
        feed_mask=classify_oracle(frame, scene),
    )

    assert evaluate(frame, matched, feed, profile=PROFILE).sample_offset_max_mm < 0.3
    assert evaluate(frame, mismatched, feed, profile=PROFILE).sample_offset_max_mm > 50.0


def test_cone_pile_true_volume_matches_closed_form():
    feed = _pile(base_level=1200.0, peak_height=800.0, pile_radius=1100.0)
    scene, frame = _capture(feed)
    recon = reconstruct(frame, feed_mask=classify_oracle(frame, scene))
    m = evaluate(frame, recon, feed, profile=PROFILE, grid=1024)

    expected = PROFILE.volume_below_level(1200.0) + feed.pile_volume()
    assert m.volume_true_mm3 == pytest.approx(expected, rel=2e-3)


def test_cone_pile_under_reads_because_the_apex_is_never_sampled():
    feed = _pile(base_level=1200.0, peak_height=800.0, pile_radius=1100.0)
    scene, frame = _capture(feed)
    recon = reconstruct(frame, feed_mask=classify_oracle(frame, scene))
    m = evaluate(frame, recon, feed, profile=PROFILE)

    assert m.volume_error_mm3 < 0.0
    assert m.residual_bias_mm < 0.0


def test_finer_grids_reduce_the_error():
    feed = _pile(base_level=1200.0, peak_height=800.0, pile_radius=1100.0)
    errors = []
    for size in (8, 16, 32):
        scene, frame = _capture(feed, width=size, height=size)
        recon = reconstruct(frame, feed_mask=classify_oracle(frame, scene))
        errors.append(abs(evaluate(frame, recon, feed, profile=PROFILE).volume_error_frac))

    assert errors[0] > errors[1] > errors[2]


def test_plane_fit_beats_mean_level_on_a_tilted_surface():
    feed = _pile(base_level=1500.0, peak_height=500.0, pile_radius=1400.0,
                 pile_centre=(600.0, 0.0))
    scene, frame = _capture(feed)
    mask = classify_oracle(frame, scene)

    scores = {}
    for method in ("mean_level", "plane_fit", "linear_interp"):
        recon = reconstruct(frame, method=method, feed_mask=mask)
        scores[method] = evaluate(frame, recon, feed, profile=PROFILE).residual_rms_mm

    assert scores["linear_interp"] < scores["plane_fit"] < scores["mean_level"]


def test_geometric_classifier_matches_oracle_when_feed_fills_the_view():
    feed = _flat(2000.0)
    scene, frame = _capture(feed)
    oracle = classify_oracle(frame, scene)
    geometric = classify_geometric(
        reconstruct(frame, feed_mask=frame.valid).points_world, frame.valid,
        profile=PROFILE, wall_margin=1.0,
    )
    assert np.array_equal(oracle, geometric)


def test_geometric_classifier_rejects_wall_returns():
    feed = _flat(300.0)
    scene, frame = _capture(feed)
    points = reconstruct(frame, feed_mask=frame.valid).points_world
    geometric = classify_geometric(points, frame.valid, profile=PROFILE, wall_margin=25.0)

    assert geometric.sum() < frame.valid.sum()
    assert np.array_equal(geometric, geometric & classify_oracle(frame, scene))


def test_all_fitters_produce_a_renderable_mesh():
    feed = _flat(2000.0)
    scene, frame = _capture(feed)
    for method in ("mean_level", "plane_fit", "linear_interp", "rbf"):
        recon = reconstruct(frame, method=method, feed_mask=classify_oracle(frame, scene))
        vertices, faces = surface_mesh(recon.fit, profile=PROFILE, rings=16, sectors=24)
        assert np.isfinite(vertices).all()
        assert faces.max() < len(vertices)
