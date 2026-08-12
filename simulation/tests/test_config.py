import json
from pathlib import Path

import numpy as np
import pytest

from tof_sim import config as config_module
from tof_sim.camera import Intrinsics
from tof_sim.geometry import CallableHeightField, ConicalPileFeed, FlatFeed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["scene.template.yaml", "flat_half_full.yaml", "centre_cone_pile.yaml", "hopper_silo.yaml", "offset_tilted.yaml"])
def test_shipped_configs_load_and_build(name):
    cfg = config_module.load(CONFIGS / name)
    profile = cfg.build_profile()
    feed = cfg.build_feed(profile)
    scene = cfg.build_scene(feed, profile)
    frame = cfg.build_sensor().capture(scene)

    assert frame.shape == (cfg.camera.height, cfg.camera.width)
    assert frame.valid.any()


def test_feed_types_build_expected_surfaces():
    silo = config_module.SiloConfig(type="cylinder", radius_mm=1000.0).build()
    assert isinstance(config_module.FeedConfig(type="flat").build(silo), FlatFeed)
    assert isinstance(config_module.FeedConfig(type="conical_pile").build(silo), ConicalPileFeed)

    expression = config_module.FeedConfig(
        type="expression", expression="1000.0 + 100.0 * sin(x / 300.0)"
    ).build(silo)
    assert isinstance(expression, CallableHeightField)
    assert expression.height(np.array(0.0), np.array(0.0)) == pytest.approx(1000.0)
    assert np.isnan(expression.height(np.array(5000.0), np.array(0.0)))


def test_expression_feed_rejects_missing_expression():
    with pytest.raises(ValueError):
        config_module.FeedConfig(type="expression").build(config_module.SiloConfig().build())


def test_camera_model_selection():
    camera = config_module.CameraConfig(width=8, height=8, hfov_deg=50.0, vfov_deg=40.0)
    matched = config_module.ReconstructConfig(camera_model="matched").intrinsics(camera)
    vendor = config_module.ReconstructConfig(camera_model="vendor_zcorrection").intrinsics(camera)

    assert matched.hfov_deg == pytest.approx(50.0)
    assert vendor.fx == pytest.approx(Intrinsics.from_vendor_zcorrection(8, 8).fx)

    with pytest.raises(ValueError):
        config_module.ReconstructConfig(camera_model="nonsense").intrinsics(camera)


def test_cli_writes_metrics_and_views(tmp_path):
    from scripts.run_simulation import main

    out = tmp_path / "run"
    assert main([
        "--config", str(CONFIGS / "flat_half_full.yaml"), "--out", str(out), "--grid", "128",
    ]) == 0

    for name in ("metrics.json", "distance_map.png", "scene.png", "scene.html",
                 "residual.png", "residual.html"):
        assert (out / name).exists(), name

    metrics = json.loads((out / "metrics.json").read_text())
    assert abs(metrics["volume_error_frac"]) < 1e-3
    assert metrics["zone_count"] == 64
