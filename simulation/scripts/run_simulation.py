from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tof_sim import config as config_module
from tof_sim import viz
from tof_sim.metrics import evaluate
from tof_sim.reconstruct import FITTERS, classify_geometric, classify_oracle, reconstruct, surface_mesh


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recon", choices=sorted(FITTERS))
    parser.add_argument("--camera-model", choices=("matched", "vendor_zcorrection"))
    parser.add_argument("--classifier", choices=("geometric", "oracle"))
    parser.add_argument("--grid", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def _report(cfg, metrics, profile) -> str:
    capacity = profile.capacity()
    rows = [
        ("scene", cfg.name),
        ("silo", f"{cfg.silo.type}, capacity {capacity / 1e9:.3f} m3"),
        ("placement", cfg.sensor.describe()),
        ("zones", f"{cfg.camera.width}x{cfg.camera.height}"
                  f"  ({metrics.valid_zone_count}/{metrics.zone_count} valid,"
                  f" {metrics.feed_zone_count} on feed)"),
        ("field of view", f"{cfg.camera.hfov_deg:.2f} x {cfg.camera.vfov_deg:.2f} deg"),
        ("method", f"{cfg.reconstruct.method} / {cfg.reconstruct.classifier}"
                   f" / camera={cfg.reconstruct.camera_model}"),
        ("true volume", f"{metrics.volume_true_mm3 / 1e9:12.6f} m3"),
        ("reconstructed volume", f"{metrics.volume_recon_mm3 / 1e9:12.6f} m3"),
        ("volume error", f"{metrics.volume_error_mm3 / 1e9:12.6f} m3"
                         f"  ({100 * metrics.volume_error_frac:+.3f} %)"),
        ("fill true / recon", f"{100 * metrics.volume_true_mm3 / capacity:6.2f} %"
                              f" / {100 * metrics.volume_recon_mm3 / capacity:.2f} %"),
        ("equivalent level error", f"{metrics.level_error_mm:+9.2f} mm"),
        ("residual RMS", f"{metrics.residual_rms_mm:9.2f} mm"),
        ("residual bias", f"{metrics.residual_bias_mm:+9.2f} mm"),
        ("residual max abs", f"{metrics.residual_max_abs_mm:9.2f} mm"),
        ("residual p95", f"{metrics.residual_p95_mm:9.2f} mm"),
        ("sample offset RMS", f"{metrics.sample_offset_rms_mm:9.4f} mm"),
        ("sample offset max", f"{metrics.sample_offset_max_mm:9.4f} mm"),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.rjust(width)} : {v}" for k, v in rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = config_module.load(args.config)
    if args.recon:
        cfg.reconstruct.method = args.recon
    if args.camera_model:
        cfg.reconstruct.camera_model = args.camera_model
    if args.classifier:
        cfg.reconstruct.classifier = args.classifier
    if args.grid:
        cfg.metrics.grid = args.grid

    profile = cfg.build_profile()
    feed = cfg.build_feed(profile)
    scene = cfg.build_scene(feed, profile)
    frame = cfg.build_sensor().capture(scene)

    if cfg.reconstruct.classifier == "oracle":
        feed_mask = classify_oracle(frame, scene)
    else:
        from tof_sim.camera import unproject

        preview = unproject(np.nan_to_num(frame.distance_mm), cfg.reconstruct.intrinsics(cfg.camera),
                            frame.pose)
        feed_mask = classify_geometric(
            preview, frame.valid, profile=profile, wall_margin=cfg.reconstruct.wall_margin_mm,
        )

    recon = reconstruct(frame, method=cfg.reconstruct.method,
                        intrinsics=cfg.reconstruct.intrinsics(cfg.camera), feed_mask=feed_mask)
    metrics = evaluate(frame, recon, feed, profile=profile, grid=cfg.metrics.grid)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2) + "\n")

    headless = not args.show
    written = {"metrics": out / "metrics.json"}
    written["distance_map"] = viz.distance_map_view(frame, out / "distance_map.png")
    written |= {f"scene_{k}": v for k, v in
                viz.render(viz.scene_view(scene, frame, headless=headless), out, "scene", headless).items()}

    vertices, faces = surface_mesh(recon.fit, profile=profile)
    residual = vertices[:, 2] - feed.height(vertices[:, 0], vertices[:, 1])
    written |= {f"residual_{k}": v for k, v in
                viz.render(viz.residual_view(scene, (vertices, faces), residual, frame, headless=headless),
                           out, "residual", headless).items()}

    print(_report(cfg, metrics, profile))
    print("\n  wrote:")
    for name, path in written.items():
        print(f"    {name:16s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
