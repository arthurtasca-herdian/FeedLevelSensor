from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tof_sim import config as config_module
from tof_sim import viz
from tof_sim.config import SimulationConfig
from tof_sim.camera import unproject
from tof_sim.geometry import Scene
from tof_sim.metrics import sampled_fraction, volume_under
from tof_sim.reading import frame_from_reading, load_reading
from tof_sim.reconstruct import FITTERS, check_is_contained_in_silo, reconstruct, surface_mesh


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place one measured capture in the silo and render it. No metrics are "
                    "produced: a real capture has no ground truth to score against.")
    parser.add_argument("--config", required=True, type=Path,
                        help="scene YAML describing the installation; its feed and metrics "
                             "sections are unused")
    parser.add_argument("--reading", required=True, type=Path,
                        help=".npz written by python-poc/export_reading.py")
    parser.add_argument("--out", required=True, type=Path,
                        help="directory to write the views into")
    parser.add_argument("--recon", choices=sorted(FITTERS),
                        help="surface fitter, overriding the config")
    parser.add_argument("--show", action="store_true",
                        help="open an interactive window instead of rendering headless")
    parser.add_argument("--spread", action="store_true",
                        help="also fit every other method and report the range of volumes")
    return parser.parse_args(argv)


def fitter_spread(feed_points, profile, grid: int) -> tuple:
    """Volume from every fitter over one set of points, plus the fitters that could not run.

    Which surface to draw through 64 samples is an assumption, not a measurement, so the range
    these disagree over is what the capture failed to pin down. Coverage cannot answer this: it
    reports sampled *area*, and two captures with the same coverage can disagree here by tenfold
    when one of them leaves the surface between its samples unconstrained.
    """
    volumes, failed = {}, []
    for name, fitter in sorted(FITTERS.items()):
        try:
            volumes[name] = volume_under(fitter(feed_points), profile, grid=grid)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Fitters have different minimum sample counts, so a sparse capture can rule some out.
            failed.append(f"{name} ({exc})")
    return volumes, failed


def _summary(volume, coverage, profile, heights, spread=None) -> list:
    capacity = profile.capacity()
    rows = [
        ("feed volume", f"{volume / 1e9:.3f} m3"),
        ("fill", f"{100 * volume / capacity:.2f} % of {capacity / 1e9:.3f} m3 capacity"),
        ("feed height", f"{heights.min():.0f} .. {heights.max():.0f} mm"
                        f"  (mean {heights.mean():.0f} mm)"),
        # A zone grid is rectangular and a silo is not, so the corners of the cross-section are
        # never sampled. Everything outside the hull is the fitter extrapolating.
        ("coverage", f"{100 * coverage:.1f} % of the cross-section sampled"
                     f"  ({100 * (1 - coverage):.1f} % extrapolated)"),
    ]
    if spread:
        volumes, failed = spread
        values = sorted(volumes.values())
        band = (values[-1] - values[0]) / np.mean(values) if values else float("nan")
        rows.append(("model spread", f"{values[0] / 1e9:.3f} .. {values[-1] / 1e9:.3f} m3"
                                     f"  ({100 * band:.1f} % across {len(values)} fitters)"))
        rows.append(("  by fitter", ", ".join(
            f"{name} {v / 1e9:.3f}" for name, v in sorted(volumes.items(), key=lambda kv: kv[1]))))
        if failed:
            rows.append(("  could not fit", "; ".join(failed)))
    return rows


def _report(cfg, reading, recon, profile, summary) -> str:
    rows = [
        ("reading", reading.describe()),
        ("captured", reading.meta.get("host_utc", "?")),
        ("installation", cfg.name),
        ("silo", f"{cfg.silo.type}, capacity {profile.capacity() / 1e9:.3f} m3"),
        ("placement", cfg.sensor.describe()),
        ("zones", f"{cfg.camera.width}x{cfg.camera.height}"
                  f"  ({int(recon.feed_mask.sum())} on feed of"
                  f" {int(np.isfinite(reading.distances_mm).sum())} returned)"),
        ("field of view", f"{cfg.camera.hfov_deg:.2f} x {cfg.camera.vfov_deg:.2f} deg"),
        ("method", f"{recon.method} / geometric / camera={cfg.reconstruct.camera_model}"),
        *summary,
    ]
    warnings = reading.meta.get("warnings")
    if warnings:
        rows.append(("frame warnings", warnings))
    if not reading.meta.get("integrity_ok", True):
        rows.append(("integrity", "FAILED in the capture log"))
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.rjust(width)} : {v}" for k, v in rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg: SimulationConfig = config_module.load(args.config)
    if args.recon:
        cfg.reconstruct.method = args.recon

    reading = load_reading(args.reading)
    intrinsics = cfg.reconstruct.intrinsics(cfg.camera)
    frame = frame_from_reading(reading, intrinsics, cfg.sensor.pose(),
                               min_range_mm=cfg.sensor.min_range_mm,
                               max_range_mm=cfg.sensor.max_range_mm)

    profile = cfg.build_profile()
    scene = Scene()
    for name, surface, role in profile.surfaces():
        scene.add(name, surface, role=role)

    preview = unproject(np.nan_to_num(frame.distance_mm), intrinsics, frame.pose)
    feed_mask = check_is_contained_in_silo(
        preview, frame.valid, profile=profile, wall_margin=cfg.reconstruct.wall_margin_mm,
    )
    recon = reconstruct(
        frame,
        method=cfg.reconstruct.method,
        intrinsics=intrinsics,
        feed_mask=feed_mask)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headless = not args.show

    volume = volume_under(recon.fit, profile, grid=cfg.metrics.grid)
    coverage = sampled_fraction(recon.feed_points, profile, grid=cfg.metrics.grid)
    spread = fitter_spread(recon.feed_points, profile, cfg.metrics.grid) if args.spread else None
    summary = _summary(volume, coverage, profile, recon.feed_points[:, 2], spread)
    overlay = "\n".join(f"{k}: {v}" for k, v in summary)

    vertices, faces = surface_mesh(recon.fit, profile=profile)
    written = {"distance_map": viz.distance_map_view(frame, out / "distance_map.png")}
    written |= {f"reading_{k}": v for k, v in viz.render(
        viz.reading_view(scene, frame, recon.points_world, (vertices, faces),
                         feed_mask=recon.feed_mask, headless=headless, overlay=overlay),
        out, "reading", headless, overlay=overlay).items()}

    print(_report(cfg, reading, recon, profile, summary))
    print("\n  wrote:")
    for name, path in written.items():
        print(f"    {name:16s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
