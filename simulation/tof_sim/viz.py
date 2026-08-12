"""PyVista views of the simulated scene and of reconstruction error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from .camera import Intrinsics
from .geometry import Scene
from .palette import (
    DISTANCE_CMAP,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    RESIDUAL_CMAP,
    ROLE_STYLE,
    SERIES_TRUE,
    SURFACE,
)
from .sensor import Frame

_SCALAR_BAR = {
    "color": INK_SECONDARY,
    "title_font_size": 13,
    "label_font_size": 11,
    "width": 0.30,
    "height": 0.05,
    "position_x": 0.35,
    "position_y": 0.03,
    "n_labels": 5,
}


def to_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    faces = np.asarray(faces)
    padded = np.hstack([np.full((len(faces), 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=float), padded)


def _new_plotter(headless: bool, title: str) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=headless, window_size=(1280, 900), title=title)
    plotter.set_background(SURFACE)
    return plotter


def _add_scene_geometry(plotter: pv.Plotter, scene: Scene, feed_opacity: float | None = None) -> None:
    for element in scene.elements:
        style = dict(ROLE_STYLE.get(element.role, ROLE_STYLE["other"]))
        if element.role == "feed" and feed_opacity is not None:
            style["opacity"] = feed_opacity
        vertices, faces = element.surface.triangulate()
        plotter.add_mesh(to_polydata(vertices, faces), label=element.name, **style)


def _frustum_edges(intrinsics: Intrinsics, pose: np.ndarray, length: float) -> pv.PolyData:
    corners_uv = [(0, 0), (intrinsics.width, 0), (intrinsics.width, intrinsics.height), (0, intrinsics.height)]
    dirs = []
    for u, v in corners_uv:
        d = np.array([(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0])
        dirs.append(d / np.linalg.norm(d))
    dirs = np.asarray(dirs) @ pose[:3, :3].T

    origin = pose[:3, 3]
    far = origin + length * dirs
    points = np.vstack([origin[None, :], far])
    lines = [[2, 0, i + 1] for i in range(4)]
    lines += [[2, i + 1, (i + 1) % 4 + 1] for i in range(4)]
    return pv.PolyData(points, lines=np.hstack(lines))


def _ray_segments(frame: Frame) -> pv.PolyData:
    hits = frame.hit_points[frame.valid]
    origins = np.broadcast_to(frame.origin, hits.shape)
    points = np.empty((2 * len(hits), 3))
    points[0::2] = origins
    points[1::2] = hits
    lines = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(hits))])
    return pv.PolyData(points, lines=lines)


def scene_view(scene: Scene, frame: Frame, headless: bool = True, show_rays: bool = True,
               show_frustum: bool = True) -> pv.Plotter:
    """Ground-truth geometry with the sensor's rays and the points they land on."""
    plotter = _new_plotter(headless, "ToF scene")
    _add_scene_geometry(plotter, scene)

    hits = frame.hit_points[frame.valid]
    distances = frame.distance_mm[frame.valid]

    if show_rays and len(hits):
        plotter.add_mesh(_ray_segments(frame), color=INK_MUTED, line_width=1, opacity=0.5)

    if len(hits):
        cloud = pv.PolyData(hits)
        cloud["distance (mm)"] = distances
        plotter.add_mesh(
            cloud, scalars="distance (mm)", cmap=DISTANCE_CMAP, render_points_as_spheres=True,
            point_size=14, scalar_bar_args={"title": "range (mm)", **_SCALAR_BAR},
        )

    reach = float(np.nanmax(frame.distance_mm)) if frame.valid.any() else 1000.0
    if show_frustum:
        plotter.add_mesh(_frustum_edges(frame.intrinsics, frame.pose, reach),
                         color=INK_PRIMARY, line_width=2, opacity=0.55)

    plotter.add_mesh(pv.Sphere(radius=max(reach * 0.012, 1.0), center=frame.origin),
                     color=INK_PRIMARY)
    plotter.add_axes(color=INK_SECONDARY)
    plotter.camera_position = "xz"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 18
    plotter.reset_camera()
    return plotter


def residual_view(scene: Scene, recon_surface, residual_mm: np.ndarray, frame: Frame,
                  headless: bool = True, symmetric_limit: float | None = None) -> pv.Plotter:
    """Reconstructed surface coloured by signed error against the true feed surface."""
    plotter = _new_plotter(headless, "Reconstruction residual")
    _add_scene_geometry(plotter, scene, feed_opacity=0.25)

    vertices, faces = recon_surface
    mesh = to_polydata(vertices, faces)
    mesh["residual (mm)"] = np.asarray(residual_mm, dtype=float)

    limit = symmetric_limit or float(np.nanmax(np.abs(residual_mm)))
    limit = max(limit, 1e-6)
    plotter.add_mesh(
        mesh, scalars="residual (mm)", cmap=RESIDUAL_CMAP, clim=(-limit, limit),
        scalar_bar_args={"title": "reconstructed - true (mm)", **_SCALAR_BAR},
        label="reconstructed surface",
    )

    hits = frame.hit_points[frame.valid]
    if len(hits):
        plotter.add_mesh(pv.PolyData(hits), color=INK_PRIMARY, render_points_as_spheres=True,
                         point_size=10, label="sensor samples")

    # The reconstructed surface is named by the scalar bar, so a legend swatch for it would
    # imply a flat colour it never has.
    plotter.add_legend(
        [["true feed surface", SERIES_TRUE], ["sensor samples", INK_PRIMARY]],
        bcolor=SURFACE, border=True, face="rectangle", size=(0.24, 0.08), loc="upper left",
    )
    plotter.add_axes(color=INK_SECONDARY)
    plotter.camera_position = "xz"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 18
    plotter.reset_camera()
    return plotter


def distance_map_view(frame: Frame, out_path: Path | str, max_labelled: int = 16) -> Path:
    """Heatmap of the zone grid, mirroring how the driver presents a live pixel map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = frame.shape
    fig, ax = plt.subplots(figsize=(1.0 + width * 0.55, 1.4 + height * 0.55))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    data = np.ma.masked_invalid(frame.distance_mm)
    cmap = DISTANCE_CMAP.with_extremes(bad=GRIDLINE)
    image = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest")

    if max(height, width) <= max_labelled:
        mid = 0.5 * (np.nanmin(frame.distance_mm) + np.nanmax(frame.distance_mm))
        for row in range(height):
            for col in range(width):
                value = frame.distance_mm[row, col]
                if not np.isfinite(value):
                    ax.text(col, row, "-", ha="center", va="center", fontsize=7, color=INK_MUTED)
                    continue
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if value > mid else INK_PRIMARY)

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("zone column", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("zone row", color=INK_SECONDARY, fontsize=9)
    ax.set_title(f"reported range per zone (mm) - {width}x{height}", color=INK_PRIMARY,
                 fontsize=11, pad=10)

    bar = fig.colorbar(image, ax=ax, shrink=0.82)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def render(plotter: pv.Plotter, out_dir: Path | str, stem: str, headless: bool = True) -> dict[str, Path]:
    """Write a PNG and a standalone interactive HTML, then show the window when not headless."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    html_path = out_dir / f"{stem}.html"
    plotter.export_html(html_path)
    written["html"] = html_path

    png_path = out_dir / f"{stem}.png"
    if headless:
        plotter.screenshot(png_path)
        written["png"] = png_path
        plotter.close()
    else:
        plotter.show(screenshot=png_path)
        written["png"] = png_path
    return written
