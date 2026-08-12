"""Colour roles shared by every view.

Structural geometry (silo shell, floor, rays) is chrome and wears neutral ink, so hue is
reserved for things that carry meaning: a single-hue ramp for magnitude, a two-hue ramp with
a neutral midpoint for signed error, and fixed categorical slots for named surfaces.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

SERIES_TRUE = "#eb6834"
SERIES_RECON = "#2a78d6"

_SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b",
]

_DIVERGING_BLUE_RED = [
    "#0d366b", "#184f95", "#2a78d6", "#86b6ef",
    "#f0efec",
    "#f0a3a3", "#e34948", "#b02a2a", "#7a1414",
]

DISTANCE_CMAP = LinearSegmentedColormap.from_list("tof_distance", _SEQUENTIAL_BLUE)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list("tof_residual", _DIVERGING_BLUE_RED)

ROLE_STYLE = {
    "wall": {"color": INK_MUTED, "opacity": 0.12},
    "floor": {"color": INK_MUTED, "opacity": 0.30},
    "feed": {"color": SERIES_TRUE, "opacity": 1.0},
    "other": {"color": GRIDLINE, "opacity": 0.25},
}
