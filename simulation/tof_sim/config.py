"""YAML scene definitions and the objects they build.

A config fully determines a run: intrinsics, placement, silo, feed and the reconstruction
under test. ``configs/scene.template.yaml`` documents every key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .camera import Intrinsics, from_euler, look_at
from .geometry import CallableHeightField, ConicalPileFeed, FlatFeed, HeightField, Scene, SiloProfile
from .sensor import DEVICE_LSB_MM, IdealTofSensor

# Expressions in a config are evaluated with numpy available and builtins withheld.
_EXPRESSION_NAMESPACE = {
    "np": np, "pi": np.pi, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
    "exp": np.exp, "hypot": np.hypot, "abs": np.abs, "minimum": np.minimum,
    "maximum": np.maximum, "where": np.where,
}


@dataclass
class CameraConfig:
    width: int = 8
    height: int = 8
    hfov_deg: float = 67.38
    vfov_deg: float = 53.13

    def build(self) -> Intrinsics:
        return Intrinsics.from_fov(self.width, self.height, self.hfov_deg, self.vfov_deg)


@dataclass
class SensorConfig:
    """Sensor placement. ``euler`` gives mounting angles, ``look_at`` gives a point to aim at."""

    placement: str = "euler"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 4000.0)
    yaw_deg: float = 0.0
    # Zero angles leave the camera world-aligned, which points the optical axis at the roof.
    # A roof-mounted sensor hangs at pitch 180, so that is the default rather than 0.
    pitch_deg: float = 180.0
    roll_deg: float = 0.0
    target_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_range_mm: float = 0.0
    max_range_mm: float = 10_000.0
    quantization_mm: float = DEVICE_LSB_MM

    def pose(self) -> np.ndarray:
        if self.placement == "euler":
            return from_euler(self.position_mm, yaw_deg=self.yaw_deg,
                              pitch_deg=self.pitch_deg, roll_deg=self.roll_deg)
        if self.placement == "look_at":
            return look_at(self.position_mm, self.target_mm, roll_deg=self.roll_deg)
        raise ValueError(f"unknown sensor.placement {self.placement!r}")

    def describe(self) -> str:
        position = ", ".join(f"{v:.0f}" for v in self.position_mm)
        if self.placement == "euler":
            return (f"euler at ({position}) yaw {self.yaw_deg:g} pitch {self.pitch_deg:g}"
                    f" roll {self.roll_deg:g} deg")
        return f"look_at from ({position}) to ({', '.join(f'{v:.0f}' for v in self.target_mm)})"

    def build(self, intrinsics: Intrinsics) -> IdealTofSensor:
        return IdealTofSensor(
            intrinsics, self.pose(), min_range_mm=self.min_range_mm,
            max_range_mm=self.max_range_mm, quantization_mm=self.quantization_mm,
        )


@dataclass
class SiloConfig:
    type: str = "profiled"
    outlet_radius_mm: float = 250.0
    body_radius_mm: float = 1500.0
    roof_radius_mm: float = 350.0
    z_outlet_mm: float = 0.0
    z_hopper_top_mm: float = 1400.0
    z_cylinder_top_mm: float = 4400.0
    z_roof_top_mm: float = 5200.0
    radius_mm: float = 1500.0
    height_mm: float = 4000.0
    floor_z_mm: float = 0.0

    def build(self) -> SiloProfile:
        if self.type == "cylinder":
            return SiloProfile.cylindrical(
                radius=self.radius_mm, height=self.height_mm, floor_z=self.floor_z_mm
            )
        if self.type == "profiled":
            return SiloProfile(
                outlet_radius=self.outlet_radius_mm,
                body_radius=self.body_radius_mm,
                roof_radius=self.roof_radius_mm,
                z_outlet=self.z_outlet_mm,
                z_hopper_top=self.z_hopper_top_mm,
                z_cylinder_top=self.z_cylinder_top_mm,
                z_roof_top=self.z_roof_top_mm,
            )
        raise ValueError(f"unknown silo.type {self.type!r}")


@dataclass
class FeedConfig:
    type: str = "flat"
    level_mm: float = 2000.0
    base_level_mm: float = 1200.0
    peak_height_mm: float = 800.0
    pile_radius_mm: float = 1100.0
    pile_centre_mm: tuple[float, float] | None = None
    expression: str | None = None

    def build(self, profile: SiloProfile) -> HeightField:
        shared = {"radius": profile.body_radius, "clip_profile": profile}
        if self.type == "flat":
            return FlatFeed(level=self.level_mm, **shared)
        if self.type == "conical_pile":
            return ConicalPileFeed(
                base_level=self.base_level_mm, peak_height=self.peak_height_mm,
                pile_radius=self.pile_radius_mm, pile_centre=self.pile_centre_mm, **shared,
            )
        if self.type == "expression":
            if not self.expression:
                raise ValueError("feed.type 'expression' requires feed.expression")
            code = compile(self.expression, "<feed.expression>", "eval")

            def height(x, y, _code=code):
                return np.broadcast_to(
                    eval(_code, {"__builtins__": {}}, {**_EXPRESSION_NAMESPACE, "x": x, "y": y}),
                    np.broadcast(x, y).shape,
                ).astype(float)

            return CallableHeightField(height, **shared)
        raise ValueError(f"unknown feed.type {self.type!r}")


@dataclass
class ReconstructConfig:
    method: str = "linear_interp"
    classifier: str = "geometric"
    wall_margin_mm: float = 25.0
    camera_model: str = "matched"

    def intrinsics(self, camera: CameraConfig) -> Intrinsics:
        if self.camera_model == "matched":
            return camera.build()
        if self.camera_model == "vendor_zcorrection":
            return Intrinsics.from_vendor_zcorrection(camera.width, camera.height)
        raise ValueError(f"unknown reconstruct.camera_model {self.camera_model!r}")


@dataclass
class MetricsConfig:
    grid: int = 512


@dataclass
class SimulationConfig:
    name: str = "scene"
    camera: CameraConfig = field(default_factory=CameraConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    silo: SiloConfig = field(default_factory=SiloConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def build_profile(self) -> SiloProfile:
        return self.silo.build()

    def build_feed(self, profile: SiloProfile | None = None) -> HeightField:
        return self.feed.build(profile or self.build_profile())

    def build_scene(self, feed: HeightField, profile: SiloProfile | None = None) -> Scene:
        scene = Scene()
        for name, surface, role in (profile or self.build_profile()).surfaces():
            scene.add(name, surface, role=role)
        return scene.add("feed", feed, role="feed")

    def build_sensor(self) -> IdealTofSensor:
        return self.sensor.build(self.camera.build())


def _section(data: dict, key: str, cls):
    return cls(**(data.get(key) or {}))


def load(path: Path | str) -> SimulationConfig:
    """Read a scene YAML file into a :class:`SimulationConfig`."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    return SimulationConfig(
        name=data.get("name", path.stem),
        camera=_section(data, "camera", CameraConfig),
        sensor=_section(data, "sensor", SensorConfig),
        silo=_section(data, "silo", SiloConfig),
        feed=_section(data, "feed", FeedConfig),
        reconstruct=_section(data, "reconstruct", ReconstructConfig),
        metrics=_section(data, "metrics", MetricsConfig),
    )
