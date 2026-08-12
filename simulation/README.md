# simulation

Noise-free simulation of the TMF8829 observing a silo feed surface: it renders a synthetic
distance grid from analytic geometry, runs the reconstruction on it, and scores the result
against ground truth, so reconstruction algorithms and sensor placements can be compared
without hardware.

## Setup

```bash
cd simulation
uv sync
uv run pytest
```

## Running

```bash
uv run python scripts/run_simulation.py --config configs/hopper_silo.yaml --out out/hopper
uv run python scripts/run_simulation.py --config configs/offset_tilted.yaml --out out/tilted
uv run python scripts/run_simulation.py --config configs/flat_half_full.yaml --out out/flat
uv run python scripts/run_simulation.py --config configs/centre_cone_pile.yaml --out out/cone
```

`--show` opens an interactive window instead of writing files only. `--recon`,
`--camera-model`, `--classifier` and `--grid` override the config for one run.

Each run writes:

```
out/<name>/
├── metrics.json
├── distance_map.png
├── scene.png
├── scene.html
├── residual.png
└── residual.html
```

The `.html` files are standalone interactive 3D views.

## Configuration

Scene files are YAML; `configs/scene.template.yaml` documents every key and its default.
Lengths are millimetres, angles degrees.

`silo.type: profiled` is the usual body — hopper cone, cylindrical shell, roof cone, no lid.
`silo.type: cylinder` is a plain tube, used by the two baseline scenes whose answers are known
in closed form. See [ADR-002](../docs/adr/002-silo-body-model.md).

`sensor.placement: euler` gives mounting angles against a nadir-pointing reference — at
yaw = pitch = roll = 0 the optical axis points straight down, pitch tilts it away from vertical
and yaw is the direction of lean. `sensor.placement: look_at` aims it at `target_mm` instead.
See [ADR-003](../docs/adr/003-sensor-placement-parametrization.md).

The `camera.hfov_deg` / `camera.vfov_deg` defaults are **placeholders** — they are the field
of view implied by the ams driver's `zCorrection`, not a datasheet figure. Replace them with
the measured optics before trusting any absolute error number.

## Layout

```
simulation/
├── configs/            scene definitions; scene.template.yaml documents the schema
├── scripts/            callable entry points
├── tests/              analytic checks, no hardware required
└── tof_sim/
    ├── camera.py       frame conventions, intrinsics, pose, unprojection
    ├── geometry.py     analytic primitives and ray intersection
    ├── sensor.py       distance grid capture
    ├── reconstruct.py  point cloud and surface fitting
    ├── metrics.py      volume, level and residual error
    ├── config.py       YAML to objects
    ├── palette.py      colour roles shared by the views
    └── viz.py          PyVista and matplotlib views
```

## Frame conventions

World: `+Z` up, origin at the centre of the silo floor, millimetres. Camera: OpenCV — `+Z`
along the optical axis, `+X` right, `+Y` down. Poses are 4x4 `world_T_cam`. See
[ADR-001](../docs/adr/001-simulation-stack-and-frame-conventions.md).
