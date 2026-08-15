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

## Placing a real capture

A measured reading is exported from a capture log by `python-poc/export_reading.py` and read back
here. The handoff is a file, not an import: nothing in `tof_sim/` knows the log format or the
vendor driver.

```bash
# in python-poc/
uv run python export_reading.py field-tests/<log>.ndjson.gz --set 1 -o reading.npz

# in simulation/
uv run python scripts/run_reading.py --config configs/<installation>.yaml \
    --reading reading.npz --out out/field
```

It reports feed volume, fill, height and **coverage** — the share of the cross-section the zone grid
spanned. A zone grid is rectangular and a silo is not, so the corners are never sampled and the
fitter extrapolates there; coverage says how much of the volume was inferred rather than measured.
`metrics.grid` sets the integration resolution. The same figures are overlaid on the rendered views.

`--spread` adds **model spread**: the same points fitted by all four methods, reported as a range.
Which surface to draw through 64 samples is an assumption, not a measurement, so the range they
disagree over is what the capture failed to pin down. It answers a question coverage cannot — two
captures can span the same *area* and still differ tenfold in how well that area constrains the
surface. On the field captures, a full silo spreads 2 % across the four fitters and a near-empty
one 24 %, at 72 % and 71 % coverage respectively.

The sensor placement and silo dimensions come from the config and must be **measured**. None of the
captures in `python-poc/field-tests/` recorded a mount pose, so for those logs the placement is a
guess and the world coordinates are only as good as that guess.

## Configuration

Scene files are YAML; `configs/scene.template.yaml` documents every key and its default.
`configs/field_capture.template.yaml` is the same schema trimmed to what a real capture uses.
Lengths are millimetres, angles degrees.

`silo.type: profiled` is the usual body — hopper cone, cylindrical shell, roof cone, no lid.
`silo.type: cylinder` is a plain tube, used by the two baseline scenes whose answers are known
in closed form. See [ADR-002](../docs/adr/002-silo-body-model.md).

`sensor.placement: euler` gives mounting angles against a world-aligned reference — at
yaw = pitch = roll = 0 the camera axes coincide with the world axes, so the optical axis points
straight **up**. Pitch is the polar angle from world `+Z` and yaw the azimuth it leans towards, so
a roof-mounted sensor hangs at `pitch: 180` and an inclinometer reading `t` becomes `180 - t`.
`sensor.placement: look_at` aims it at `target_mm` instead. See
[ADR-003](../docs/adr/003-sensor-placement-parametrization.md) and
[ADR-004](../docs/adr/004-world-aligned-euler-reference.md).

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
    ├── reading.py      measured grids exported from a capture log
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
