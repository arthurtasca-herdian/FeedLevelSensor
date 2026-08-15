import numpy as np
import pytest

from tof_sim.geometry import FlatFeed, SiloProfile
from tof_sim.metrics import sampled_fraction, volume_under
from tof_sim.reconstruct import MeanLevelFit

HOPPER = SiloProfile(outlet_radius=250.0, body_radius=1500.0, roof_radius=350.0,
                     z_outlet=0.0, z_hopper_top=1400.0, z_cylinder_top=4400.0, z_roof_top=5200.0)
CYLINDER = SiloProfile.cylindrical(radius=1500.0, height=5200.0)

# One level inside each of the three sections, plus both section joins and the very top.
LEVELS = [500.0, 1400.0, 2500.0, 4400.0, 4800.0, 5200.0]


def _level_fit(level: float) -> MeanLevelFit:
    return MeanLevelFit(np.array([[0.0, 0.0, level]], dtype=float))


def _disc(radius: float, n: int = 400) -> np.ndarray:
    # Deterministic spiral rather than random points, so the hull area is reproducible.
    k = np.arange(n, dtype=float)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n)])


@pytest.mark.parametrize("profile", [HOPPER, CYLINDER], ids=["hopper", "cylinder"])
@pytest.mark.parametrize("level", LEVELS)
def test_volume_under_a_flat_fit_matches_the_closed_form(profile, level):
    numeric = volume_under(_level_fit(level), profile, grid=1024)
    assert numeric == pytest.approx(profile.volume_below_level(level), rel=1e-3)


@pytest.mark.parametrize("level", LEVELS)
def test_volume_under_accepts_a_height_field_as_well_as_a_fit(level):
    # The extraction is only worth having if ground truth and reconstruction go through one path.
    feed = FlatFeed(level=level, radius=HOPPER.body_radius, clip_profile=HOPPER)
    assert volume_under(feed, HOPPER, grid=1024) == pytest.approx(
        volume_under(_level_fit(level), HOPPER, grid=1024), rel=1e-3)


def test_volume_under_saturates_at_capacity_for_an_overfull_surface():
    assert volume_under(_level_fit(99_000.0), HOPPER, grid=1024) == pytest.approx(
        HOPPER.capacity(), rel=1e-3)


def test_an_empty_silo_integrates_to_zero():
    assert volume_under(_level_fit(HOPPER.z_outlet), HOPPER, grid=512) == pytest.approx(0.0, abs=1.0)


def test_finer_grids_converge_on_the_closed_form():
    exact = HOPPER.volume_below_level(2500.0)
    errors = [abs(volume_under(_level_fit(2500.0), HOPPER, grid=g) - exact) for g in (128, 512, 1024)]
    assert errors[0] > errors[1] > errors[2]


def test_samples_spanning_the_silo_cover_nearly_all_of_it():
    # A hull over discrete samples is an inscribed polygon, so it approaches the disc from below
    # and only reaches it in the limit.
    fractions = [sampled_fraction(_disc(HOPPER.body_radius, n), HOPPER, grid=256)
                 for n in (100, 400, 1600)]
    assert fractions[0] > 0.85
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0, abs=0.02)


def test_a_central_patch_covers_the_area_it_spans():
    # The hull of a disc of radius r covers (r / body_radius)^2 of the cross-section.
    fraction = sampled_fraction(_disc(750.0), HOPPER, grid=256)
    assert fraction == pytest.approx(0.25, abs=0.02)


def test_coverage_grows_with_the_sampled_radius():
    fractions = [sampled_fraction(_disc(r), HOPPER, grid=256) for r in (300.0, 700.0, 1100.0, 1500.0)]
    assert fractions == sorted(fractions)
    assert fractions[0] < 0.1 < fractions[-1]


def test_coverage_of_a_square_grid_leaves_the_corners_out():
    # What an 8x8 zone grid actually does: a rectangle inscribed in a circular cross-section.
    axis = np.linspace(-1500.0 / np.sqrt(2.0), 1500.0 / np.sqrt(2.0), 8)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    # A square inscribed in a circle is 2r^2 against pi r^2.
    assert sampled_fraction(points, HOPPER, grid=256) == pytest.approx(2.0 / np.pi, abs=0.02)


def test_too_few_points_to_form_a_hull_report_no_coverage():
    assert sampled_fraction(np.zeros((2, 3)), HOPPER, grid=128) == 0.0
