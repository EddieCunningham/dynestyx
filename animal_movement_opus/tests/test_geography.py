"""Tests for the synthetic geography and its grid operations."""

import jax.numpy as jnp
import numpy as np

from champlain_dsx import default_receivers, make_synthetic_lake


def test_lake_has_water_and_land():
    geo = make_synthetic_lake(nx=50, ny=50, cell=300.0)
    frac = float(geo.water.mean())
    assert 0.2 < frac < 0.8, "lake should be a mix of water and land"


def test_border_is_land():
    geo = make_synthetic_lake(nx=50, ny=50, cell=300.0)
    w = np.asarray(geo.water)
    assert not w[0, :].any()
    assert not w[-1, :].any()
    assert not w[:, 0].any()
    assert not w[:, -1].any()


def test_is_water_off_grid_is_false():
    geo = make_synthetic_lake(nx=40, ny=40, cell=300.0)
    assert not bool(geo.is_water(jnp.array(-100.0), jnp.array(50.0)))
    assert not bool(geo.is_water(jnp.array(1e9), jnp.array(1e9)))


def test_water_cell_centres_are_on_water():
    geo = make_synthetic_lake(nx=40, ny=40, cell=300.0)
    centres = geo.water_cell_centres()
    assert centres.shape[0] == int(geo.water.sum())
    on_water = geo.is_water(centres[:, 0], centres[:, 1])
    assert bool(on_water.all())


def test_region_labels_valid_on_water_only():
    geo = make_synthetic_lake(nx=40, ny=40, cell=300.0, n_regions=3)
    region = np.asarray(geo.region)
    water = np.asarray(geo.water)
    assert set(np.unique(region[water]).tolist()).issubset({0, 1, 2})
    assert (region[~water] == -1).all()


def test_line_of_sight_blocked_by_land():
    # Two water points on opposite sides of the western peninsula must fail the
    # midpoint line-of-sight test, while a short hop within open water passes.
    geo = make_synthetic_lake(nx=100, ny=100, cell=200.0)
    cx, cy = geo.width / 2.0, geo.height / 2.0
    # Peninsula sits at y = cy - 0.06*h and spans x in (cx-0.34w, cx+0.10w).
    y_pen = cy - 0.06 * geo.height
    north = (jnp.array(cx - 0.2 * geo.width), jnp.array(y_pen + 0.12 * geo.height))
    south = (jnp.array(cx - 0.2 * geo.width), jnp.array(y_pen - 0.12 * geo.height))
    assert bool(geo.is_water(*north))
    assert bool(geo.is_water(*south))
    blocked = geo.line_of_sight(north[0], north[1], south[0], south[1])
    assert not bool(blocked)
    open_water = geo.line_of_sight(north[0], north[1], north[0], north[1] + geo.cell)
    assert bool(open_water)


def test_receivers_on_water():
    geo = make_synthetic_lake(nx=80, ny=80, cell=250.0)
    recv = default_receivers(geo)
    assert recv.shape[0] >= 5
    assert bool(geo.is_water(recv[:, 0], recv[:, 1]).all())
