"""Tests for the rasterised lake environment."""

import jax
import jax.numpy as jnp
import numpy as np

from champlain_dsx.environment import environment_from_arrays, synthetic_champlain


def _toy_env():
    # 3x4 grid, 10 m cells, origin at x_min=0, y_max=30.
    # Water in the central column band; land elsewhere.
    nav = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    depth = np.where(nav > 0, 25.0, np.nan).astype(np.float32)
    return environment_from_arrays(nav, nav, depth, x_min=0.0, y_max=30.0, res=10.0)


def test_cell_indexing_round_trips_to_known_cells():
    env = _toy_env()
    # Centre of cell (row=0, col=1) is x=15, y=25.
    assert float(env.is_navigable(jnp.float32(15.0), jnp.float32(25.0))) == 1.0
    # Centre of cell (row=0, col=0) is x=5, y=25 -> land.
    assert float(env.is_navigable(jnp.float32(5.0), jnp.float32(25.0))) == 0.0


def test_out_of_bounds_is_land():
    env = _toy_env()
    assert float(env.is_navigable(jnp.float32(-100.0), jnp.float32(5.0))) == 0.0
    assert float(env.is_navigable(jnp.float32(15.0), jnp.float32(9999.0))) == 0.0


def test_line_of_sight_uses_midpoint():
    env = _toy_env()
    # Both endpoints in the water band, midpoint also water.
    los = env.line_of_sight(
        jnp.float32(15.0), jnp.float32(5.0), jnp.float32(25.0), jnp.float32(25.0)
    )
    assert float(los) == 1.0
    # Endpoints straddle the water band but the midpoint lands on the left land
    # column (x=5).
    los2 = env.line_of_sight(
        jnp.float32(-5.0), jnp.float32(15.0), jnp.float32(15.0), jnp.float32(15.0)
    )
    assert float(los2) == 0.0


def test_lookups_are_jittable_and_vmappable():
    env = _toy_env()
    xs = jnp.array([15.0, 25.0, 5.0, -1.0])
    ys = jnp.array([25.0, 15.0, 5.0, 5.0])
    out = jax.jit(jax.vmap(env.is_navigable))(xs, ys)
    np.testing.assert_array_equal(np.asarray(out), np.array([1.0, 1.0, 0.0, 0.0]))


def test_synthetic_champlain_is_mostly_connected_water():
    env = synthetic_champlain(res=400.0)
    frac_water = float(jnp.mean(env.water_mask))
    assert 0.05 < frac_water < 0.6
    centres = env.water_cell_centres()
    assert centres.shape[0] > 100
    # Every navigable centre must read back as navigable.
    nav = jax.vmap(env.is_navigable)(centres[:, 0], centres[:, 1])
    assert float(jnp.mean(nav)) > 0.99


def test_summer_mask_removes_shallow_water():
    annual = synthetic_champlain(res=400.0, summer=False)
    summer = synthetic_champlain(res=400.0, summer=True)
    assert float(jnp.sum(summer.nav_mask)) < float(jnp.sum(annual.nav_mask))
