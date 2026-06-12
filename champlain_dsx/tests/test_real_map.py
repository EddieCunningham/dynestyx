"""Tests that exercise the acquired real Lake Champlain raster.

These are skipped when the raster has not been downloaded. Acquire it with

    uv run python champlain_dsx/scripts/acquire_champlain_map.py --res 200
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from champlain_dsx.environment import load_champlain

MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "champlain_map.npz"
pytestmark = pytest.mark.skipif(
    not MAP_PATH.exists(), reason="real Champlain raster not downloaded"
)


def test_real_map_geometry_is_plausible():
    env = load_champlain(str(MAP_PATH), coarsen=3)
    water_frac = float(jnp.mean(env.water_mask))
    # Lake Champlain is a long narrow lake; water fills a modest fraction of its
    # bounding box.
    assert 0.1 < water_frac < 0.4
    # The lake is far longer north to south than it is wide.
    assert (env.y_max - env.y_min) > 3.0 * (env.x_max - env.x_min)
    depth = np.asarray(env.depth)
    assert np.nanmax(depth) > 80.0  # the lake exceeds 100 m at its deepest


def test_summer_mask_is_a_strict_subset_of_water():
    annual = load_champlain(str(MAP_PATH), coarsen=3, summer=False)
    summer = load_champlain(str(MAP_PATH), coarsen=3, summer=True)
    assert int(summer.nav_mask.sum()) < int(annual.nav_mask.sum())
    # Summer navigability never adds cells that are not open water.
    assert bool(np.all(np.asarray(summer.nav_mask) <= np.asarray(annual.water_mask)))
