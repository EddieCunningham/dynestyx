"""Raster environment for the Lake Champlain geolocation model.

The environment holds two gridded masks and a depth field over a projected
(UTM-like) coordinate system. The navigability mask marks cells an animal may
occupy. The water mask marks open water and drives the acoustic line-of-sight
test. Both masks share one affine transform.

The grid follows the GeoTIFF convention used by the reference project. The
origin sits at the top-left corner, columns increase eastward, and rows
increase southward. A coordinate (x, y) maps to

    col = floor((x - x_min) / res)
    row = floor((y_max - y) / res)

Lookups outside the grid return land (zero), so an animal can never leave the
modelled domain.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class LakeEnvironment(eqx.Module):
    """A rasterised lake with navigability, open-water, and depth grids.

    Attributes:
        nav_mask: Float grid, shape (H, W), 1.0 where an animal may move.
        water_mask: Float grid, shape (H, W), 1.0 over open water. Used for the
            acoustic line-of-sight test.
        depth: Float grid, shape (H, W), bottom depth in metres. NaN over land.
        x_min: Western edge of the grid in projected metres.
        y_max: Northern edge of the grid in projected metres.
        res: Cell size in metres.
    """

    nav_mask: Float[Array, "H W"]
    water_mask: Float[Array, "H W"]
    depth: Float[Array, "H W"]
    x_min: float = eqx.field(static=True)
    y_max: float = eqx.field(static=True)
    res: float = eqx.field(static=True)

    @property
    def height(self) -> int:
        return self.nav_mask.shape[0]

    @property
    def width(self) -> int:
        return self.nav_mask.shape[1]

    @property
    def x_max(self) -> float:
        return self.x_min + self.width * self.res

    @property
    def y_min(self) -> float:
        return self.y_max - self.height * self.res

    def _cell(self, x: Float[Array, ""], y: Float[Array, ""]):
        col = jnp.floor((x - self.x_min) / self.res).astype(jnp.int32)
        row = jnp.floor((self.y_max - y) / self.res).astype(jnp.int32)
        return row, col

    def _lookup(self, grid, x, y, fill):
        row, col = self._cell(x, y)
        in_bounds = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        row_c = jnp.clip(row, 0, self.height - 1)
        col_c = jnp.clip(col, 0, self.width - 1)
        value = grid[row_c, col_c]
        return jnp.where(in_bounds, value, fill)

    def is_navigable(self, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        """Return 1.0 where (x, y) is a navigable cell, else 0.0."""
        return self._lookup(self.nav_mask, x, y, 0.0)

    def is_water(self, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        return self._lookup(self.water_mask, x, y, 0.0)

    def depth_at(self, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        return self._lookup(self.depth, x, y, jnp.nan)

    def line_of_sight(self, x0, y0, x1, y1) -> Float[Array, ""]:
        """Return 1.0 when the midpoint between two points is open water.

        This mirrors the reference implementation, which approximates an
        unobstructed acoustic path by testing whether the straight-line
        midpoint falls on water.
        """
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        return self.is_water(xm, ym)

    def water_cell_centres(self) -> Float[Array, "N 2"]:
        """Return the (x, y) centres of every navigable cell."""
        rows, cols = np.nonzero(np.asarray(self.nav_mask) > 0.0)
        xs = self.x_min + (cols + 0.5) * self.res
        ys = self.y_max - (rows + 0.5) * self.res
        return jnp.stack([jnp.asarray(xs), jnp.asarray(ys)], axis=-1)


def environment_from_arrays(
    nav_mask: np.ndarray,
    water_mask: np.ndarray,
    depth: np.ndarray,
    x_min: float,
    y_max: float,
    res: float,
) -> LakeEnvironment:
    """Build a LakeEnvironment from numpy grids, coercing dtypes."""
    return LakeEnvironment(
        nav_mask=jnp.asarray(nav_mask, dtype=jnp.float32),
        water_mask=jnp.asarray(water_mask, dtype=jnp.float32),
        depth=jnp.asarray(depth, dtype=jnp.float32),
        x_min=float(x_min),
        y_max=float(y_max),
        res=float(res),
    )


def _coarsen_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a mask by an integer factor, marking a block water if any cell is."""
    h, w = mask.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    block = mask[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    return block.max(axis=(1, 3))


def _coarsen_depth(depth: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a depth field by an integer factor, averaging over wet cells."""
    h, w = depth.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    block = depth[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    with warnings.catch_warnings():
        # Blocks that are entirely land average over no wet cells, which is fine.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(block, axis=(1, 3))


def load_champlain(
    path: str,
    summer: bool = False,
    coarsen: int = 1,
    depth_threshold: float = 20.0,
) -> LakeEnvironment:
    """Load a Lake Champlain environment from an acquired ``.npz`` raster.

    Args:
        path: Path to the ``.npz`` produced by ``acquire_champlain_map.py``.
        summer: When true the navigability mask drops cells shallower than
            ``depth_threshold``, the thermal habitat constraint.
        coarsen: Integer downsampling factor for a coarser, faster grid.
        depth_threshold: Summer depth threshold in metres.
    """
    data = np.load(path)
    water = np.asarray(data["water_mask"], dtype=np.float32)
    depth = np.asarray(data["depth"], dtype=np.float32)
    res = float(data["res"]) * coarsen
    x_min = float(data["x_min"])
    y_max = float(data["y_max"])

    if coarsen > 1:
        water = _coarsen_mask(water, coarsen).astype(np.float32)
        depth = _coarsen_depth(depth, coarsen).astype(np.float32)

    if summer:
        nav = (water > 0) & (np.nan_to_num(depth, nan=0.0) >= depth_threshold)
        nav_mask = nav.astype(np.float32)
    else:
        nav_mask = water.copy()

    return environment_from_arrays(nav_mask, water, depth, x_min, y_max, res)


def synthetic_champlain(
    res: float = 400.0,
    summer: bool = False,
    seed: int = 0,
) -> LakeEnvironment:
    """Generate a Lake-Champlain-like narrow lake on a projected grid.

    The lake runs north to south with a width on the order of a few kilometres
    and a length on the order of one hundred kilometres. A meandering
    centreline plus a smoothly varying half-width carve open water out of a
    land background. A central deep trough plus shallow margins produce a depth
    field. When ``summer`` is true the navigability mask drops cells shallower
    than the 20 m thermal threshold used in the paper.

    The geometry is stylised. It exists so the model, simulator, and particle
    algorithms can run end to end without the proprietary survey rasters.
    """
    rng = np.random.default_rng(seed)

    length_m = 96_000.0
    width_m = 16_000.0
    height = int(round(length_m / res))
    width = int(round(width_m / res))

    x_min, y_max = 0.0, length_m

    xs = x_min + (np.arange(width) + 0.5) * res
    ys = y_max - (np.arange(height) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)

    # Meandering centreline and slowly varying half-width along the lake.
    t = (y_max - gy) / length_m  # 0 at north, 1 at south
    centre = 0.5 * width_m + 2200.0 * np.sin(2.0 * np.pi * 1.5 * t) \
        + 1400.0 * np.sin(2.0 * np.pi * 3.1 * t + 0.7)
    half_width = 3400.0 + 1700.0 * np.sin(2.0 * np.pi * 2.3 * t + 1.1) \
        + 900.0 * np.cos(2.0 * np.pi * 4.7 * t)
    half_width = np.clip(half_width, 1200.0, None)

    offset = np.abs(gx - centre)
    water = offset < half_width

    # A couple of bays hanging off the main channel add geometric complexity.
    def bay(cx, cy, r):
        return ((gx - cx) ** 2 + (gy - cy) ** 2) < r ** 2

    water |= bay(0.5 * width_m + 3200.0, 0.78 * length_m, 3200.0)
    water |= bay(0.5 * width_m - 3600.0, 0.32 * length_m, 2600.0)

    water_mask = water.astype(np.float32)

    # Depth: deep along the channel centre, shallow near the shore, plus noise.
    rel = np.clip(offset / np.maximum(half_width, 1.0), 0.0, 1.0)
    depth = 55.0 * (1.0 - rel ** 2)
    depth += 6.0 * np.sin(2.0 * np.pi * 5.0 * t)
    depth += rng.normal(0.0, 1.5, size=gx.shape)
    depth = np.where(water, np.clip(depth, 0.5, None), np.nan).astype(np.float32)

    if summer:
        nav_mask = (water & (np.nan_to_num(depth, nan=0.0) >= 20.0)).astype(np.float32)
    else:
        nav_mask = water_mask.copy()

    return environment_from_arrays(nav_mask, water_mask, depth, x_min, y_max, res)
