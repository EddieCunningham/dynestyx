"""Synthetic geography for the acoustic-telemetry state-space model.

The real study runs on a raster of Lake Champlain. Here we stand in a small
synthetic lake on a regular grid. A boolean mask marks water cells. The same
three operations the paper needs are provided in pure JAX so they can run inside
the particle filter.

- ``is_water(x, y)`` looks up whether a metric coordinate falls on a water cell.
- ``line_of_sight(x0, y0, x1, y1)`` approximates an unobstructed path by testing
  whether the midpoint between two points is on water, matching the Julia source
  ``in_line_of_sight``.
- ``region_of(x, y)`` labels each water cell with a management region for the
  residency diagnostic.

Coordinates are in metres. Cell ``(iy, ix)`` covers
``[x0 + ix * cell, x0 + (ix + 1) * cell) x [y0 + iy * cell, ...)`` and its centre
is at ``(x0 + (ix + 0.5) * cell, y0 + (iy + 0.5) * cell)``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int


class Geography(eqx.Module):
    """A gridded water mask with metric coordinates and management regions.

    Attributes:
        water: ``(ny, nx)`` boolean mask, ``True`` on water.
        region: ``(ny, nx)`` integer region label per cell, ``-1`` on land.
        n_regions: number of management regions.
        cell: cell size in metres.
        x0, y0: metric coordinate of the lower-left corner of cell ``(0, 0)``.
    """

    water: Bool[Array, "ny nx"]
    region: Int[Array, "ny nx"]
    n_regions: int = eqx.field(static=True)
    cell: float = eqx.field(static=True)
    x0: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)

    @property
    def ny(self) -> int:
        return self.water.shape[0]

    @property
    def nx(self) -> int:
        return self.water.shape[1]

    @property
    def width(self) -> float:
        return self.nx * self.cell

    @property
    def height(self) -> float:
        return self.ny * self.cell

    def _indices(
        self, x: Float[Array, "..."], y: Float[Array, "..."]
    ) -> tuple[Int[Array, "..."], Int[Array, "..."], Bool[Array, "..."]]:
        """Return integer cell indices and an in-bounds flag for coordinates."""
        ix = jnp.floor((x - self.x0) / self.cell).astype(jnp.int32)
        iy = jnp.floor((y - self.y0) / self.cell).astype(jnp.int32)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ixc = jnp.clip(ix, 0, self.nx - 1)
        iyc = jnp.clip(iy, 0, self.ny - 1)
        return ixc, iyc, inside

    def is_water(
        self, x: Float[Array, "..."], y: Float[Array, "..."]
    ) -> Bool[Array, "..."]:
        """True where a coordinate lies on a water cell, False off-grid or on land."""
        ix, iy, inside = self._indices(x, y)
        return inside & self.water[iy, ix]

    def line_of_sight(
        self,
        x0: Float[Array, "..."],
        y0: Float[Array, "..."],
        x1: Float[Array, "..."],
        y1: Float[Array, "..."],
    ) -> Bool[Array, "..."]:
        """Approximate an unobstructed path by testing the midpoint cell."""
        return self.is_water(0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def region_of(
        self, x: Float[Array, "..."], y: Float[Array, "..."]
    ) -> Int[Array, "..."]:
        """Region label for a coordinate, ``-1`` off-grid or on land."""
        ix, iy, inside = self._indices(x, y)
        lab = self.region[iy, ix]
        return jnp.where(inside, lab, -1)

    def water_cell_centres(self) -> Float[Array, "n_water 2"]:
        """Metric centres of all water cells, shape ``(n_water, 2)``."""
        iy, ix = jnp.nonzero(self.water)
        cx = self.x0 + (ix + 0.5) * self.cell
        cy = self.y0 + (iy + 0.5) * self.cell
        return jnp.stack([cx, cy], axis=-1)


def make_synthetic_lake(
    nx: int = 100,
    ny: int = 100,
    cell: float = 200.0,
    n_regions: int = 3,
) -> Geography:
    """Build a long, narrow synthetic lake with an island and a peninsula.

    The shape is a tilted elliptical basin oriented north-south, narrowed in the
    middle by a peninsula that intrudes from the west and pierced by a small
    island. These obstacles make the water-truncation of movement and the
    line-of-sight term in the detection model bite, which is the point of the
    synthetic case. Regions split the lake into ``n_regions`` north-south bands.
    """
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)  # (ny, nx)

    w = nx * cell
    h = ny * cell
    cx, cy = w / 2.0, h / 2.0

    # Elliptical basin, long axis north-south.
    a = 0.34 * w  # east-west half-width
    b = 0.46 * h  # north-south half-width
    water = ((gx - cx) / a) ** 2 + ((gy - cy) / b) ** 2 <= 1.0

    # Peninsula intruding from the west wall near mid-lake.
    pen = (
        (gx < cx + 0.10 * w)
        & (gx > cx - 0.34 * w)
        & (np.abs(gy - (cy - 0.06 * h)) < 0.05 * h)
    )
    water = water & ~pen

    # Small island in the northern basin.
    island = (gx - (cx + 0.07 * w)) ** 2 + (gy - (cy + 0.24 * h)) ** 2 <= (
        0.05 * w
    ) ** 2
    water = water & ~island

    # North-south region bands over the basin extent.
    y_lo, y_hi = cy - b, cy + b
    frac = np.clip((gy - y_lo) / (y_hi - y_lo), 0.0, 1.0 - 1e-9)
    region = np.floor(frac * n_regions).astype(np.int32)
    region = np.where(water, region, -1)

    return Geography(
        water=jnp.asarray(water),
        region=jnp.asarray(region),
        n_regions=int(n_regions),
        cell=float(cell),
        x0=0.0,
        y0=0.0,
    )


def default_receivers(geo: Geography) -> Float[Array, "n_receivers 2"]:
    """A sparse set of receiver coordinates placed on water across the lake.

    Placement is hand-picked relative to the lake extent so that ranges do not
    overlap and at least one receiver sits behind the peninsula, where the
    line-of-sight term matters. Any candidate that lands on land is dropped.
    """
    w, h = geo.width, geo.height
    cx, cy = w / 2.0, h / 2.0
    candidates = np.array(
        [
            [cx + 0.00 * w, cy + 0.40 * h],
            [cx - 0.12 * w, cy + 0.22 * h],
            [cx + 0.14 * w, cy + 0.10 * h],
            [cx - 0.10 * w, cy + 0.00 * h],
            [cx + 0.10 * w, cy - 0.12 * h],
            [cx - 0.06 * w, cy - 0.20 * h],
            [cx + 0.04 * w, cy - 0.34 * h],
            [cx - 0.02 * w, cy - 0.02 * h],
        ]
    )
    on_water = np.asarray(geo.is_water(jnp.asarray(candidates[:, 0]), jnp.asarray(candidates[:, 1])))
    return jnp.asarray(candidates[on_water])
