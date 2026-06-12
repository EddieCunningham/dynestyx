"""Synthetic lake domain and receiver layout."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass
class Domain:
    """Rectangular lake domain with acoustic receivers.

    Coordinates are in metres. The origin is the southwest corner.

    Attributes:
        width: East-west extent in metres.
        height: North-south extent in metres.
        receiver_locs: (K, 2) array of receiver positions.
        region_boundaries: north boundary of each south-to-north region split.
    """

    width: float
    height: float
    receiver_locs: Array
    region_boundaries: Array

    @property
    def n_receivers(self) -> int:
        return int(self.receiver_locs.shape[0])

    def in_domain(self, pos: Array) -> Array:
        """Return True if pos (2,) is inside the rectangular lake."""
        return (
            (pos[0] >= 0)
            & (pos[0] <= self.width)
            & (pos[1] >= 0)
            & (pos[1] <= self.height)
        )

    def region_index(self, pos: Array) -> Array:
        """Return the region index (0-based south-to-north) for pos (2,)."""
        return jnp.searchsorted(self.region_boundaries, pos[1])


def make_synthetic_lake(
    width: float = 20_000.0,
    height: float = 50_000.0,
    n_regions: int = 5,
    n_cols: int = 3,
    n_rows: int = 5,
) -> Domain:
    """Build a simple rectangular lake with a grid of receivers.

    The lake is 20 km wide and 50 km tall, matching the approximate scale
    of Lake Champlain. Receivers are placed on a regular grid, and the lake
    is divided into equal-height north-south regions.

    Args:
        width: Lake width in metres.
        height: Lake height in metres.
        n_regions: Number of equal-height regions south to north.
        n_cols: Number of receiver columns.
        n_rows: Number of receiver rows.

    Returns:
        Domain with receiver locations and region boundaries.
    """
    xs = jnp.linspace(width * 0.2, width * 0.8, n_cols)
    ys = jnp.linspace(height * 0.1, height * 0.9, n_rows)
    xx, yy = jnp.meshgrid(xs, ys)
    receiver_locs = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    region_boundaries = jnp.linspace(0, height, n_regions + 1)[1:-1]

    return Domain(
        width=float(width),
        height=float(height),
        receiver_locs=receiver_locs,
        region_boundaries=region_boundaries,
    )
