"""Occurrence maps and regional residency from trajectories and particles.

The paper summarises a reconstructed track in two ways. An occurrence
distribution is a map of where the animal spent its time. Residency is the
fraction of time spent in each management region. We compute both from a true
simulated trajectory and from the weighted particle cloud of a smoother, so the
two can be compared as the correctness signal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .geography import Geography


def occurrence_map(
    geo: Geography,
    xy: Float[Array, "n 2"],
    weights: Float[Array, "n"] | None = None,
) -> Float[Array, "ny nx"]:
    """Bin locations onto the grid to form a normalised occurrence map.

    Args:
        geo: the geography that defines the grid.
        xy: locations, shape ``(n, 2)``.
        weights: optional non-negative weights per location. Uniform if ``None``.

    Returns:
        A ``(ny, nx)`` array of probability mass summing to one over water.
    """
    if weights is None:
        weights = jnp.ones(xy.shape[0])
    ix = jnp.clip(jnp.floor((xy[:, 0] - geo.x0) / geo.cell).astype(jnp.int32), 0, geo.nx - 1)
    iy = jnp.clip(jnp.floor((xy[:, 1] - geo.y0) / geo.cell).astype(jnp.int32), 0, geo.ny - 1)
    flat = iy * geo.nx + ix
    grid = jnp.zeros(geo.ny * geo.nx).at[flat].add(weights)
    grid = grid.reshape(geo.ny, geo.nx)
    total = grid.sum()
    return grid / jnp.where(total > 0, total, 1.0)


def residency_from_track(geo: Geography, xy: Float[Array, "T 2"]) -> Float[Array, "n_regions"]:
    """Fraction of time steps a track spends in each region."""
    labels = geo.region_of(xy[:, 0], xy[:, 1])
    counts = jnp.array([jnp.sum(labels == r) for r in range(geo.n_regions)])
    return counts / jnp.maximum(counts.sum(), 1)


def residency_from_particles(
    geo: Geography,
    particles: Float[Array, "T n 2"],
    log_weights: Float[Array, "T n"],
) -> Float[Array, "n_regions"]:
    """Expected fraction of time in each region from weighted smoother particles.

    For each time step the per-region probability is the weighted share of
    particles falling in that region. Averaging over time gives the residency.
    """
    w = jax.nn.softmax(log_weights, axis=1)  # (T, n)
    labels = geo.region_of(particles[..., 0], particles[..., 1])  # (T, n)
    per_region = jnp.stack(
        [jnp.sum(jnp.where(labels == r, w, 0.0), axis=1) for r in range(geo.n_regions)],
        axis=-1,
    )  # (T, n_regions)
    return per_region.mean(axis=0)


def occurrence_from_particles(
    geo: Geography,
    particles: Float[Array, "T n 2"],
    log_weights: Float[Array, "T n"],
) -> Float[Array, "ny nx"]:
    """Occurrence map from the full weighted particle cloud across time."""
    w = jax.nn.softmax(log_weights, axis=1)  # (T, n)
    xy = particles.reshape(-1, 2)
    weights = w.reshape(-1)
    return occurrence_map(geo, xy, weights)


def occupancy_area_cells(
    occ: Float[Array, "ny nx"], mass: float = 0.95
) -> int:
    """Number of cells in the smallest set holding ``mass`` of the probability.

    This is the paper's spatial-uncertainty diagnostic, the area spanned by 95%
    of the probability mass.
    """
    flat = jnp.sort(occ.reshape(-1))[::-1]
    cumulative = jnp.cumsum(flat)
    return int(jnp.sum(cumulative < mass) + 1)
