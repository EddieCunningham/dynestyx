"""Recovery metrics for simulated trajectories.

These compare a reconstructed posterior against the known simulated truth. They
quantify point-estimate accuracy, posterior spread, and agreement between the
estimated and true occupancy distributions.
"""

from __future__ import annotations

import numpy as np

from champlain_dsx.inference import normalized_weights, occupancy_map


def position_rmse(true_path: np.ndarray, est_path: np.ndarray) -> float:
    """Root mean squared distance between true and estimated positions."""
    true_xy = np.asarray(true_path)[:, :2]
    est_xy = np.asarray(est_path)[:, :2]
    return float(np.sqrt(np.mean(np.sum((true_xy - est_xy) ** 2, axis=1))))


def spatial_uncertainty(particles: np.ndarray, log_weights: np.ndarray) -> float:
    """Mean per-time posterior spread in metres.

    The spread at a time step is the square root of the summed weighted
    variances of the x and y coordinates. Averaging over time gives a single
    scalar describing how tightly the posterior localises the animal.
    """
    particles = np.asarray(particles)
    weights = normalized_weights(log_weights)
    mean = np.sum(weights[..., None] * particles[..., :2], axis=1, keepdims=True)
    var = np.sum(weights[..., None] * (particles[..., :2] - mean) ** 2, axis=1)
    return float(np.mean(np.sqrt(var.sum(axis=1))))


def true_occupancy_map(env, true_path: np.ndarray) -> np.ndarray:
    """Occupancy distribution of a single known trajectory."""
    path = np.asarray(true_path)
    particles = path[:, None, :]  # (T, 1, 3)
    log_weights = np.zeros((path.shape[0], 1))
    return occupancy_map(env, particles, log_weights)


def occupancy_total_variation(est_map: np.ndarray, true_map: np.ndarray) -> float:
    """Total variation distance between two occupancy distributions, in [0, 1]."""
    return float(0.5 * np.abs(np.asarray(est_map) - np.asarray(true_map)).sum())


def credible_region_coverage(
    env, particles: np.ndarray, log_weights: np.ndarray, true_path: np.ndarray,
    level: float = 0.9,
) -> float:
    """Fraction of time steps whose true cell lies in the level-mass region.

    At each time step the particle weights define an occupancy distribution over
    cells. The level-mass region is the smallest set of cells whose summed
    weight reaches ``level``. The metric reports how often the true position
    falls inside that region, a calibration-style check of the posterior.
    """
    particles = np.asarray(particles)
    weights = normalized_weights(log_weights)
    path = np.asarray(true_path)
    T, N = weights.shape

    cols = np.floor((particles[..., 0] - env.x_min) / env.res).astype(int)
    rows = np.floor((env.y_max - particles[..., 1]) / env.res).astype(int)
    cell = rows * env.width + cols

    true_col = np.floor((path[:, 0] - env.x_min) / env.res).astype(int)
    true_row = np.floor((env.y_max - path[:, 1]) / env.res).astype(int)
    true_cell = true_row * env.width + true_col

    hits = 0
    for t in range(T):
        order = np.argsort(weights[t])[::-1]
        cumw = np.cumsum(weights[t][order])
        keep = order[: np.searchsorted(cumw, level) + 1]
        region = set(cell[t][keep].tolist())
        hits += int(true_cell[t] in region)
    return hits / T
