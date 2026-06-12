"""Particle filtering and smoothing, plus occupancy maps.

The bootstrap particle filter targets the filtering distribution p(s_t | y_{1:t}).
The tracing particle smoother reweights filter genealogies to target the
marginal smoothing distribution p(s_t | y_{1:T}). Smoothed particles and weights
turn into an occupancy distribution over the lake grid, the discrete analogue of
the occurrence probability in the paper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

from dynestyx import Filter, Smoother
from dynestyx.inference.filters import PFConfig
from dynestyx.inference.smoothers import PFSmootherConfig


def run_filter(
    model,
    times,
    obs_values,
    key,
    n_particles: int = 5000,
    record_particles: bool = True,
) -> dict:
    """Run the bootstrap particle filter and return the recorded trace sites."""
    config = PFConfig(
        n_particles=n_particles,
        record_filtered_states_mean=True,
        record_filtered_states_cov_diag=True,
        record_filtered_particles=record_particles,
        record_filtered_log_weights=record_particles,
        record_max_elems=n_particles * len(times) * 4,
    )
    with Filter(filter_config=config):
        out = Predictive(model, num_samples=1, exclude_deterministic=False)(
            key, obs_times=jnp.asarray(times), obs_values=jnp.asarray(obs_values)
        )
    return out


def run_smoother(
    model,
    times,
    obs_values,
    key,
    n_particles: int = 5000,
    record_particles: bool = True,
) -> dict:
    """Run the tracing particle smoother and return the recorded trace sites."""
    config = PFSmootherConfig(
        n_particles=n_particles,
        pf_backward_sampling_method="tracing",
        record_smoothed_states_mean=True,
        record_smoothed_states_cov_diag=True,
        record_smoothed_particles=record_particles,
        record_smoothed_log_weights=record_particles,
        record_max_elems=n_particles * len(times) * 4,
    )
    with Smoother(smoother_config=config):
        out = Predictive(model, num_samples=1, exclude_deterministic=False)(
            key, obs_times=jnp.asarray(times), obs_values=jnp.asarray(obs_values)
        )
    return out


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """Convert per-time log weights, shape (T, N), to normalized weights."""
    lw = np.asarray(log_weights)
    lw = lw - lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def occupancy_map(env, particles: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    """Build a normalized occupancy distribution over the lake grid.

    Args:
        env: LakeEnvironment defining the grid.
        particles: Particle states, shape (T, N, 3).
        log_weights: Particle log weights, shape (T, N).

    Returns:
        Grid of shape (H, W) summing to one over navigable cells, the time-
        averaged probability of occupying each cell.
    """
    particles = np.asarray(particles)
    weights = normalized_weights(log_weights)
    T = particles.shape[0]

    cols = np.floor((particles[..., 0] - env.x_min) / env.res).astype(int)
    rows = np.floor((env.y_max - particles[..., 1]) / env.res).astype(int)
    in_bounds = (rows >= 0) & (rows < env.height) & (cols >= 0) & (cols < env.width)

    grid = np.zeros((env.height, env.width), dtype=np.float64)
    flat = grid.ravel()
    idx = rows * env.width + cols
    np.add.at(
        flat,
        idx[in_bounds],
        (weights[in_bounds] / T),
    )
    total = flat.sum()
    if total > 0:
        flat /= total
    return flat.reshape(env.height, env.width)


def smoothed_mean(out: dict) -> np.ndarray:
    """Extract the smoothed posterior mean path, shape (T, 3)."""
    return np.asarray(out["f_smoothed_states_mean"][0])


def filtered_mean(out: dict) -> np.ndarray:
    """Extract the filtered posterior mean path, shape (T, 3)."""
    return np.asarray(out["f_filtered_states_mean"][0])
