"""Receiver placement and trajectory simulation.

Simulation draws trajectories from the movement model and detections from the
observation model. The simulator is driven through the multi-trajectory lax.scan
path of dynestyx, which samples the transition and observation distributions
directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpyro.infer import Predictive

from dynestyx import DiscreteTimeSimulator


def grid_receivers(env, spacing: float = 4500.0, margin: float = 2000.0) -> jnp.ndarray:
    """Place receivers on a regular grid, keeping only those over open water.

    A regular array gives the lake reasonable acoustic coverage so simulated
    trajectories generate detections at several receivers.
    """
    xs = np.arange(env.x_min + margin, env.x_max, spacing)
    ys = np.arange(env.y_min + margin, env.y_max, spacing)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    on_water = np.asarray(
        jax.vmap(env.is_water)(jnp.asarray(cand[:, 0]), jnp.asarray(cand[:, 1]))
    ) > 0.5
    return jnp.asarray(cand[on_water])


def random_receivers(env, n_receivers: int, seed: int = 0) -> jnp.ndarray:
    """Draw receiver coordinates uniformly from navigable cells."""
    centres = np.asarray(env.water_cell_centres())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(centres), size=n_receivers, replace=False)
    return jnp.asarray(centres[idx])


def simulate(model, times, key, n_sim: int = 2) -> dict:
    """Simulate ``n_sim`` trajectories and their detections.

    Args:
        model: A dynestyx model function from ``make_model``.
        times: Observation times, shape (T,).
        key: PRNG key.
        n_sim: Number of independent trajectories. Use at least 2 so the
            simulator takes the vectorised lax.scan path.

    Returns:
        Dict with ``states`` of shape (n_sim, T, 3), ``observations`` of shape
        (n_sim, T, K), and ``times`` of shape (T,).
    """
    if n_sim < 2:
        raise ValueError("simulate requires n_sim >= 2 for the lax.scan path.")
    times = jnp.asarray(times)
    predictive = Predictive(model, num_samples=1, exclude_deterministic=False)
    with DiscreteTimeSimulator(n_simulations=n_sim):
        out = predictive(key, predict_times=times)
    return {
        "states": np.asarray(out["f_states"][0]),
        "observations": np.asarray(out["f_observations"][0]),
        "times": np.asarray(times),
    }
