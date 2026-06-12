"""Convenience wrappers for simulating data and running the particle algorithms.

These functions wrap the NumPyro and dynestyx boilerplate so that the notebook
and the tests share one code path. ``simulate_dataset`` draws a trajectory and
its detections from the model. ``run_filter`` runs the bootstrap particle filter
and returns the marginal log-likelihood and filtered means. ``run_smoother``
runs the particle smoother and returns the weighted particle cloud used to build
occurrence maps and residency estimates.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpyro.handlers import seed
from numpyro.infer import Predictive

from dynestyx import DiscreteTimeSimulator, Filter, Smoother
from dynestyx.inference.filters import PFConfig
from dynestyx.inference.smoothers import PFSmootherConfig


def simulate_dataset(
    model,
    n_steps: int,
    key,
    *,
    name: str = "fish",
    min_detections: int = 0,
    max_attempts: int = 50,
):
    """Draw a trajectory and detections from the model.

    When ``min_detections`` is positive the draw is repeated with fresh keys
    until the dataset has at least that many detections or ``max_attempts`` is
    exhausted, so downstream tests have a usable signal.

    Returns a dict with ``obs_times``, ``obs_values``, ``states`` and the integer
    seed that produced the dataset.
    """
    obs_times = jnp.arange(n_steps)
    predictive = Predictive(model, num_samples=1, exclude_deterministic=False)
    keys = jr.split(key, max_attempts)
    chosen = None
    for i in range(max_attempts):
        with DiscreteTimeSimulator():
            out = predictive(keys[i], predict_times=obs_times)
        obs = out[f"{name}_observations"][0, 0]
        if int(np.asarray(obs).sum()) >= min_detections:
            chosen = (i, out)
            break
        chosen = (i, out)
    i, out = chosen
    return {
        "obs_times": obs_times,
        "obs_values": jnp.asarray(out[f"{name}_observations"][0, 0]),
        "states": np.asarray(out[f"{name}_states"][0, 0]),
        "seed": i,
        "n_detections": int(np.asarray(out[f"{name}_observations"][0, 0]).sum()),
    }


def run_filter(
    model,
    obs_times,
    obs_values,
    key,
    *,
    name: str = "fish",
    n_particles: int = 2000,
):
    """Run the bootstrap particle filter and return the filtered summaries."""
    config = PFConfig(n_particles=n_particles, record_filtered_states_mean=True)
    with seed(rng_seed=key):
        with Filter(filter_config=config):
            out = Predictive(model, num_samples=1, exclude_deterministic=False)(
                key, obs_times=obs_times, obs_values=obs_values
            )
    return {
        "marginal_loglik": float(np.asarray(out[f"{name}_marginal_loglik"]).ravel()[0]),
        "filtered_mean": np.asarray(out[f"{name}_filtered_states_mean"][0]),
    }


def run_smoother(
    model,
    obs_times,
    obs_values,
    key,
    *,
    name: str = "fish",
    n_particles: int = 3000,
    n_smoother_particles: int = 400,
):
    """Run the particle smoother and return the weighted particle cloud.

    The returned ``particles`` and ``log_weights`` approximate the marginal
    smoothing distribution ``f(s_t | y_{1:T})`` at every time step, which is the
    target the paper maps.
    """
    config = PFSmootherConfig(
        n_particles=n_particles,
        pf_n_smoother_particles=n_smoother_particles,
        record_smoothed_particles=True,
        record_smoothed_log_weights=True,
        record_smoothed_states_mean=True,
    )
    with seed(rng_seed=key):
        with Smoother(smoother_config=config):
            out = Predictive(model, num_samples=1, exclude_deterministic=False)(
                key, obs_times=obs_times, obs_values=obs_values
            )
    return {
        "marginal_loglik": float(np.asarray(out[f"{name}_marginal_loglik"]).ravel()[0]),
        "particles": jnp.asarray(out[f"{name}_smoothed_particles"][0]),
        "log_weights": jnp.asarray(out[f"{name}_smoothed_log_weights"][0]),
        "smoothed_mean": np.asarray(out[f"{name}_smoothed_states_mean"][0]),
    }
