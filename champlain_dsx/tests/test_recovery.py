"""End-to-end recovery tests for the particle filter and smoother."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from champlain_dsx.environment import synthetic_champlain
from champlain_dsx.inference import occupancy_map, run_filter, run_smoother
from champlain_dsx.metrics import (
    credible_region_coverage,
    occupancy_total_variation,
    position_rmse,
    spatial_uncertainty,
    true_occupancy_map,
)
from champlain_dsx.model import ChamplainParameters, make_model
from champlain_dsx.simulate import grid_receivers, simulate


def _setup(seed=0, T=120):
    env = synthetic_champlain(res=600.0)
    receivers = grid_receivers(env, spacing=4500.0)
    model = make_model(env, receivers, ChamplainParameters())
    times = jnp.arange(T, dtype=jnp.float32)
    sim = simulate(model, times, jr.PRNGKey(seed), n_sim=2)
    # Use the trajectory that generated more detections so the test is not a
    # degenerate zero-information case.
    counts = sim["observations"].sum(axis=(1, 2))
    j = int(np.argmax(counts))
    return env, model, times, sim["states"][j], sim["observations"][j]


def test_smoother_localises_better_than_the_prior():
    env, model, times, true_path, obs_values = _setup(seed=1)
    out = run_smoother(model, times, obs_values, jr.PRNGKey(10), n_particles=3000)
    est = np.asarray(out["f_smoothed_states_mean"][0])
    rmse = position_rmse(true_path, est)
    # The lake spans tens of kilometres. A useful reconstruction localises the
    # animal to a few kilometres.
    lake_scale = env.y_max - env.y_min
    assert rmse < 0.1 * lake_scale
    assert rmse < 4000.0


def test_filter_and_smoother_run_and_record_particles():
    env, model, times, true_path, obs_values = _setup(seed=2, T=60)
    fout = run_filter(model, times, obs_values, jr.PRNGKey(11), n_particles=1500)
    assert fout["f_filtered_particles"].shape[1:] == (60, 1500, 3)
    sout = run_smoother(model, times, obs_values, jr.PRNGKey(12), n_particles=1500)
    parts = np.asarray(sout["f_smoothed_particles"][0])
    lw = np.asarray(sout["f_smoothed_log_weights"][0])
    assert parts.shape == (60, 1500, 3)
    assert np.all(np.isfinite(lw))


def test_smoothed_particles_stay_on_water():
    env, model, times, true_path, obs_values = _setup(seed=3, T=60)
    sout = run_smoother(model, times, obs_values, jr.PRNGKey(13), n_particles=1500)
    parts = np.asarray(sout["f_smoothed_particles"][0])  # (T, N, 3)
    flat = parts.reshape(-1, 3)
    import jax
    nav = np.asarray(
        jax.vmap(env.is_navigable)(jnp.asarray(flat[:, 0]), jnp.asarray(flat[:, 1]))
    )
    # Essentially all particle mass lives on navigable water.
    assert nav.mean() > 0.98


def test_occupancy_metrics_are_sensible():
    env, model, times, true_path, obs_values = _setup(seed=4, T=90)
    sout = run_smoother(model, times, obs_values, jr.PRNGKey(14), n_particles=3000)
    parts = np.asarray(sout["f_smoothed_particles"][0])
    lw = np.asarray(sout["f_smoothed_log_weights"][0])

    est_map = occupancy_map(env, parts, lw)
    np.testing.assert_allclose(est_map.sum(), 1.0, atol=1e-6)

    true_map = true_occupancy_map(env, true_path)
    tv = occupancy_total_variation(est_map, true_map)
    assert 0.0 <= tv <= 1.0

    spread = spatial_uncertainty(parts, lw)
    assert spread > 0.0

    coverage = credible_region_coverage(env, parts, lw, true_path, level=0.9)
    # The true path should fall inside the 90% region most of the time.
    assert coverage > 0.5
