"""Integration tests: simulation, particle-filter likelihood, and recovery."""

import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cholera import THETA_STAR
import inference as inf


def test_simulation_scale():
    # The Table 1 parameters should produce Dhaka-scale monthly deaths.
    _, _, obs, states = inf.simulate(THETA_STAR, n_months=180, key=jr.PRNGKey(0))
    assert obs.shape == (180,)
    assert 100.0 < float(obs.mean()) < 2000.0
    assert float(obs.max()) > 1500.0
    sfrac = states[:, 0] / 2.42e6
    assert 0.1 < float(sfrac.min()) and float(sfrac.max()) < 0.6


def test_loglik_finite_and_peaked():
    times, ctrl, obs, _ = inf.simulate(THETA_STAR, n_months=120, key=jr.PRNGKey(0))
    z0 = inf.default_initial_state(2.42e6)
    fp = inf.free_params(THETA_STAR)
    ll_true = inf.marginal_loglik(
        **fp, obs_times=times, obs_values=obs, ctrl_values=ctrl, z0=z0,
        key=jr.PRNGKey(1), n_particles=400,
    )
    fp_bad = dict(fp, eps=jnp.array(1.6))
    ll_bad = inf.marginal_loglik(
        **fp_bad, obs_times=times, obs_values=obs, ctrl_values=ctrl, z0=z0,
        key=jr.PRNGKey(1), n_particles=400,
    )
    assert jnp.isfinite(ll_true)
    assert float(ll_true) > float(ll_bad)


def test_profile_recovers_eps():
    times, ctrl, obs, _ = inf.simulate(THETA_STAR, n_months=180, key=jr.PRNGKey(0))
    z0 = inf.default_initial_state(2.42e6)
    grid = jnp.linspace(0.4, 1.4, 13)
    ll = inf.profile("eps", grid, THETA_STAR,
                     obs_times=times, obs_values=obs, ctrl_values=ctrl, z0=z0,
                     key=jr.PRNGKey(3), n_particles=400)
    eps_hat = float(grid[jnp.argmax(ll)])
    assert abs(eps_hat - float(THETA_STAR.eps)) < 0.25


if __name__ == "__main__":
    test_simulation_scale()
    test_loglik_finite_and_peaked()
    test_profile_recovers_eps()
    print("all model tests passed")
