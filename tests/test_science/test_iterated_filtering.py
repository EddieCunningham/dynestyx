"""Science test for IF2 iterated filtering.

Uses a 1D linear-Gaussian model where the MLE is known. Verifies that IF2
recovers the true parameter within tolerance and that the log-likelihood trace
is non-NaN.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

from dynestyx.inference.iterated_filtering import IF2Config, IF2Result, run_if2


# ---------------------------------------------------------------------------
# 1D linear-Gaussian model
# x_{t+1} = alpha * x_t + eps,  eps ~ N(0, sigma_x^2)
# y_t     = x_t + eta,           eta ~ N(0, sigma_y^2)
# theta = {'log_alpha': log(alpha)}   (unconstrained)
# ---------------------------------------------------------------------------

TRUE_ALPHA = 0.7
SIGMA_X = 0.3
SIGMA_Y = 0.2
X0_MEAN = 0.0
X0_STD = 1.0


def _make_model_fns():
    """Return (init_fn, transition_fn, log_obs_fn) for the 1D LG model."""

    def init_fn(key, theta):
        return dist.Normal(X0_MEAN, X0_STD).sample(key)[None]  # shape (1,)

    def transition_fn(key, x, theta, t_prev, t_next):
        alpha = jnp.exp(theta["log_alpha"])
        mean = alpha * x[0]
        return dist.Normal(mean, SIGMA_X).sample(key)[None]  # shape (1,)

    def log_obs_fn(y, x, theta):
        return dist.Normal(x[0], SIGMA_Y).log_prob(y[0])

    return init_fn, transition_fn, log_obs_fn


def _simulate_data(key, n_steps: int = 200):
    """Simulate observations from the true 1D LG model."""
    key_x, key_proc, key_obs = jr.split(key, 3)

    def step(x, noise):
        x_next = TRUE_ALPHA * x + SIGMA_X * noise
        return x_next, x_next

    x0 = jr.normal(key_x) * X0_STD + X0_MEAN
    proc_noises = jr.normal(key_proc, shape=(n_steps,))
    _, xs = jax.lax.scan(step, x0, proc_noises)

    obs_noise = jr.normal(key_obs, shape=(n_steps,)) * SIGMA_Y
    ys = xs + obs_noise

    obs_times = jnp.arange(1, n_steps + 1, dtype=jnp.float32)
    obs_values = ys[:, None]  # shape (N, 1)
    return obs_times, obs_values


@pytest.mark.slow
def test_if2_recovers_true_alpha():
    """IF2 should recover TRUE_ALPHA = 0.7 within tolerance on 1D LG data.

    With N=200 observations and J=2000 particles the selection pressure is
    strong enough to overcome the parameter diffusion from 200 inner steps per
    outer iteration. rw_sds=0.02 keeps the per-step diffusion to ≈0.2 std
    per outer iteration in log-alpha space.
    """
    key = jr.PRNGKey(0)
    key_data, key_if2 = jr.split(key)

    obs_times, obs_values = _simulate_data(key_data, n_steps=200)
    init_fn, transition_fn, log_obs_fn = _make_model_fns()

    theta_0 = {"log_alpha": jnp.log(jnp.array(0.6))}  # start near truth
    rw_sds = {"log_alpha": jnp.array(0.02)}

    config = IF2Config(n_particles=2000, n_iterations=100, cooling_fraction_50=0.5)

    result = run_if2(
        key_if2,
        init_fn,
        transition_fn,
        log_obs_fn,
        theta_0,
        rw_sds,
        obs_times,
        obs_values,
        config=config,
    )

    assert isinstance(result, IF2Result)
    assert result.log_lik_trace.shape == (100,)
    assert not jnp.isnan(result.log_lik_trace).any(), "log_lik_trace contains NaN"
    assert not jnp.isinf(result.log_lik_trace).any(), "log_lik_trace contains Inf"

    # Log-likelihood should generally trend upward (last > first)
    assert result.log_lik_trace[-1] > result.log_lik_trace[0], (
        f"log-lik did not improve: {result.log_lik_trace[0]:.2f} -> {result.log_lik_trace[-1]:.2f}"
    )

    # Parameter estimate should be close to the true alpha within sampling noise.
    log_alpha_swarm = result.theta_swarm["log_alpha"]
    assert log_alpha_swarm.shape == (2000,)
    alpha_estimate = jnp.exp(log_alpha_swarm.mean())
    assert jnp.abs(alpha_estimate - TRUE_ALPHA) < 0.15, (
        f"alpha estimate {alpha_estimate:.3f} too far from true {TRUE_ALPHA}"
    )


def test_if2_shapes_and_no_nan():
    """Smoke test: IF2 runs without NaN on tiny data (no slow mark)."""
    key = jr.PRNGKey(1)
    key_data, key_if2 = jr.split(key)

    obs_times, obs_values = _simulate_data(key_data, n_steps=20)
    init_fn, transition_fn, log_obs_fn = _make_model_fns()

    theta_0 = {"log_alpha": jnp.log(jnp.array(0.5))}
    rw_sds = {"log_alpha": jnp.array(0.05)}

    config = IF2Config(n_particles=50, n_iterations=5, cooling_fraction_50=0.5)

    result = run_if2(
        key_if2,
        init_fn,
        transition_fn,
        log_obs_fn,
        theta_0,
        rw_sds,
        obs_times,
        obs_values,
        config=config,
    )

    assert result.log_lik_trace.shape == (5,)
    assert result.theta_swarm["log_alpha"].shape == (50,)
    assert not jnp.isnan(result.log_lik_trace).any()
    assert not jnp.isnan(result.theta_swarm["log_alpha"]).any()


def test_if2_flat_array_theta():
    """IF2 works when theta is a flat jnp.ndarray rather than a dict."""
    key = jr.PRNGKey(2)
    key_data, key_if2 = jr.split(key)

    obs_times, obs_values = _simulate_data(key_data, n_steps=20)

    def init_fn(key, theta):
        return dist.Normal(X0_MEAN, X0_STD).sample(key)[None]

    def transition_fn(key, x, theta, t_prev, t_next):
        alpha = jnp.exp(theta[0])
        return dist.Normal(alpha * x[0], SIGMA_X).sample(key)[None]

    def log_obs_fn(y, x, theta):
        return dist.Normal(x[0], SIGMA_Y).log_prob(y[0])

    theta_0 = jnp.log(jnp.array([0.5]))
    rw_sds = jnp.array([0.05])

    config = IF2Config(n_particles=50, n_iterations=5, cooling_fraction_50=0.5)

    result = run_if2(
        key_if2,
        init_fn,
        transition_fn,
        log_obs_fn,
        theta_0,
        rw_sds,
        obs_times,
        obs_values,
        config=config,
    )

    assert result.theta_swarm.shape == (50, 1)
    assert not jnp.isnan(result.log_lik_trace).any()
