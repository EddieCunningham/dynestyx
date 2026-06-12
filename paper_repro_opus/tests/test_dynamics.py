"""Smoke tests for the cholera dynamics: rollout shape, seasonality, gradients."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cholera import (
    THETA_STAR,
    CholeraTransition,
    euler_maruyama_month,
    observation_dist,
    transmission_rate,
    K_CLASSES,
)


def initial_state(P0):
    # Roughly 38% susceptible, a small seed of infectives, the rest immune.
    S0 = 0.38 * P0
    I0 = 200.0
    R = (P0 - S0 - I0) / 3.0
    return jnp.array([S0, I0, R, R, R, 0.0])


def rollout(key, n_months=240, P0=2.42e6):
    times = jnp.arange(n_months + 1, dtype=float)  # months
    u = jnp.array([P0, 0.0])  # constant population, no growth
    z0 = initial_state(P0)

    def step(carry, t_idx):
        z, k = carry
        k, sub = jax.random.split(k)
        t_now = times[t_idx]
        t_next = times[t_idx + 1]
        z_next = CholeraTransition(z, u, t_now, t_next, THETA_STAR).sample(sub)
        y = observation_dist(z_next, u, t_next, THETA_STAR.tau).sample(sub)
        return (z_next, k), (z_next, y)

    _, (zs, ys) = jax.lax.scan(step, (z0, key), jnp.arange(n_months))
    return times[1:], zs, ys


def test_single_transition_shape():
    z = initial_state(2.42e6)
    u = jnp.array([2.42e6, 0.0])
    z1 = CholeraTransition(z, u, 0.0, 1.0, THETA_STAR).sample(jax.random.PRNGKey(0))
    assert z1.shape == (K_CLASSES + 3,)
    assert jnp.all(z1[:5] >= 0.0)
    assert z1[5] >= 0.0  # nonnegative monthly deaths


def test_seasonality_present():
    times, zs, ys = rollout(jax.random.PRNGKey(1), n_months=240)
    deaths = zs[:, 5]
    # Cholera deaths should concentrate seasonally rather than stay flat.
    assert deaths.max() > 3.0 * (deaths.mean() + 1.0), (deaths.max(), deaths.mean())
    assert deaths.mean() > 10.0, deaths.mean()
    assert jnp.all(jnp.isfinite(zs))
    assert jnp.all(jnp.isfinite(ys))


def test_population_conserved_roughly():
    times, zs, ys = rollout(jax.random.PRNGKey(2), n_months=120, P0=2.42e6)
    total = zs[:, :5].sum(axis=1)
    # Total population should stay near P0 (births balance deaths).
    rel = jnp.abs(total - 2.42e6) / 2.42e6
    assert rel.max() < 0.1, rel.max()


def test_transition_differentiable():
    z = initial_state(2.42e6)
    u = jnp.array([2.42e6, 0.0])
    noise = jax.random.normal(jax.random.PRNGKey(3), (20,))

    def deaths_of_eps(eps):
        p = THETA_STAR._replace(eps=eps)
        zf = euler_maruyama_month(z, u, 0.0, 1.0, p, noise)
        return zf[5]

    g = jax.grad(deaths_of_eps)(0.8)
    assert jnp.isfinite(g)


if __name__ == "__main__":
    test_single_transition_shape()
    test_seasonality_present()
    test_population_conserved_roughly()
    test_transition_differentiable()
    times, zs, ys = rollout(jax.random.PRNGKey(1), n_months=240)
    deaths = zs[:, 5]
    print("monthly deaths: min %.1f mean %.1f max %.1f" % (deaths.min(), deaths.mean(), deaths.max()))
    print("susceptible fraction range: %.3f .. %.3f" % ((zs[:, 0] / 2.42e6).min(), (zs[:, 0] / 2.42e6).max()))
    print("infectious max: %.1f" % zs[:, 1].max())
    print("all dynamics tests passed")
