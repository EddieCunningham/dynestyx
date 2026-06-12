"""Model factories and data generation for the IF1 experiment.

Each ``make_*`` function maps a constrained parameter vector to a dynestyx
``DynamicalModel``. The free parameters are factored out of the corresponding numpyro models
in ``tests/models.py`` so the same dynamics can be rebuilt per particle inside the filter.

``simulate_pomp`` draws a single trajectory and its observations directly from a model through
the same simulation contract the filter uses, which keeps the data-generating process and the
inference model exactly aligned.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.models.lti_dynamics import LTI_discrete

from experiments.iterated_filtering.iterated_filtering import BoxTransform

_NO_CONTROL = jnp.zeros(0)


# ---------------------------------------------------------------------------
# Linear-Gaussian model (free alpha = A[0, 0]); exact MLE available via Kalman filter
# ---------------------------------------------------------------------------

_LG_Q = 0.1 * jnp.eye(2)
_LG_H = jnp.array([[1.0, 0.0]])
_LG_R = jnp.array([[0.1**2]])


def make_lti_model(theta: jax.Array) -> DynamicalModel:
    """Discrete linear-Gaussian model with free coupling ``alpha = A[0, 0]``."""
    alpha = theta[0]
    A = jnp.array([[alpha, 0.1], [0.1, 0.8]])
    return LTI_discrete(A=A, Q=_LG_Q, H=_LG_H, R=_LG_R)


def lti_transform() -> BoxTransform:
    return BoxTransform(lower=jnp.array([-0.7]), upper=jnp.array([0.7]))


# ---------------------------------------------------------------------------
# Stochastic volatility (free phi); non-Gaussian observation, needs the particle filter
# ---------------------------------------------------------------------------

_SV_SIGMA_ETA = 0.5


def make_stoch_vol_model(theta: jax.Array) -> DynamicalModel:
    """Stochastic volatility with free AR(1) persistence ``phi``."""
    phi = theta[0]

    def state_evolution(x, u, t_now, t_next):
        return dist.Normal(phi * x, _SV_SIGMA_ETA)

    def observation_model(x, u, t):
        return dist.Normal(0.0, jnp.exp(x / 2.0))

    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )


def stoch_vol_transform() -> BoxTransform:
    return BoxTransform(lower=jnp.array([0.0]), upper=jnp.array([1.0]))


# ---------------------------------------------------------------------------
# Discrete Lorenz 63 (free rho); nonlinear dynamics, Gaussian partial observation
# ---------------------------------------------------------------------------

_L63_DT = 0.01
_L63_Q = 0.01 * jnp.eye(3)
_L63_H = jnp.array([[1.0, 0.0, 0.0]])
_L63_R = jnp.array([[1.0**2]])
_L63_IC = dist.MultivariateNormal(loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3))


def make_l63_model(theta: jax.Array) -> DynamicalModel:
    """Discrete-time Lorenz 63 with free drift parameter ``rho``, observing the first coordinate."""
    rho = theta[0]

    def drift(x):
        return jnp.array(
            [
                10.0 * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - (8.0 / 3.0) * x[2],
            ]
        )

    def state_evolution(x, u, t_now, t_next):
        loc = x + _L63_DT * drift(x)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=_L63_Q)

    return DynamicalModel(
        control_dim=0,
        initial_condition=_L63_IC,
        state_evolution=state_evolution,
        observation_model=LinearGaussianObservation(H=_L63_H, R=_L63_R),
    )


def l63_transform() -> BoxTransform:
    return BoxTransform(lower=jnp.array([10.0]), upper=jnp.array([40.0]))


# ---------------------------------------------------------------------------
# Direct simulation through the model contract
# ---------------------------------------------------------------------------


def simulate_pomp(
    make_model: Callable,
    theta: jax.Array,
    key: jax.Array,
    times: jax.Array,
):
    """Draw one trajectory and its observations from ``make_model(theta)``.

    The first observation is generated from the initial-condition sample, and each later
    observation follows a transition step. This matches the generative assumption used by the
    iterated filter.

    Returns:
        A pair ``(obs_values, states)``. ``obs_values`` has shape ``(N,)`` for scalar
        observations or ``(N, obs_dim)`` otherwise. ``states`` stacks the latent states.
    """
    model = make_model(theta)
    n = times.shape[0]
    dt0 = times[1] - times[0]
    t_prev = jnp.concatenate([times[:1] - dt0, times[:-1]])
    is_first = jnp.arange(n) == 0

    k_init, k_scan = jr.split(key)
    x0 = model.initial_condition.sample(k_init)

    def step(carry, inp):
        x_prev, key = carry
        t_p, t_n, first = inp
        k_state, k_obs, k_next = jr.split(key, 3)
        x_cur = jax.lax.cond(
            first,
            lambda: x_prev,
            lambda: model.state_evolution(x_prev, _NO_CONTROL, t_p, t_n).sample(k_state),
        )
        y_cur = model.observation_model(x_cur, _NO_CONTROL, t_n).sample(k_obs)
        return (x_cur, k_next), (y_cur, x_cur)

    _, (obs_values, states) = jax.lax.scan(step, (x0, k_scan), (t_prev, times, is_first))
    return obs_values, states
