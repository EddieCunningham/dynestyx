"""The cholera model wired into dynestyx as a NumPyro model function.

A single model function serves simulation and inference. Free parameters are
sampled with the `obs=` pattern so that passing a value fixes the parameter
(for simulation or for profiling) and passing `None` exposes it for inference.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx import DynamicalModel

from cholera import THETA_STAR, CholeraParams, CholeraTransition, observation_dist, K_CLASSES

STATE_DIM = K_CLASSES + 3


def make_initial_condition(z0):
    """Point-mass initial condition at the known starting state."""
    return dist.Delta(z0, event_dim=1)


def cholera_model(
    b=None,
    log_omega=None,
    eps=None,
    log_tau=None,
    z0=None,
    fixed: CholeraParams = THETA_STAR,
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    """Cholera state-space model.

    The free parameters exposed for inference are the seasonal coefficients `b`,
    the log reservoir strength `log_omega`, the environmental noise `eps`, and
    the log measurement overdispersion `log_tau`. Remaining parameters are held
    at `fixed`. Pass a concrete value for any parameter to clamp it.
    """
    b = numpyro.sample(
        "b", dist.Normal(0.0, 5.0).expand([6]).to_event(1), obs=b if b is not None else None
    )
    log_omega = numpyro.sample(
        "log_omega", dist.Normal(jnp.log(1e-4), 2.0), obs=log_omega
    )
    eps = numpyro.sample("eps", dist.Uniform(0.0, 2.0), obs=eps)
    log_tau = numpyro.sample("log_tau", dist.Normal(jnp.log(0.25), 0.5), obs=log_tau)

    params = fixed._replace(
        b=b, omega=jnp.exp(log_omega), eps=eps, tau=jnp.exp(log_tau)
    )

    if z0 is None:
        z0 = _default_initial_state(ctrl_values)

    dynamics = DynamicalModel(
        initial_condition=make_initial_condition(z0),
        state_evolution=lambda x, u, t_now, t_next: CholeraTransition(
            x, u, t_now, t_next, params
        ),
        observation_model=lambda x, u, t: observation_dist(x, u, t, params.tau),
        control_dim=2,
    )
    return dsx.sample(
        "chol",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )


def _default_initial_state(ctrl_values):
    P0 = 2.42e6 if ctrl_values is None else ctrl_values[0, 0]
    S0 = 0.38 * P0
    I0 = 200.0
    R = (P0 - S0 - I0) / 3.0
    return jnp.array([S0, I0, R, R, R, 0.0])
