"""Assembly of the dynestyx state-space model for acoustic geolocation.

The model pairs the land-truncated correlated random walk with the acoustic
observation model and exposes a single dynestyx model function. The same
function drives simulation, filtering, and smoothing. Movement and observation
parameters are fixed to externally estimated values, matching the paper, which
calibrates them from telemetry and range-test data rather than inferring them
jointly with the trajectory.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

import dynestyx as dsx
from dynestyx import DynamicalModel

from champlain_dsx.distributions import CorrelatedRandomWalk, UniformOnWater
from champlain_dsx.observation import make_observation_model


@dataclasses.dataclass(frozen=True)
class ChamplainParameters:
    """Movement and observation parameters for the lake trout model.

    Defaults are the paper's best parameterisation. Step length follows a
    Gamma(shape, scale) truncated above by ``mobility``. The turning angle is a
    mixture of a truncated Normal with standard deviation ``sigma_heading`` and
    a uniform escape component with weight ``mix_uniform_weight``. Detection
    probability is logistic in distance with intercept ``alpha`` and slope
    ``beta``, cut off beyond ``gamma`` metres and where line of sight fails.
    """

    mobility: float = 216.0
    shape: float = 3.25
    scale: float = 25.0
    sigma_heading: float = 0.4
    mix_uniform_weight: float = 0.01
    alpha: float = 0.9039
    beta: float = -0.0021
    gamma: float = 7000.0
    n_attempts: int = 50


def build_dynamics(env, receivers, params: ChamplainParameters) -> DynamicalModel:
    """Build the dynestyx DynamicalModel for a lake, receiver array, and parameters."""
    receivers = jnp.asarray(receivers)
    cell_centres = env.water_cell_centres()

    initial_condition = UniformOnWater(cell_centres, env.res)

    def state_evolution(x, u, t_now, t_next):
        return CorrelatedRandomWalk(
            state=x,
            env=env,
            step_shape=jnp.asarray(params.shape),
            step_scale=jnp.asarray(params.scale),
            mobility=jnp.asarray(params.mobility),
            sigma_heading=jnp.asarray(params.sigma_heading),
            mix_uniform_weight=jnp.asarray(params.mix_uniform_weight),
            n_attempts=params.n_attempts,
        )

    observation_model = make_observation_model(
        env, receivers, params.alpha, params.beta, params.gamma
    )

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )


def make_model(env, receivers, params: ChamplainParameters | None = None):
    """Return a dynestyx model function bound to a lake, receivers, and parameters.

    The returned function follows the dynestyx contract. Pass ``predict_times``
    to simulate, or ``obs_times`` and ``obs_values`` to condition on detections
    for filtering or smoothing.
    """
    if params is None:
        params = ChamplainParameters()

    def model(
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
    ):
        dynamics = build_dynamics(env, receivers, params)
        return dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
        )

    return model
