"""The discrete-time state-space model for acoustic-telemetry geolocation.

This is the model of Lavender et al., equations 1 through 9, expressed for
``dynestyx``. The latent state is ``s_t = (x, y, phi)``, a two-dimensional
location and a heading. Time is discrete with a fixed step (two minutes in the
paper). All parameters are fixed a priori. Inference targets the latent states.

Three pieces define the model.

- ``UniformOnWater`` is the initial condition: location uniform over the water
  cells, heading uniform on ``(-pi, pi)`` (eqn 3).
- ``MoveModel`` is the land-truncated correlated random walk (eqns 4 to 6). It
  returns a sampling-only transition distribution. The next location is a
  deterministic function of a step length and a turning angle, so the density in
  ``(x, y, phi)`` space is degenerate. The bootstrap particle filter and the
  genealogy-tracing particle smoother need only to sample the transition, which
  is what we provide.
- ``DetectionModel`` is the observation process (eqns 7 to 9): an independent
  Bernoulli per receiver whose detection probability decays logistically with
  distance, is truncated beyond a maximum range, and requires a line of sight.

``build_model`` assembles these into a NumPyro model function suitable for
simulation, filtering and smoothing.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

import dynestyx as dsx
from dynestyx import DynamicalModel

from .geography import Geography


# --------------------------------------------------------------------------- #
# Model parameters (the paper's "best" parameterisation).
# --------------------------------------------------------------------------- #


class MoveParams(eqx.Module):
    """Movement-model parameters (eqns 5 and 6)."""

    shape: float = 3.25  # truncated Gamma shape k
    scale: float = 25.0  # truncated Gamma scale theta (metres per step)
    mobility: float = 216.0  # maximum step length (metres)
    sigma: float = 0.4  # turning-angle standard deviation (radians)
    p_uniform: float = 0.01  # mixture weight on the uniform turning component
    max_tries: int = eqx.field(static=True, default=64)  # rejection-sampling budget


class ObsParams(eqx.Module):
    """Acoustic detection-probability parameters (eqn 9)."""

    alpha: float = 0.904
    beta: float = -0.002
    gamma: float = 7000.0  # maximum detection range (metres)


def _wrap_angle(a: Float[Array, "..."]) -> Float[Array, "..."]:
    """Wrap an angle to ``(-pi, pi]``."""
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# --------------------------------------------------------------------------- #
# Initial condition: uniform over water.
# --------------------------------------------------------------------------- #


class UniformOnWater(dist.Distribution):
    """Uniform location over water cells with a uniform heading.

    A draw picks a water cell uniformly at random, places the location uniformly
    within that cell, and draws a heading uniformly on ``(-pi, pi)``. The event
    is the three-vector ``(x, y, phi)``.
    """

    arg_constraints: dict = {}
    support = dist.constraints.real_vector

    def __init__(self, water_centres: Float[Array, "n 2"], cell: float, *, validate_args=None):
        self.water_centres = water_centres
        self.cell = cell
        super().__init__(batch_shape=(), event_shape=(3,), validate_args=validate_args)

    def _sample_one(self, key: Array) -> Float[Array, "3"]:
        k_cell, k_jit, k_phi = jax.random.split(key, 3)
        n = self.water_centres.shape[0]
        idx = jax.random.randint(k_cell, (), 0, n)
        centre = self.water_centres[idx]
        jitter = jax.random.uniform(k_jit, (2,), minval=-0.5 * self.cell, maxval=0.5 * self.cell)
        phi = jax.random.uniform(k_phi, (), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.array([centre[0] + jitter[0], centre[1] + jitter[1], phi])

    def sample(self, key: Array, sample_shape: tuple = ()):
        if sample_shape == ():
            return self._sample_one(key)
        n = math.prod(sample_shape)
        keys = jax.random.split(key, n)
        out = jax.vmap(self._sample_one)(keys)
        return out.reshape(tuple(sample_shape) + (3,))

    def log_prob(self, value):  # pragma: no cover - not used by the chosen inference
        raise NotImplementedError("UniformOnWater is a sampling-only initial condition.")

    def tree_flatten(self):
        return (self.water_centres,), (self.cell,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (water_centres,) = children
        (cell,) = aux_data
        return cls(water_centres, cell)


# --------------------------------------------------------------------------- #
# Movement: land-truncated correlated random walk.
# --------------------------------------------------------------------------- #


class _CRWTransition(dist.Distribution):
    """Sampling-only transition for the land-truncated correlated random walk.

    Given the previous state ``(x, y, phi)``, a draw proposes a step length from
    a Gamma distribution and a turning angle from a normal-uniform mixture, then
    moves to the new location and heading by equation 4. A proposal is rejected
    when the step exceeds the mobility bound or when the new location is not on
    water, and is redrawn up to ``max_tries`` times. If no valid step is found
    the walker stays in place, which is the safe fallback for a particle hemmed
    in by land.
    """

    arg_constraints: dict = {}
    support = dist.constraints.real_vector

    def __init__(self, x_prev: Float[Array, "3"], geo: Geography, params: MoveParams, *, validate_args=None):
        self.x_prev = x_prev
        self.geo = geo
        self.params = params
        super().__init__(batch_shape=(), event_shape=(3,), validate_args=validate_args)

    def _sample_one(self, key: Array) -> Float[Array, "3"]:
        p = self.params
        x0, y0, phi0 = self.x_prev[0], self.x_prev[1], self.x_prev[2]
        rate = 1.0 / p.scale

        def cond(carry):
            i, _key, found, _state = carry
            return (~found) & (i < p.max_tries)

        def body(carry):
            i, key, found, state = carry
            key, k_d, k_mix, k_n, k_u = jax.random.split(key, 5)
            d = jax.random.gamma(k_d, p.shape) / rate
            dphi_normal = p.sigma * jax.random.normal(k_n)
            dphi_uniform = jax.random.uniform(k_u, (), minval=-jnp.pi, maxval=jnp.pi)
            pick_uniform = jax.random.bernoulli(k_mix, p.p_uniform)
            dphi = _wrap_angle(jnp.where(pick_uniform, dphi_uniform, dphi_normal))
            phi_new = _wrap_angle(phi0 + dphi)
            x_new = x0 + d * jnp.cos(phi_new)
            y_new = y0 + d * jnp.sin(phi_new)
            valid = (d <= p.mobility) & self.geo.is_water(x_new, y_new)
            new_state = jnp.where(valid, jnp.array([x_new, y_new, phi_new]), state)
            return (i + 1, key, found | valid, new_state)

        init = (jnp.int32(0), key, jnp.bool_(False), self.x_prev)
        _, _, _, state = jax.lax.while_loop(cond, body, init)
        return state

    def sample(self, key: Array, sample_shape: tuple = ()):
        if sample_shape == ():
            return self._sample_one(key)
        n = math.prod(sample_shape)
        keys = jax.random.split(key, n)
        out = jax.vmap(self._sample_one)(keys)
        return out.reshape(tuple(sample_shape) + (3,))

    def log_prob(self, value):  # pragma: no cover - degenerate density, not used
        raise NotImplementedError(
            "The correlated-random-walk transition is sampling-only. Its density "
            "in (x, y, phi) is degenerate, so the bootstrap filter and the "
            "tracing particle smoother use only sampling."
        )

    def tree_flatten(self):
        return (self.x_prev, self.geo, self.params), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x_prev, geo, params = children
        return cls(x_prev, geo, params)


class MoveModel(eqx.Module):
    """Discrete-time state evolution callable returning a CRW transition."""

    geo: Geography
    params: MoveParams = MoveParams()

    def __call__(self, x, u, t_now, t_next) -> _CRWTransition:
        return _CRWTransition(x, self.geo, self.params)


# --------------------------------------------------------------------------- #
# Observation: per-receiver Bernoulli detections.
# --------------------------------------------------------------------------- #


class DetectionModel(eqx.Module):
    """Observation model: an independent Bernoulli detection per receiver."""

    geo: Geography
    receivers: Float[Array, "n_receivers 2"]
    params: ObsParams = ObsParams()

    def detection_prob(self, x, y) -> Float[Array, "n_receivers"]:
        """Per-receiver detection probability for a location (eqn 9)."""
        p = self.params
        rx = self.receivers[:, 0]
        ry = self.receivers[:, 1]
        d = jnp.sqrt((rx - x) ** 2 + (ry - y) ** 2)
        prob = jax.nn.sigmoid(p.alpha + p.beta * d)
        los = self.geo.line_of_sight(x, y, rx, ry)
        visible = (d < p.gamma) & los
        return jnp.where(visible, prob, 0.0)

    def __call__(self, x, u, t):
        prob = self.detection_prob(x[0], x[1])
        return dist.Independent(dist.Bernoulli(probs=prob), 1)


# --------------------------------------------------------------------------- #
# Assemble the model.
# --------------------------------------------------------------------------- #


def build_model(
    geo: Geography,
    receivers: Float[Array, "n_receivers 2"],
    move_params: MoveParams = MoveParams(),
    obs_params: ObsParams = ObsParams(),
    name: str = "fish",
):
    """Build a NumPyro model function for the geolocation state-space model.

    The returned function is reused for simulation, filtering and smoothing. It
    takes ``obs_times`` and ``obs_values`` to condition on detections and
    ``predict_times`` to roll the model forward and generate data.
    """
    water_centres = geo.water_cell_centres()

    def model(
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
    ):
        dynamics = DynamicalModel(
            initial_condition=UniformOnWater(water_centres, geo.cell),
            state_evolution=MoveModel(geo, move_params),
            observation_model=DetectionModel(geo, receivers, obs_params),
        )
        return dsx.sample(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
        )

    return model
