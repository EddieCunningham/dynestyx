"""Custom NumPyro distributions for the movement model.

Two distributions describe the latent state s_t = (x, y, phi), where (x, y) is a
projected position in metres and phi is a heading in radians.

`UniformOnWater` is the initial condition. It draws a position uniformly over
the navigable cells of the lake and a heading uniformly on (-pi, pi].

`CorrelatedRandomWalk` is the one-step transition. A step length is drawn from a
Gamma distribution truncated above by the mobility bound. A turning angle is
drawn from a mixture of a wrapped-normal-like truncated Normal and a uniform
escape component. The proposed position is rejected when it leaves navigable
water, and the draw repeats up to a fixed number of attempts.

Both distributions implement `sample` only. The bootstrap particle filter and
the tracing particle smoother consume the transition through `sample`, so the
intractable land-truncated density is never needed. `log_prob` raises.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro.distributions import constraints

_TWO_PI = 2.0 * math.pi
_PI = math.pi


def _wrap_angle(theta):
    """Wrap an angle to (-pi, pi]."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


class UniformOnWater(dist.Distribution):
    """Uniform initial condition over navigable cells and heading.

    Args:
        cell_centres: Array of shape (N, 2) with the (x, y) centres of navigable
            cells.
        res: Cell size in metres. Positions are jittered uniformly within the
            chosen cell.
    """

    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, cell_centres, res, validate_args=None):
        self.cell_centres = cell_centres
        self.res = res
        super().__init__(batch_shape=(), event_shape=(3,), validate_args=False)

    def _sample_one(self, key):
        k_cell, k_jx, k_jy, k_phi = jr.split(key, 4)
        n = self.cell_centres.shape[0]
        idx = jr.randint(k_cell, (), 0, n)
        base = self.cell_centres[idx]
        half = 0.5 * self.res
        jx = jr.uniform(k_jx, (), minval=-half, maxval=half)
        jy = jr.uniform(k_jy, (), minval=-half, maxval=half)
        phi = jr.uniform(k_phi, (), minval=-_PI, maxval=_PI)
        return jnp.stack([base[0] + jx, base[1] + jy, phi])

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape)
        if len(sample_shape) == 0:
            return self._sample_one(key)
        n = math.prod(int(s) for s in sample_shape)
        keys = jr.split(key, n)
        out = jax.vmap(self._sample_one)(keys)
        return out.reshape(sample_shape + (3,))

    def log_prob(self, value):
        raise NotImplementedError(
            "UniformOnWater is sample-only; the particle filter never calls log_prob."
        )

    def tree_flatten(self):
        return (self.cell_centres,), (self.res,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cell_centres,) = children
        (res,) = aux
        return cls(cell_centres, res)


class CorrelatedRandomWalk(dist.Distribution):
    """Land-truncated correlated random walk transition for s_t given s_{t-1}.

    Args:
        state: Current state (x, y, phi), shape (3,).
        env: A LakeEnvironment used for the navigability test.
        step_shape: Gamma shape k for the step length.
        step_scale: Gamma scale theta for the step length (mean k * theta).
        mobility: Upper truncation on the step length in metres.
        sigma_heading: Standard deviation of the truncated-Normal turning angle.
        mix_uniform_weight: Probability of drawing the turning angle from the
            uniform escape component instead of the truncated Normal.
        n_attempts: Number of rejection attempts before falling back to staying
            in place with a reoriented heading.
    """

    arg_constraints = {}
    support = constraints.real_vector

    def __init__(
        self,
        state,
        env,
        step_shape,
        step_scale,
        mobility,
        sigma_heading,
        mix_uniform_weight,
        n_attempts=50,
        validate_args=None,
    ):
        self.state = state
        self.env = env
        self.step_shape = step_shape
        self.step_scale = step_scale
        self.mobility = mobility
        self.sigma_heading = sigma_heading
        self.mix_uniform_weight = mix_uniform_weight
        self.n_attempts = n_attempts
        super().__init__(batch_shape=(), event_shape=(3,), validate_args=False)

    def _propose(self, key):
        k_mix, k_norm, k_unif, k_step = jr.split(key, 4)
        px, py, phi = self.state[0], self.state[1], self.state[2]

        use_uniform = jr.bernoulli(k_mix, self.mix_uniform_weight)
        dphi_normal = dist.TruncatedNormal(
            loc=0.0, scale=self.sigma_heading, low=-_PI, high=_PI
        ).sample(k_norm)
        dphi_uniform = jr.uniform(k_unif, (), minval=-_PI, maxval=_PI)
        dphi = jnp.where(use_uniform, dphi_uniform, dphi_normal)

        step = dist.Gamma(concentration=self.step_shape, rate=1.0 / self.step_scale).sample(k_step)

        heading = phi + dphi
        new_x = px + step * jnp.cos(heading)
        new_y = py + step * jnp.sin(heading)
        new_phi = _wrap_angle(heading)
        return step, new_x, new_y, new_phi

    def _sample_one(self, key):
        keys = jr.split(key, self.n_attempts)
        steps, xs, ys, phis = jax.vmap(self._propose)(keys)
        navigable = self.env.is_navigable(xs, ys) > 0.5
        within_reach = steps <= self.mobility
        valid = navigable & within_reach

        idx = jnp.argmax(valid)
        any_valid = jnp.any(valid)
        chosen = jnp.stack([xs[idx], ys[idx], phis[idx]])
        # If every attempt fails, hold position but reorient so the particle can
        # try a fresh direction at the next step.
        fallback = jnp.stack([self.state[0], self.state[1], phis[0]])
        return jnp.where(any_valid, chosen, fallback)

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape)
        if len(sample_shape) == 0:
            return self._sample_one(key)
        n = math.prod(int(s) for s in sample_shape)
        keys = jr.split(key, n)
        out = jax.vmap(self._sample_one)(keys)
        return out.reshape(sample_shape + (3,))

    def log_prob(self, value):
        raise NotImplementedError(
            "CorrelatedRandomWalk is sample-only; the land-truncated density is "
            "intractable and the bootstrap particle filter never calls log_prob."
        )

    def tree_flatten(self):
        children = (
            self.state,
            self.env,
            self.step_shape,
            self.step_scale,
            self.mobility,
            self.sigma_heading,
            self.mix_uniform_weight,
        )
        aux = (self.n_attempts,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (state, env, step_shape, step_scale, mobility, sigma_heading, mix_uniform_weight) = children
        (n_attempts,) = aux
        return cls(
            state,
            env,
            step_shape,
            step_scale,
            mobility,
            sigma_heading,
            mix_uniform_weight,
            n_attempts,
        )
