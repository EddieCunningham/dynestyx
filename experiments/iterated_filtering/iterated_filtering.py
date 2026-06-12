"""Iterated filtering (IF1) for partially observed Markov processes.

This module implements the IF1 algorithm of Ionides, Breto and King (PNAS 2006),
"Inference for nonlinear dynamical systems", with the moment-matching update given
in the Annals of Statistics 2011 follow-up. IF1 is a plug-and-play maximum-likelihood
estimator. It needs only the ability to simulate the latent transition and evaluate the
observation density, both of which a dynestyx ``DynamicalModel`` exposes through
``.sample()`` and ``.log_prob()``.

The estimator repeatedly runs a bootstrap particle filter in which every particle carries
its own copy of the parameter vector. Across the observation times the parameter performs a
Gaussian random walk, and the random-walk standard deviation is cooled geometrically across
iterations. At each iteration the weighted filter means and prediction covariances of the
parameter particles drive a Newton-like update of the point estimate.

The algorithm builds a dedicated instrumented filter rather than reusing the package filter,
because IF1 needs the weighted parameter moments computed from the predicted particles before
resampling, and because each particle must propagate under its own parameter value. The filter
reuses the model simulation contract and the systematic resampling routine from cuthbertlib.

All perturbation and the parameter update happen on an unconstrained scale. A bijector maps
the unconstrained parameters to the constrained scale that ``make_model`` expects, so the
random walk never leaves the feasible set.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from cuthbertlib.resampling import systematic

# ---------------------------------------------------------------------------
# Parameter transforms (unconstrained <-> constrained)
# ---------------------------------------------------------------------------


class BoxTransform(eqx.Module):
    """Elementwise bijector between the real line and per-parameter intervals.

    A finite interval ``(lower, upper)`` is reached through a scaled logistic map. A
    component with infinite bounds on both sides is passed through unchanged. The forward
    map sends unconstrained values to constrained values, and the inverse map sends them
    back.
    """

    lower: jax.Array
    upper: jax.Array

    def __init__(self, lower, upper):
        self.lower = jnp.asarray(lower, dtype=float)
        self.upper = jnp.asarray(upper, dtype=float)

    @property
    def _finite(self) -> jax.Array:
        return jnp.isfinite(self.lower) & jnp.isfinite(self.upper)

    def forward(self, u: jax.Array) -> jax.Array:
        width = jnp.where(self._finite, self.upper - self.lower, 1.0)
        boxed = self.lower + width * jax.nn.sigmoid(u)
        return jnp.where(self._finite, boxed, u)

    def inverse(self, c: jax.Array) -> jax.Array:
        width = jnp.where(self._finite, self.upper - self.lower, 1.0)
        frac = jnp.clip((c - self.lower) / width, 1e-6, 1.0 - 1e-6)
        unboxed = jnp.log(frac) - jnp.log1p(-frac)
        return jnp.where(self._finite, unboxed, c)


def identity_transform(dim: int) -> BoxTransform:
    """Bijector that leaves every component unchanged."""
    inf = jnp.full((dim,), jnp.inf)
    return BoxTransform(lower=-inf, upper=inf)


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IF1Config:
    """Settings for an IF1 run.

    Attributes:
        n_iterations: Number of iterations ``M``.
        n_particles: Number of particles ``J`` in the embedded filter.
        cooling_fraction: Geometric cooling factor ``alpha`` in ``(0, 1)``. The random-walk
            standard deviation at iteration ``m`` (zero-indexed) is ``rw_sd * alpha**m``.
        initial_perturbation_scale: Multiplier ``tau`` on the random-walk standard deviation
            used to spread the parameter particles around the current estimate at the start
            of each iteration.
        ridge: Value added to the diagonal of every prediction covariance before inversion.
        resampling: Resampling scheme, ``"systematic"`` only at present.
    """

    n_iterations: int
    n_particles: int = 1000
    cooling_fraction: float = 0.95
    initial_perturbation_scale: float = 2.0
    ridge: float = 1e-6
    resampling: str = "systematic"


class IF1Result(NamedTuple):
    """Output of an IF1 run, all parameters on the constrained scale.

    Attributes:
        theta: Final parameter estimate, shape ``(p,)``.
        theta_history: Estimate at every iteration including the initial value, shape
            ``(M + 1, p)``.
        loglik_history: Particle-filter marginal log-likelihood estimate at each iteration,
            shape ``(M,)``.
    """

    theta: jax.Array
    theta_history: jax.Array
    loglik_history: jax.Array


# ---------------------------------------------------------------------------
# Internal filter state
# ---------------------------------------------------------------------------


class _Carry(NamedTuple):
    key: jax.Array
    x: jax.Array  # (J, ...) per-particle latent state
    theta: jax.Array  # (J, p) per-particle unconstrained parameter


class _StepOut(NamedTuple):
    theta_hat_F: jax.Array  # (p,) weighted mean of predicted parameters
    V_P: jax.Array  # (p, p) weighted covariance of predicted parameters
    log_norm: jax.Array  # () log mean weight, the per-step log-likelihood increment


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _cooling(rw_sd: jax.Array, alpha: float, m: jax.Array) -> jax.Array:
    """Random-walk standard deviation at iteration ``m`` (zero-indexed)."""
    return rw_sd * alpha**m


def _if1_moments(wnorm: jax.Array, theta: jax.Array):
    """Filter mean and prediction covariance of the predicted parameter particles.

    The filter mean is the importance-weighted mean of the predicted particles, which is the
    reduced-variance form of the filter mean in equation (29) of Ionides et al. The prediction
    covariance is the equal-weight spread of the same predicted particles, since after
    resampling at the previous step the particles carry uniform weight and the perturbation
    adds a zero-mean increment. Weighting the covariance by the importance weights would
    instead give the posterior spread and break the step-size scaling of the update.

    Args:
        wnorm: Normalized importance weights, shape ``(J,)``.
        theta: Predicted parameter particles, shape ``(J, p)``.

    Returns:
        A pair ``(theta_hat_F, V_P)`` with shapes ``(p,)`` and ``(p, p)``.
    """
    j = theta.shape[0]
    theta_hat_F = wnorm @ theta
    pred_mean = jnp.mean(theta, axis=0)
    centered = theta - pred_mean
    V_P = (centered.T @ centered) / j
    return theta_hat_F, V_P


def _eq21_update(
    theta_m: jax.Array,
    theta_hat_F: jax.Array,
    V_P: jax.Array,
    ridge: float,
) -> jax.Array:
    """Apply the IF1 parameter update of Ionides et al.

    The update is ``theta_{m+1} = theta_m + V^P_1 sum_n (V^P_n)^{-1}
    (theta_hat^F_n - theta_hat^F_{n-1})`` with ``theta_hat^F_0 = theta_m``.

    Args:
        theta_m: Current estimate, shape ``(p,)``.
        theta_hat_F: Per-step weighted filter means, shape ``(N, p)``.
        V_P: Per-step prediction covariances, shape ``(N, p, p)``.
        ridge: Diagonal loading added before inversion.

    Returns:
        The next estimate, shape ``(p,)``.
    """
    p = theta_m.shape[-1]
    V_P = V_P + ridge * jnp.eye(p)
    prev = jnp.concatenate([theta_m[None, :], theta_hat_F[:-1]], axis=0)
    diffs = theta_hat_F - prev  # (N, p)
    weighted = jnp.linalg.solve(V_P, diffs[..., None])[..., 0]  # (N, p)
    summed = jnp.sum(weighted, axis=0)  # (p,)
    return theta_m + V_P[0] @ summed


def _resample_indices(key: jax.Array, log_w: jax.Array) -> jax.Array:
    """Systematic resampling indices for the given log-weights."""
    n = log_w.shape[0]
    idx, _, _ = systematic.resampling(key, log_w, jnp.zeros(n), n)
    return idx


# ---------------------------------------------------------------------------
# One filtering pass and one iteration
# ---------------------------------------------------------------------------


def _build_step(make_model, transform, ctrl_values):
    """Build the per-observation scan step for a fixed cooling level.

    The returned function closes over ``make_model`` and ``transform`` and takes the cooling
    standard deviation as part of the scanned constants. It propagates each particle under
    that particle's own parameter, weights by the observation density, records the weighted
    parameter moments of the predicted particles, then resamples.
    """

    def propagate_one(key_j, x_j, theta_unc_j, u_prev, t_prev, t_now, is_first):
        model_j = make_model(transform.forward(theta_unc_j))

        def _first(_):
            return x_j

        def _evolve(_):
            d = model_j.state_evolution(x_j, u_prev, t_prev, t_now)
            return d.sample(key_j)

        return jax.lax.cond(is_first, _first, _evolve, operand=None)

    def logw_one(x_j, theta_unc_j, u_now, t_now, y_now):
        model_j = make_model(transform.forward(theta_unc_j))
        edist = model_j.observation_model(x_j, u_now, t_now)
        return jnp.asarray(edist.log_prob(y_now)).sum()

    def step(carry: _Carry, step_inputs):
        sigma_m, t_prev, t_now, u_prev, u_now, y_now, is_first = step_inputs
        key, x_prev, theta_prev = carry
        j = theta_prev.shape[0]
        k_perturb, k_prop, k_resample, k_next = jr.split(key, 4)

        eps = jr.normal(k_perturb, theta_prev.shape)
        theta_pred = theta_prev + sigma_m * eps

        keys_prop = jr.split(k_prop, j)
        x_new = eqx.filter_vmap(propagate_one, in_axes=(0, 0, 0, None, None, None, None))(
            keys_prop, x_prev, theta_pred, u_prev, t_prev, t_now, is_first
        )

        log_w = eqx.filter_vmap(logw_one, in_axes=(0, 0, None, None, None))(
            x_new, theta_pred, u_now, t_now, y_now
        )

        wnorm = jax.nn.softmax(log_w)
        theta_hat_F, V_P = _if1_moments(wnorm, theta_pred)
        log_norm = jax.nn.logsumexp(log_w) - jnp.log(j)

        idx = _resample_indices(k_resample, log_w)
        new_carry = _Carry(k_next, x_new[idx], theta_pred[idx])
        return new_carry, _StepOut(theta_hat_F, V_P, log_norm)

    return step


def _build_iteration(make_model, transform, config, step_inputs_no_sigma):
    """Build one IF1 iteration as a pure function of estimate, cooling, and key."""
    step = _build_step(make_model, transform, None)

    def init_particles(key, theta_m, sigma_init):
        k_state, k_theta = jr.split(key)
        p = theta_m.shape[-1]
        n = config.n_particles
        eps = jr.normal(k_theta, (n, p))
        theta0 = theta_m[None, :] + sigma_init[None, :] * eps

        def sample_x(key_j, theta_unc_j):
            return make_model(transform.forward(theta_unc_j)).initial_condition.sample(key_j)

        keys = jr.split(k_state, n)
        x0 = eqx.filter_vmap(sample_x)(keys, theta0)
        return x0, theta0

    def iteration(key, theta_m, sigma_m):
        k_init, k_scan = jr.split(key)
        sigma_init = config.initial_perturbation_scale * sigma_m
        x0, theta0 = init_particles(k_init, theta_m, sigma_init)

        n_steps = step_inputs_no_sigma[0].shape[0]
        sigma_seq = jnp.broadcast_to(sigma_m, (n_steps, sigma_m.shape[-1]))
        scan_inputs = (sigma_seq, *step_inputs_no_sigma)

        carry0 = _Carry(k_scan, x0, theta0)
        _, outs = jax.lax.scan(step, carry0, scan_inputs)
        loglik = jnp.sum(outs.log_norm)
        theta_next = _eq21_update(theta_m, outs.theta_hat_F, outs.V_P, config.ridge)
        return theta_next, loglik

    return iteration


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _prepare_step_inputs(obs_times, obs_values, ctrl_values, control_dim):
    """Assemble the per-observation scan inputs shared across all iterations."""
    n = obs_times.shape[0]
    if ctrl_values is None:
        ctrl_values = jnp.zeros((n, control_dim), dtype=obs_times.dtype)
    dt0 = obs_times[1] - obs_times[0]
    t_prev = jnp.concatenate([obs_times[:1] - dt0, obs_times[:-1]])
    u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)
    is_first = jnp.arange(n) == 0
    return (t_prev, obs_times, u_prev, ctrl_values, obs_values, is_first)


@eqx.filter_jit
def run_iterated_filtering(
    make_model: Callable,
    config: IF1Config,
    rng_key: jax.Array,
    init_theta: jax.Array,
    obs_times: jax.Array,
    obs_values: jax.Array,
    rw_sd: jax.Array,
    transform: BoxTransform | None = None,
    ctrl_values: jax.Array | None = None,
    control_dim: int = 0,
) -> IF1Result:
    """Estimate parameters of a POMP by iterated filtering.

    Args:
        make_model: Maps a constrained parameter vector of shape ``(p,)`` to a dynestyx
            ``DynamicalModel``. It must be a pure function with no Python branching on traced
            values, since it is evaluated under ``vmap`` once per particle.
        config: IF1 settings.
        rng_key: PRNG key.
        init_theta: Initial estimate on the constrained scale, shape ``(p,)``.
        obs_times: Observation times, shape ``(N,)``.
        obs_values: Observations, shape ``(N, obs_dim)``.
        rw_sd: Per-parameter random-walk standard deviation at the first iteration, defined on
            the unconstrained scale, shape ``(p,)``.
        transform: Bijector from unconstrained to constrained parameters. Identity by default.
        ctrl_values: Optional controls aligned to ``obs_times``, shape ``(N, control_dim)``.
        control_dim: Control dimension, used to build a zero control when ``ctrl_values`` is
            ``None``.

    Returns:
        An :class:`IF1Result` with all parameters on the constrained scale.
    """
    init_theta = jnp.asarray(init_theta, dtype=float)
    rw_sd = jnp.asarray(rw_sd, dtype=float)
    p = init_theta.shape[-1]
    if transform is None:
        transform = identity_transform(p)

    step_inputs = _prepare_step_inputs(obs_times, obs_values, ctrl_values, control_dim)
    iteration = _build_iteration(make_model, transform, config, step_inputs)

    theta0_unc = transform.inverse(init_theta)
    alpha = config.cooling_fraction

    def scan_iter(carry, m):
        key, theta_m = carry
        k_iter, k_next = jr.split(key)
        sigma_m = _cooling(rw_sd, alpha, m)
        theta_next, loglik = iteration(k_iter, theta_m, sigma_m)
        return (k_next, theta_next), (theta_next, loglik)

    (_, theta_final_unc), (theta_hist_unc, loglik_hist) = jax.lax.scan(
        scan_iter, (rng_key, theta0_unc), jnp.arange(config.n_iterations)
    )

    theta_hist_unc = jnp.concatenate([theta0_unc[None, :], theta_hist_unc], axis=0)
    theta_history = eqx.filter_vmap(transform.forward)(theta_hist_unc)
    theta_final = transform.forward(theta_final_unc)
    return IF1Result(theta_final, theta_history, loglik_hist)


class IF1Inference:
    """Object wrapper around :func:`run_iterated_filtering`.

    This mirrors the runner-class style of ``dynestyx.inference.mcmc.MCMCInference``. It holds
    the configuration, the model factory, and the parameter bijector, and exposes a ``run``
    method.
    """

    def __init__(
        self,
        config: IF1Config,
        make_model: Callable,
        transform: BoxTransform | None = None,
    ):
        self.config = config
        self.make_model = make_model
        self.transform = transform

    def run(
        self,
        rng_key: jax.Array,
        init_theta: jax.Array,
        obs_times: jax.Array,
        obs_values: jax.Array,
        rw_sd: jax.Array,
        ctrl_values: jax.Array | None = None,
        control_dim: int = 0,
    ) -> IF1Result:
        return run_iterated_filtering(
            self.make_model,
            self.config,
            rng_key,
            init_theta,
            obs_times,
            obs_values,
            rw_sd,
            transform=self.transform,
            ctrl_values=ctrl_values,
            control_dim=control_dim,
        )
