"""Iterated filtering (IF2) for maximum likelihood in nonlinear POMPs.

Implements the IF2 algorithm from:
  Ionides, Bretó, King (2006). Inference for nonlinear dynamical systems. PNAS.
  Ionides et al. (2015). Inference for dynamic and latent variable models. JASA.

The algorithm maximizes the likelihood of a partially observed Markov process by
augmenting the latent state with parameters that undergo a decreasing random walk,
running a particle filter on the augmented system, and extracting the terminal
parameter distribution as the MLE approximation.

IF2 is plug-and-play: it requires only a simulator for the transition density and
an evaluator for the observation density.
"""

import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array


@dataclasses.dataclass
class IF2Config:
    """Configuration for the IF2 iterated filtering algorithm.

    Attributes:
        n_particles: Number of parameter/state particles J.
        n_iterations: Number of outer iterations M. The parameter swarm
            contracts toward the MLE as m increases.
        cooling_fraction_50: Fraction of the initial random walk standard
            deviation remaining after 50 outer iterations. The cooling factor
            at iteration m is cooling_fraction_50^(m/50).
    """

    n_particles: int = 1_000
    n_iterations: int = 50
    cooling_fraction_50: float = 0.5


class IF2Result(NamedTuple):
    """Output of the IF2 algorithm.

    Attributes:
        theta_swarm: Final parameter swarm as a pytree where every leaf has a
            leading dimension of size n_particles.
        log_lik_trace: Estimated log p(y_{1:N}) at each outer iteration,
            shape (n_iterations,).
    """

    theta_swarm: Any
    log_lik_trace: Array


def _systematic_resample(key: Array, log_weights: Array) -> Array:
    """Systematic resampling returning integer indices of shape (J,).

    Args:
        key: JAX PRNG key.
        log_weights: Unnormalized log importance weights, shape (J,).

    Returns:
        Integer resampling indices, shape (J,).
    """
    j = log_weights.shape[0]
    cumsum = jnp.cumsum(jax.nn.softmax(log_weights))
    u = jr.uniform(key) / j + jnp.arange(j) / j
    return jnp.searchsorted(cumsum, u)


def _perturb_theta(key: Array, theta: Any, rw_sds: Any, cool: Array) -> Any:
    """Add scaled Gaussian perturbations to every leaf of theta.

    Args:
        key: JAX PRNG key.
        theta: Pytree of parameter arrays, each with a leading J dim.
        rw_sds: Pytree of random walk standard deviations, same structure as
            a single-particle theta (no leading J dim).
        cool: Scalar cooling factor multiplied into rw_sds.

    Returns:
        Perturbed theta with the same pytree structure and shapes.
    """
    leaves, treedef = jax.tree.flatten(theta)
    sd_leaves, _ = jax.tree.flatten(rw_sds)
    keys = jr.split(key, len(leaves))
    perturbed = [
        leaf + cool * sd * jr.normal(k, shape=leaf.shape)
        for leaf, sd, k in zip(leaves, sd_leaves, keys)
    ]
    return treedef.unflatten(perturbed)


def run_if2(
    key: Array,
    init_fn: Callable,
    transition_fn: Callable,
    log_obs_fn: Callable,
    theta_0: Any,
    rw_sds: Any,
    obs_times: Array,
    obs_values: Array,
    config: IF2Config = IF2Config(),
    ctrl_values: Array | None = None,
) -> IF2Result:
    """Run iterated filtering (IF2) to maximize the likelihood of a POMP model.

    IF2 is plug-and-play: it requires only a state simulator (init_fn,
    transition_fn) and an observation log-probability evaluator (log_obs_fn).
    No transition density evaluation is needed.

    The algorithm follows Ionides, Bretó, King (2006) PNAS and the updated IF2
    variant from Ionides et al. (2015) JASA. At each outer iteration, the
    parameter swarm is perturbed with decreasing Gaussian noise, a particle
    filter assimilates the observations, and the terminal swarm approximates
    the MLE.

    Args:
        key: JAX PRNG key.
        init_fn: Callable (key, theta) -> x0 of shape (state_dim,). Samples
            the initial state for a given parameter value.
        transition_fn: Callable (key, x, theta, t_prev, t_next) -> x_next of
            shape (state_dim,). Simulates one step of the latent process.
        log_obs_fn: Callable (y, x, theta) -> scalar. Evaluates the log
            observation density log p(y_t | x_t, theta) at a single time step.
        theta_0: Pytree of initial parameter values (no leading J dim). All
            leaves must be JAX arrays.
        rw_sds: Pytree of random walk standard deviations with the same
            structure as theta_0.
        obs_times: Observation times, shape (N,).
        obs_values: Observation values, shape (N, obs_dim) or (N,).
        config: IF2Config controlling n_particles, n_iterations, cooling.
        ctrl_values: Unused; reserved for future control-input support.

    Returns:
        IF2Result with final parameter swarm and per-iteration log-likelihood.

    Example:
        Using a DynamicalModel via eqx.partition / eqx.combine::

            import equinox as eqx
            params, static = eqx.partition(dynamics, eqx.is_array)

            def init_fn(key, theta):
                return eqx.combine(theta, static).initial_condition.sample(key)

            def transition_fn(key, x, theta, t_prev, t_next):
                d = eqx.combine(theta, static)
                return d.state_evolution(x, None, t_prev, t_next).sample(key)

            def log_obs_fn(y, x, theta):
                d = eqx.combine(theta, static)
                return d.observation_model(x, None, 0.0).log_prob(y).sum()

            result = run_if2(
                key, init_fn, transition_fn, log_obs_fn,
                params, rw_sds, obs_times, obs_values,
            )
    """
    j = config.n_particles
    m_total = config.n_iterations
    a = config.cooling_fraction_50

    obs_times = jnp.asarray(obs_times)
    obs_values = jnp.asarray(obs_values)
    n_steps = obs_times.shape[0]

    # t_prev_all[n] is the time from which particles are propagated to reach
    # obs_times[n].  For n=0, use one nominal time step before the first
    # observation so discrete-time models (where t doesn't matter) work without
    # special-casing.
    dt0 = jnp.where(n_steps > 1, obs_times[1] - obs_times[0], jnp.ones(()))
    t_prev_first = obs_times[0] - dt0
    t_prev_all = jnp.concatenate([t_prev_first[None], obs_times[:-1]])

    # Build the initial parameter swarm: J copies of theta_0.
    theta_swarm_init = jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf)[None], (j,) + jnp.asarray(leaf).shape
        ),
        theta_0,
    )

    def outer_step(carry: tuple, m: Array) -> tuple:
        """One outer iteration of IF2 (Algorithm 3, Ionides et al. 2015)."""
        theta_swarm, key = carry

        cool = jnp.asarray(a, dtype=jnp.float32) ** (m / 50.0)

        key, k_perturb_init, k_init, k_inner = jr.split(key, 4)

        # Line 2: perturb initial parameter swarm.
        theta_swarm = _perturb_theta(k_perturb_init, theta_swarm, rw_sds, cool)

        # Line 3: initialize state particles from the perturbed parameters.
        init_keys = jr.split(k_init, j)
        x0 = jax.vmap(init_fn, in_axes=(0, 0))(init_keys, theta_swarm)

        def inner_step(inner_carry: tuple, inputs: tuple) -> tuple:
            """One step of the particle filter (Lines 5–9 of Algorithm 3)."""
            x, theta, key = inner_carry
            y, t, tp = inputs

            key, k_perturb, k_prop, k_resample = jr.split(key, 4)

            # Line 5: perturb parameters; cool closes over outer scope.
            theta = _perturb_theta(k_perturb, theta, rw_sds, cool)

            # Line 6: propagate state.
            prop_keys = jr.split(k_prop, j)
            x_next = jax.vmap(transition_fn, in_axes=(0, 0, 0, None, None))(
                prop_keys, x, theta, tp, t
            )

            # Line 7: compute log-weights.
            log_w = jax.vmap(log_obs_fn, in_axes=(None, 0, 0))(y, x_next, theta)

            # Estimate marginal log p(y_t | y_{1:t-1}).
            log_lik_t = jax.nn.logsumexp(log_w) - jnp.log(j)

            # Lines 8–9: systematic resample.
            indices = _systematic_resample(k_resample, log_w)
            x_next = x_next[indices]
            theta = jax.tree.map(lambda leaf: leaf[indices], theta)

            return (x_next, theta, key), log_lik_t

        init_inner = (x0, theta_swarm, k_inner)
        inner_inputs = (obs_values, obs_times, t_prev_all)
        (_, theta_final, _), log_liks = jax.lax.scan(
            inner_step, init_inner, inner_inputs
        )

        total_log_lik = jnp.sum(log_liks)

        # Line 11: save terminal swarm.
        return (theta_final, key), (theta_final, total_log_lik)

    init_outer = (theta_swarm_init, key)
    _, (swarms, log_lik_trace) = jax.lax.scan(
        outer_step, init_outer, jnp.arange(1, m_total + 1, dtype=jnp.float32)
    )

    # swarms has shape (M, J, ...) per leaf; return the last iteration's swarm.
    theta_final = jax.tree.map(lambda s: s[-1], swarms)

    return IF2Result(theta_swarm=theta_final, log_lik_trace=log_lik_trace)
