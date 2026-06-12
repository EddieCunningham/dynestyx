"""Helpers for simulating, loading data, and scoring the cholera model.

Time is measured in months throughout, matching the paper. A single
particle-filter pass returns the marginal log-likelihood at one parameter
vector. Profiling sweeps that score over a grid.
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd

from numpyro.infer import Predictive

from dynestyx import DiscreteTimeSimulator, Filter
from dynestyx.inference.filter_configs import PFConfig, PFResamplingConfig

from cholera import THETA_STAR, CholeraParams
from model import cholera_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dhaka_cholera.csv")


def default_initial_state(P0, sfrac=0.38, I0=500.0):
    """Compartment counts at the start of the record."""
    S0 = sfrac * P0
    R = (P0 - S0 - I0) / 3.0
    return jnp.array([S0, I0, R, R, R, 0.0])


def free_params(theta: CholeraParams):
    """Unpack the inference parameters from a CholeraParams in model order."""
    return dict(
        b=theta.b,
        log_omega=jnp.log(theta.omega),
        eps=theta.eps,
        log_tau=jnp.log(theta.tau),
    )


def make_sim_grid(n_months=180, P0=2.42e6):
    times = jnp.arange(n_months, dtype=float)
    ctrl = jnp.stack([jnp.full(n_months, P0), jnp.zeros(n_months)], axis=-1)
    return times, ctrl


def simulate(theta: CholeraParams, n_months=180, P0=2.42e6, key=jr.PRNGKey(0)):
    """Draw one trajectory and its monthly cholera-death observations."""
    times, ctrl = make_sim_grid(n_months, P0)
    z0 = default_initial_state(P0)
    predictive = Predictive(cholera_model, num_samples=1, exclude_deterministic=False)
    with DiscreteTimeSimulator():
        pred = predictive(
            key, z0=z0, predict_times=times, ctrl_times=times, ctrl_values=ctrl,
            **free_params(theta),
        )
    obs = pred["chol_observations"][0, 0].reshape(-1)
    states = pred["chol_states"][0, 0]
    return times, ctrl, obs, states


def load_dhaka():
    """Load the Dhaka cholera series with a month index and population control."""
    df = pd.read_csv(DATA_PATH)
    months = jnp.asarray((df.time.values - df.time.values[0]) * 12.0)
    deaths = jnp.asarray(df.cholera_deaths.values.astype(float))
    pop = jnp.asarray(df.population.values)
    dpop = jnp.gradient(pop)  # population change per month
    ctrl = jnp.stack([pop, dpop], axis=-1)
    return df, months, deaths, pop, ctrl


def marginal_loglik(
    b, log_omega, eps, log_tau, *, obs_times, obs_values, ctrl_values, z0,
    key=jr.PRNGKey(0), n_particles=500, differentiable=False,
):
    """Particle-filter marginal log-likelihood at one parameter vector."""
    resampling = (
        PFResamplingConfig(differential_method="straight_through")
        if differentiable
        else PFResamplingConfig()
    )
    with Filter(PFConfig(n_particles=n_particles, crn_seed=key, resampling_method=resampling)):
        out = Predictive(cholera_model, num_samples=1, exclude_deterministic=False)(
            key, b=b, log_omega=log_omega, eps=eps, log_tau=log_tau, z0=z0,
            obs_times=obs_times, obs_values=obs_values,
            ctrl_times=obs_times, ctrl_values=ctrl_values,
        )
    return out["chol_marginal_loglik"].reshape(-1)[0]


def make_eps_tau_loglik(base: CholeraParams, *, obs_times, obs_values, ctrl_values, z0,
                        n_particles=500):
    """A compiled (eps, log_tau, key) -> marginal log-likelihood with b, omega fixed.

    Fixing the seed at call time lets each call draw fresh particle noise, which
    is what a pseudo-marginal sampler needs for an unbiased likelihood estimate.
    The function compiles once and is reused across many proposals.
    """
    b = base.b
    log_omega = jnp.log(base.omega)

    @eqx.filter_jit
    def fn(eps, log_tau, key):
        with Filter(PFConfig(n_particles=n_particles, crn_seed=key)):
            out = Predictive(cholera_model, num_samples=1, exclude_deterministic=False)(
                key, b=b, log_omega=log_omega, eps=eps, log_tau=log_tau, z0=z0,
                obs_times=obs_times, obs_values=obs_values,
                ctrl_times=obs_times, ctrl_values=ctrl_values,
            )
        return out["chol_marginal_loglik"].reshape(-1)[0]

    return fn


def pmmh_eps_tau(loglik_fn, *, n_steps=200, step=(0.06, 0.06), eps0=0.8, log_tau0=None,
                 tau_prior=(np.log(0.25), 0.5), seed=0):
    """Particle marginal Metropolis-Hastings over (eps, log_tau).

    A random-walk proposal is accepted using the unbiased particle-filter
    likelihood. The chain targets the exact posterior despite the likelihood
    being estimated. Returns an array of (eps, tau) draws and the acceptance
    rate.
    """
    log_tau0 = tau_prior[0] if log_tau0 is None else log_tau0
    rng = np.random.default_rng(seed)
    key = jr.PRNGKey(1000 + seed)

    def log_prior(eps, log_tau):
        if eps <= 0.0 or eps >= 2.0:
            return -np.inf
        return -0.5 * ((log_tau - tau_prior[0]) / tau_prior[1]) ** 2

    eps, lt = eps0, log_tau0
    key, k = jr.split(key)
    ll = float(loglik_fn(eps, lt, k))
    lp = log_prior(eps, lt)
    draws, n_acc = [], 0
    for _ in range(n_steps):
        eps_p = eps + step[0] * rng.standard_normal()
        lt_p = lt + step[1] * rng.standard_normal()
        lpp = log_prior(eps_p, lt_p)
        if np.isfinite(lpp):
            key, k = jr.split(key)
            llp = float(loglik_fn(eps_p, lt_p, k))
            if np.log(rng.random()) < (llp + lpp) - (ll + lp):
                eps, lt, ll, lp = eps_p, lt_p, llp, lpp
                n_acc += 1
        draws.append((eps, np.exp(lt)))
    return np.array(draws), n_acc / n_steps


def profile(vary, grid, base: CholeraParams, *, obs_times, obs_values, ctrl_values, z0,
            key=jr.PRNGKey(7), n_particles=500):
    """Profile the marginal log-likelihood over one named parameter.

    `vary` is one of "b0", "eps", "log_omega", "log_tau". Other parameters stay
    at `base`. Common random numbers (a fixed filter seed) make the curve smooth.
    """
    fp = free_params(base)

    def score(x):
        kw = dict(fp)
        if vary == "b0":
            kw["b"] = base.b.at[0].set(x)
        elif vary == "eps":
            kw["eps"] = x
        elif vary == "log_omega":
            kw["log_omega"] = x
        elif vary == "log_tau":
            kw["log_tau"] = x
        else:
            raise ValueError(vary)
        return marginal_loglik(
            **kw, obs_times=obs_times, obs_values=obs_values,
            ctrl_values=ctrl_values, z0=z0, key=key, n_particles=n_particles,
        )

    return jax.vmap(score)(grid)
