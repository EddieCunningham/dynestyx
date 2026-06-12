"""Cholera compartment model of Ionides, Breto and King (2006) in dynestyx.

The latent state is a continuous-time stochastic SIRS process with k Erlang
immune classes, environmental noise on the susceptible-to-infected channel, and
seasonally forced transmission. Observations are monthly cholera deaths with
overdispersed Gaussian measurement noise.

The state vector carried by the filter is

    z = [S, I, R_1, ..., R_k, M],

where M holds the cholera deaths accrued over the most recent observation
interval. Each transition recomputes M from zero, so the monthly death count is
reset automatically without a framework-level accumulator hook.
"""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from jaxtyping import Array, Float

from seasonal import periodic_bspline_basis

K_CLASSES = 3
N_SUBSTEPS = 20  # Euler-Maruyama steps per month, dt = 1/20 month
NBASIS = 6
PERIOD = 12.0  # annual period, with time measured in months

# Free and fixed parameters of the transmission model. Rates are per month.
CholeraParams = namedtuple(
    "CholeraParams",
    ["b", "omega", "eps", "gamma", "m_c", "m", "r", "tau"],
)

# Table 1 generating parameters (theta*) from the paper.
THETA_STAR = CholeraParams(
    b=jnp.array([-0.58, 4.73, -5.76, 2.37, 1.69, 2.56]),
    omega=1.76e-4,
    eps=0.80,
    gamma=1.0 / 0.75,
    m_c=0.046,
    m=1.0 / 600.0,
    r=1.0 / 120.0,
    tau=0.25,
)


def transmission_rate(t: Float[Array, "..."], b: Float[Array, " nbasis"]) -> Float[Array, "..."]:
    """Seasonal transmission beta(t) = exp(sum_k b_k s_k(t)), t in months."""
    basis = periodic_bspline_basis(t, nbasis=NBASIS, degree=3, period=PERIOD)
    return jnp.exp(basis @ b)


def euler_maruyama_month(z, u, t_now, t_next, p: CholeraParams, noise):
    """Integrate one observation interval with N_SUBSTEPS Euler-Maruyama steps.

    `u` is the control [P, dP/dt] (population and its monthly rate of change).
    `noise` is a vector of N_SUBSTEPS standard normal draws driving the
    environmental term. Returns the next state with M holding the interval's
    cholera deaths.
    """
    P, dPdt = u[0], u[1]
    dt = (t_next - t_now) / N_SUBSTEPS
    kr = K_CLASSES * p.r

    def step(carry, inputs):
        S, I, R1, R2, R3 = carry
        sub_t, zeta = inputs
        beta = transmission_rate(sub_t, p.b)
        lam = beta * I / P + p.omega  # force of infection

        births = (dPdt + p.m * P) * dt
        dN_SI = lam * S * dt + p.eps * (I / P) * S * jnp.sqrt(dt) * zeta
        dN_SD = p.m * S * dt
        dN_IR1 = p.gamma * I * dt
        dN_IC = p.m_c * I * dt
        dN_ID = p.m * I * dt
        loss_imm = kr * R3 * dt

        S_new = S + births - dN_SI - dN_SD + loss_imm
        I_new = I + dN_SI - dN_IR1 - dN_IC - dN_ID
        R1_new = R1 + dN_IR1 - kr * R1 * dt - p.m * R1 * dt
        R2_new = R2 + kr * R1 * dt - kr * R2 * dt - p.m * R2 * dt
        R3_new = R3 + kr * R2 * dt - kr * R3 * dt - p.m * R3 * dt

        new = jnp.stack([S_new, I_new, R1_new, R2_new, R3_new])
        new = jnp.maximum(new, 0.0)  # counts stay nonnegative
        return tuple(new), dN_IC

    sub_times = t_now + dt * jnp.arange(N_SUBSTEPS)
    carry0 = tuple(z[:5])
    carry, deaths = jax.lax.scan(step, carry0, (sub_times, noise))
    S, I, R1, R2, R3 = carry
    M = deaths.sum()  # cholera deaths over the interval, reset each call
    return jnp.stack([S, I, R1, R2, R3, M])


class CholeraTransition(dist.Distribution):
    """One-month transition kernel sampled by Euler-Maruyama integration.

    Forward sampling only. The bootstrap particle filter and the simulator both
    draw the next state via `sample`; the transition density is never needed.
    """

    arg_constraints: dict = {}
    support = constraints.real_vector
    pytree_data_fields = ("z", "u", "t_now", "t_next", "params")

    def __init__(self, z, u, t_now, t_next, params: CholeraParams, validate_args=None):
        self.z = z
        self.u = u
        self.t_now = t_now
        self.t_next = t_next
        self.params = params
        super().__init__(batch_shape=(), event_shape=(K_CLASSES + 3,), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, sample_shape + (N_SUBSTEPS,))
        flat_noise = noise.reshape((-1, N_SUBSTEPS))

        def one(n):
            return euler_maruyama_month(self.z, self.u, self.t_now, self.t_next, self.params, n)

        out = jax.vmap(one)(flat_noise)
        return out.reshape(sample_shape + (K_CLASSES + 3,))

    def log_prob(self, value):
        raise NotImplementedError("CholeraTransition is forward-sampling only.")


def observation_dist(z, u, t, tau):
    """Monthly cholera deaths y ~ N(M, (tau M)^2), M the interval death count."""
    M = z[K_CLASSES + 2]
    scale = tau * jnp.maximum(M, 1.0)
    return dist.Normal(loc=M, scale=scale)
