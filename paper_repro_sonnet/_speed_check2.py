"""Check steady-state SVI speed and loss with corrected I0 prior."""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import optax

import dynestyx as dsx
from dynestyx import DynamicalModel, Filter
from dynestyx.inference.filters import EnKFConfig

_raw = [
    2641, 939, 905, 1219, 368, 78, 29, 12, 30, 44, 270, 1149,
    633, 501, 855, 1271, 666, 101, 62, 23, 20, 28, 461, 892,
    751, 170, 253, 906, 700, 98, 57, 72, 471, 4217, 5168, 4747,
    2380, 852, 1166, 2122, 576, 60, 53, 62, 241, 403, 551, 739,
    862, 348, 490, 5596, 1180, 142, 41, 28, 39, 748, 3934, 3562,
    587, 311, 1639, 1903, 601, 110, 32, 19, 82, 420, 1014, 1073,
    416, 168, 909, 1355, 447, 59, 13, 21, 43, 109, 338, 470,
    489, 394, 483, 842, 356, 29, 17, 16, 57, 110, 488, 1727,
    1253, 359, 245, 549, 215, 9, 7, 31, 236, 279, 819, 1728,
    1942, 1251, 3521, 3412, 290, 46, 35, 14, 79, 852, 2951, 2656,
]  # first 120 months (10 years) for speed testing

cholera_deaths = jnp.array(_raw, dtype=jnp.float32)
T = len(cholera_deaths)
obs_times  = jnp.array([1891.0 + (i + 1) / 12.0 for i in range(T)])
obs_values = cholera_deaths[:, None]

census_years = jnp.array([1891.0, 1901.0, 1911.0, 1921.0, 1931.0, 1941.0])
census_pop   = jnp.array([2_420_656.0, 2_649_522.0, 2_960_402.0,
                           3_125_967.0, 3_432_577.0, 4_222_142.0])

_GAMMA   = 1.0 / 0.75
_M_C     = 0.046
_K       = 3
_R_STAGE = _K / 120.0
_RHO_D   = _M_C * _GAMMA
N_SUBSTEPS = 2

# I0 prior corrected: first observation has 2641 deaths → I0 ~ 2641/rho_d ≈ 43000
# Use log(10000) as center with wide prior
_LOG_I0_PRIOR_MU = jnp.log(10_000.0)


def seasonal_basis(t_years):
    f = 2.0 * jnp.pi * (t_years % 1.0)
    return jnp.array([1.0, jnp.cos(f), jnp.sin(f),
                       jnp.cos(2*f), jnp.sin(2*f), jnp.cos(3*f)])


def cholera_model(obs_times=None, obs_values=None):
    b         = numpyro.sample("b",         dist.Normal(0.0, 1.0).expand([6]).to_event(1))
    log_omega = numpyro.sample("log_omega", dist.Normal(-6.0, 2.0))
    log_tau   = numpyro.sample("log_tau",   dist.Normal(-1.0, 1.0))
    log_eps   = numpyro.sample("log_eps",   dist.Normal(-1.0, 1.5))
    logit_s0  = numpyro.sample("logit_s0",  dist.Normal(0.0,  1.5))
    log_i0    = numpyro.sample("log_i0",    dist.Normal(_LOG_I0_PRIOR_MU, 1.5))

    omega = jnp.exp(log_omega)
    tau   = jnp.exp(log_tau)
    eps   = jnp.exp(log_eps)

    P0     = 2_420_656.0
    I0     = jnp.exp(log_i0)
    S0     = jax.nn.sigmoid(logit_s0) * P0
    R0     = jnp.maximum(P0 - S0 - I0, 1e3)
    x_init = jnp.array([S0, I0, R0 / 3.0, R0 / 3.0, R0 / 3.0])
    init_cov = jnp.diag(jnp.array([1e4, 1.0, 1e4, 1e4, 1e4]))

    def state_evolution(x, u, t_now, t_next):
        beta = jnp.exp(jnp.clip(jnp.dot(b, seasonal_basis(t_now)), -8.0, 8.0))
        P_t  = jnp.interp(t_now, census_years, census_pop)
        dt   = (t_next - t_now) * 12.0 / N_SUBSTEPS

        def substep(x_curr, _):
            S, I, R1, R2, R3 = (
                jnp.maximum(x_curr[0], 0.0), jnp.maximum(x_curr[1], 0.0),
                jnp.maximum(x_curr[2], 0.0), jnp.maximum(x_curr[3], 0.0),
                jnp.maximum(x_curr[4], 0.0),
            )
            foi = beta * I / P_t + omega
            dS  = (-foi * S + _R_STAGE * R3) * dt
            dI  = (foi * S - _GAMMA * I) * dt
            dR1 = ((1 - _M_C) * _GAMMA * I - _R_STAGE * R1) * dt
            dR2 = (_R_STAGE * R1 - _R_STAGE * R2) * dt
            dR3 = (_R_STAGE * R2 - _R_STAGE * R3) * dt
            return jnp.array([S+dS, I+dI, R1+dR1, R2+dR2, R3+dR3]), None

        x_mean, _ = jax.lax.scan(substep, x, None, length=N_SUBSTEPS)
        x_mean = jnp.maximum(x_mean, 0.0)

        S0c  = jnp.maximum(x[0], 0.0)
        I0c  = jnp.maximum(x[1], 0.0)
        foi0 = beta * I0c / P_t + omega
        q_si = eps**2 * foi0 * S0c
        reg  = 1.0
        Q = jnp.diag(jnp.array([q_si + reg, q_si + reg, reg, reg, reg]))
        Q = Q.at[0, 1].set(-q_si)
        Q = Q.at[1, 0].set(-q_si)
        return dist.MultivariateNormal(x_mean, Q)

    def observation_fn(x, u, t):
        I   = jnp.maximum(x[1], 0.0)
        mu  = _RHO_D * I
        sig = tau * jnp.maximum(mu, 1.0)
        return dist.Normal(mu, sig)

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(x_init, init_cov),
        state_evolution=state_evolution,
        observation_model=observation_fn,
    )
    return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


def filter_conditioned_model():
    with Filter(EnKFConfig()):
        return cholera_model(obs_times=obs_times, obs_values=obs_values)


print(f"=== Speed check v2: T={T} months ===\n")

key = jr.PRNGKey(42)
key, k1, k2, k3 = jr.split(key, 4)

guide   = AutoDelta(filter_conditioned_model)
opt     = optax.adam(1e-2)
svi_run = SVI(filter_conditioned_model, guide, opt, loss=Trace_ELBO())

print("Compiling (1 step)...")
t0 = time.time()
result = svi_run.run(k1, num_steps=1, progress_bar=False)
t1 = time.time()
print(f"  Compile+run: {t1-t0:.1f}s, loss={result.losses[-1]:.0f}")

print("Running 50 steps (steady-state speed)...")
t0 = time.time()
result = svi_run.run(k2, num_steps=50, progress_bar=False)
t1 = time.time()
ms_per_step = (t1-t0)/50*1000
print(f"  50 steps: {t1-t0:.1f}s ({ms_per_step:.0f} ms/step)")
print(f"  Final loss: {result.losses[-1]:.0f}")

print(f"\nWith T={T}, estimated time for:")
print(f"  300 SVI steps: {300*ms_per_step/1000:.0f}s")
print(f"  Full T=600 would be ~2x slower: {600*ms_per_step/1000:.0f}s per 300 steps")
