"""Quick check: does the cholera model JIT-compile and run in reasonable time?"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import optax

import dynestyx as dsx
from dynestyx import DynamicalModel, Filter
from dynestyx.inference.filters import EnKFConfig

# ---- data (abbreviated) ----
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
    607, 172, 325, 2191, 584, 58, 38, 8, 22, 50, 380, 2059,
    938, 389, 767, 1882, 286, 94, 61, 10, 106, 281, 357, 1388,
    810, 306, 381, 1308, 702, 87, 9, 14, 36, 46, 553, 1302,
    618, 147, 414, 768, 373, 39, 10, 36, 151, 1130, 3437, 4041,
    1415, 207, 92, 128, 147, 32, 7, 59, 426, 2644, 2891, 4249,
    2291, 797, 680, 1036, 404, 41, 19, 12, 10, 121, 931, 2158,
    1886, 803, 397, 613, 132, 48, 17, 22, 26, 34, 344, 657,
    117, 75, 443, 972, 646, 107, 18, 6, 9, 5, 12, 142,
    133, 189, 1715, 3115, 1412, 182, 50, 37, 77, 475, 1730, 1489,
    620, 190, 571, 1558, 440, 27, 7, 14, 93, 1462, 2467, 1703,
    1262, 458, 453, 717, 232, 26, 16, 18, 9, 78, 353, 897,
    777, 404, 799, 2067, 613, 98, 19, 26, 47, 171, 767, 1896,
    887, 325, 816, 1653, 355, 85, 54, 88, 609, 882, 1363, 2178,
    580, 396, 1493, 2154, 683, 78, 19, 10, 27, 88, 1178, 1862,
    611, 478, 2697, 3395, 520, 67, 41, 36, 209, 559, 971, 2144,
    1099, 494, 586, 508, 269, 27, 19, 21, 12, 22, 333, 676,
    487, 262, 535, 979, 170, 25, 9, 19, 13, 45, 229, 673,
    432, 107, 373, 1126, 339, 19, 11, 3, 15, 101, 539, 709,
    200, 208, 926, 1783, 831, 103, 37, 17, 33, 179, 426, 795,
    481, 491, 773, 936, 325, 101, 22, 25, 24, 88, 633, 513,
    298, 93, 687, 1750, 356, 33, 2, 18, 70, 648, 2471, 1270,
    616, 193, 706, 1372, 668, 107, 58, 21, 23, 93, 318, 867,
    332, 118, 437, 2233, 491, 27, 7, 21, 96, 360, 783, 1492,
    550, 176, 633, 922, 267, 91, 42, 4, 10, 7, 43, 377,
    563, 284, 298, 625, 131, 35, 12, 8, 9, 83, 502, 551,
    256, 198, 664, 1701, 425, 76, 17, 9, 16, 5, 141, 806,
    1603, 587, 530, 771, 511, 97, 35, 39, 156, 1097, 1233, 1418,
    1125, 420, 1592, 4169, 1535, 371, 139, 55, 85, 538, 1676, 1435,
    804, 370, 477, 394, 306, 132, 84, 87, 53, 391, 1541, 1859,
    894, 326, 853, 1891, 1009, 131, 77, 63, 66, 33, 178, 1003,
    1051, 488, 911, 1806, 837, 280, 132, 76, 381, 1328, 2639, 2164,
    1082, 326, 254, 258, 119, 106, 93, 29, 17, 17, 17, 46,
    79, 135, 1290, 2240, 561, 116, 24, 15, 33, 18, 16, 38,
    26, 45, 151, 168, 57, 32, 29, 27, 20, 106, 1522, 2013,
    434, 205, 528, 634, 195, 45, 33, 19, 20, 46, 107, 725,
    572, 183, 2199, 4018, 428, 67, 31, 8, 44, 484, 1324, 2054,
    467, 216, 673, 887, 353, 73, 46, 15, 20, 27, 25, 38,
    158, 312, 1226, 1021, 222, 90, 31, 93, 368, 657, 2208, 2178,
    702, 157, 317, 146, 63, 27, 22, 23, 28, 225, 483, 319,
    120, 59, 274, 282, 155, 31, 16, 15, 12, 14, 14, 42,
]
cholera_deaths = jnp.array(_raw, dtype=jnp.float32)
obs_times  = jnp.array([1891.0 + (i + 1) / 12.0 for i in range(600)])
obs_values = cholera_deaths[:, None]

census_years = jnp.array([1891.0, 1901.0, 1911.0, 1921.0, 1931.0, 1941.0])
census_pop   = jnp.array([2_420_656.0, 2_649_522.0, 2_960_402.0,
                           3_125_967.0, 3_432_577.0, 4_222_142.0])

_GAMMA   = 1.0 / 0.75
_M_C     = 0.046
_K       = 3
_R_STAGE = _K / 120.0
_RHO_D   = _M_C * _GAMMA
N_SUBSTEPS = 2   # dt = 0.5 months per sub-step; gamma*dt = 0.667 < 1


def seasonal_basis(t_years):
    f = 2.0 * jnp.pi * (t_years % 1.0)
    return jnp.array([1.0, jnp.cos(f), jnp.sin(f),
                       jnp.cos(2*f), jnp.sin(2*f), jnp.cos(3*f)])


def cholera_model(obs_times=None, obs_values=None, predict_times=None):
    b         = numpyro.sample("b",         dist.Normal(0.0, 1.0).expand([6]).to_event(1))
    log_omega = numpyro.sample("log_omega", dist.Normal(-6.0, 2.0))
    log_tau   = numpyro.sample("log_tau",   dist.Normal(-1.0, 1.0))
    log_eps   = numpyro.sample("log_eps",   dist.Normal(-1.0, 1.5))
    logit_s0  = numpyro.sample("logit_s0",  dist.Normal(0.0,  1.5))
    log_i0    = numpyro.sample("log_i0",    dist.Normal(jnp.log(200.0), 1.5))

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
    return dsx.sample("f", dynamics,
                      obs_times=obs_times, obs_values=obs_values,
                      predict_times=predict_times)


def filter_conditioned_model():
    with Filter(EnKFConfig()):
        return cholera_model(obs_times=obs_times, obs_values=obs_values)


print("=== Speed check: cholera SEIR model ===\n")

key = jr.PRNGKey(42)
key, k_svi = jr.split(key)

guide   = AutoDelta(filter_conditioned_model)
opt     = optax.adam(1e-2)
svi_run = SVI(filter_conditioned_model, guide, opt, loss=Trace_ELBO())

print("Running 1 SVI step (includes JIT compilation)...")
t0 = time.time()
result = svi_run.run(k_svi, num_steps=1)
t1 = time.time()
print(f"  First step (compile + run): {t1-t0:.1f}s")

print("Running 10 SVI steps (post-JIT)...")
key, k2 = jr.split(key)
t0 = time.time()
result = svi_run.run(k2, num_steps=10)
t1 = time.time()
print(f"  10 steps: {t1-t0:.1f}s  ({(t1-t0)/10*1000:.0f} ms/step)")
print(f"  Loss: {result.losses[-1]:.1f}")

print("\nEstimated time for 1000 SVI steps:", (t1-t0)/10*1000*1000/1000, "s")
