"""Correlated random walk state-space model for acoustic telemetry.

State: s_t = (x_t, y_t, phi_t)  -- 2D position (m) + heading (rad)
Step length: d_t ~ Gamma_{[0, mobility]}(k, theta)  (m per 2-min step)
Turning angle: Delta_phi_t ~ 0.99 * Normal_{[-pi,pi]}(0, sigma^2)
                             + 0.01 * Uniform(-pi, pi)
Detection: y_{t,k} | s_t ~ Bernoulli(p(s_t, r_k))
  p(s_t, r_k) = sigmoid(alpha + beta * |s_t - r_k|)  if |s_t - r_k| < gamma
               0                                       otherwise

Parameters are fixed at the best-model values from Lavender et al. 2026.

The bootstrap PF with tracing smoother only calls .sample() on the state
evolution distribution, never .log_prob(). CRWStep exploits this: it
implements sample() correctly and raises NotImplementedError for log_prob().
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array

import dynestyx as dsx
from dynestyx import DynamicalModel
from dynestyx.inference.smoother_configs import PFSmootherConfig

from animal_movement.domain import Domain

# Best-model parameters (Lavender et al. 2026, Table S4)
K_STEP = 3.25        # Gamma shape
THETA_STEP = 25.0    # Gamma scale (m)
MOBILITY = 216.0     # Max step length (m per 2-min step)
SIGMA_PHI = 0.4      # Turning angle std (rad)
ALPHA_DET = 0.9039   # Detection logistic intercept
BETA_DET = -0.0021   # Detection logistic slope (per m)
GAMMA_DET = 7000.0   # Max detection range (m)


def _sample_truncated_gamma(key, k, theta, high, sample_shape=()):
    """Sample Gamma(k, scale=theta) conditioned on the result being below high.

    Uses rejection sampling via lax.while_loop, which is fully JIT-compatible
    and works inside lax.scan. For Gamma(3.25, 25) with high=216 the acceptance
    rate exceeds 99.9 percent so the loop almost never iterates more than once.
    """

    def one_draw(key):
        def cond(state):
            _, x = state
            return x >= high

        def body(state):
            key, _ = state
            key, sub = jax.random.split(key)
            x = jax.random.gamma(sub, k) * theta
            return key, x

        key, sub = jax.random.split(key)
        x0 = jax.random.gamma(sub, k) * theta
        _, x = jax.lax.while_loop(cond, body, (key, x0))
        return x

    if not sample_shape:
        return one_draw(key)

    n = 1
    for s in sample_shape:
        n *= s
    keys = jax.random.split(key, n)
    samples = jax.vmap(one_draw)(keys)
    return samples.reshape(sample_shape)


class CRWStep(dist.Distribution):
    """Correlated random walk one-step transition distribution.

    Given the current state (x, y, phi), samples the next state
    (x', y', phi') via:
      d ~ TruncatedGamma(k, theta, high=mobility)
      Delta_phi ~ 0.99 * TruncatedNormal(0, sigma^2, [-pi, pi])
                + 0.01 * Uniform(-pi, pi)
      phi' = phi + Delta_phi
      x'   = x + d * cos(phi')
      y'   = y + d * sin(phi')

    log_prob is not implemented. This is intentional: the bootstrap PF
    with tracing smoother never evaluates the transition density, only
    samples from it. If log_prob is called, a clear error is raised.
    """

    arg_constraints = {}
    support = dist.constraints.real_vector
    has_rsample = False

    def __init__(self, prev_state, k, theta, mobility, sigma_phi):
        self.prev_state = prev_state
        self.k = k
        self.theta = theta
        self.mobility = mobility
        self.sigma_phi = sigma_phi
        super().__init__(batch_shape=(), event_shape=(3,))

    def sample(self, key, sample_shape=()):
        key_d, key_mix, key_normal, key_uniform = jax.random.split(key, 4)

        x_prev = self.prev_state[..., 0]
        y_prev = self.prev_state[..., 1]
        phi_prev = self.prev_state[..., 2]

        d = _sample_truncated_gamma(key_d, self.k, self.theta, self.mobility, sample_shape)

        use_normal = jax.random.bernoulli(key_mix, p=0.99, shape=sample_shape)
        angle_normal = dist.TruncatedNormal(
            0.0, self.sigma_phi, low=-jnp.pi, high=jnp.pi
        ).sample(key_normal, sample_shape)
        angle_uniform = dist.Uniform(-jnp.pi, jnp.pi).sample(key_uniform, sample_shape)
        delta_phi = jnp.where(use_normal, angle_normal, angle_uniform)

        phi_new = phi_prev + delta_phi
        x_new = x_prev + d * jnp.cos(phi_new)
        y_new = y_prev + d * jnp.sin(phi_new)

        return jnp.stack([x_new, y_new, phi_new], axis=-1)

    def log_prob(self, value):
        raise NotImplementedError(
            "CRWStep.log_prob is not implemented. "
            "Use PFSmootherConfig(pf_backward_sampling_method='tracing') "
            "which does not require the transition log density."
        )


def _detection_prob(pos: Array, receiver_locs: Array) -> Array:
    """Logistic detection probability for each receiver.

    Args:
        pos: (2,) animal position.
        receiver_locs: (K, 2) receiver positions.

    Returns:
        (K,) detection probabilities in [0, 1].
    """
    diffs = receiver_locs - pos[None, :]
    distances = jnp.linalg.norm(diffs, axis=-1)
    logit_p = ALPHA_DET + BETA_DET * distances
    p = jax.nn.sigmoid(logit_p)
    in_range = distances < GAMMA_DET
    return jnp.where(in_range, p, 0.0)


def animal_movement_model(
    domain: Domain,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    """Dynestyx model for acoustic telemetry animal tracking.

    The state is (x, y, phi): 2D position in metres + heading in radians.
    Observations are binary detection vectors of length K (one entry per
    receiver) at each time step.

    All model parameters are fixed at the best-model values from
    Lavender et al. 2026; only the latent states are inferred.

    Args:
        domain: Lake domain with receiver locations.
        obs_times: (T,) observation time indices.
        obs_values: (T, K) binary detection matrix.
        predict_times: (T,) times at which to generate predictions.
    """
    receiver_locs = domain.receiver_locs

    def state_evolution(x, u, t_now, t_next):
        return CRWStep(
            prev_state=x,
            k=K_STEP,
            theta=THETA_STEP,
            mobility=MOBILITY,
            sigma_phi=SIGMA_PHI,
        )

    def observation_model(x, u, t):
        pos = x[:2]
        p = _detection_prob(pos, receiver_locs)
        return dist.Independent(dist.Bernoulli(probs=p), 1)

    initial_condition = dist.MultivariateNormal(
        loc=jnp.array(
            [domain.width / 2.0, domain.height / 2.0, 0.0]
        ),
        covariance_matrix=jnp.diag(
            jnp.array(
                [
                    (domain.width / 4.0) ** 2,
                    (domain.height / 4.0) ** 2,
                    jnp.pi**2 / 3.0,
                ]
            )
        ),
    )

    dynamics = DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )
