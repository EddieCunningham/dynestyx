"""Acoustic observation model.

Each receiver reports a detection or non-detection at every time step. The
detection probability declines logistically with the transmitter-to-receiver
distance, drops to zero beyond a maximum range, and drops to zero when the
straight-line path is blocked by land.

    p(s_t, r_k) = logistic(alpha + beta * |s_t - r_k|)   if |s_t - r_k| < gamma
                                                          and line of sight holds
                = 0                                       otherwise

with beta negative, so detection probability falls with distance. Detections at
distinct receivers are conditionally independent given the state.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist


def detection_probability(state, receivers, env, alpha, beta, gamma):
    """Per-receiver detection probability for a single state.

    Args:
        state: State (x, y, phi), shape (3,).
        receivers: Receiver coordinates, shape (K, 2).
        env: LakeEnvironment providing the line-of-sight test.
        alpha: Logistic intercept.
        beta: Logistic distance slope (negative).
        gamma: Maximum detection range in metres.

    Returns:
        Array of shape (K,) with detection probabilities in [0, 1].
    """
    px, py = state[0], state[1]
    dx = receivers[:, 0] - px
    dy = receivers[:, 1] - py
    distance = jnp.sqrt(dx**2 + dy**2)

    midpoint_x = px + 0.5 * dx
    midpoint_y = py + 0.5 * dy
    has_los = env.is_water(midpoint_x, midpoint_y) > 0.5
    in_range = distance < gamma

    logistic = 1.0 / (1.0 + jnp.exp(-(alpha + beta * distance)))
    return jnp.where(in_range & has_los, logistic, 0.0)


def make_observation_model(env, receivers, alpha, beta, gamma):
    """Return a dynestyx observation callable mapping (x, u, t) to a distribution.

    The returned distribution is a product of independent Bernoulli detections,
    one per receiver. Receivers with zero detection probability contribute a
    near -inf log-density to any detection and a zero log-density to any
    non-detection, which lets the filter rule out impossible particles without
    producing NaNs.
    """
    receivers = jnp.asarray(receivers)

    def observation_model(x, u, t):
        probs = detection_probability(x, receivers, env, alpha, beta, gamma)
        return dist.Independent(dist.Bernoulli(probs=probs), 1)

    return observation_model
