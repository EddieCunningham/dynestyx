"""Building-block distributions for the animal movement model.

These are helper functions for the two stochastic components of the
correlated random walk: the step-length distribution (truncated Gamma) and
the turning-angle distribution (mixture of a truncated Normal and a Uniform).
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp


def step_length_dist(
    concentration: float = 3.25,
    scale: float = 25.0,
    mobility: float = 216.0,
) -> dist.Gamma:
    """Return the base Gamma distribution for step lengths.

    The step length is drawn from Gamma(concentration, rate=1/scale) truncated
    to [0, mobility] meters.  Because the CDF at `mobility` is very close to 1
    for the default parameters (roughly 0.988), the distribution object returned
    here is the plain (untruncated) Gamma; use `step_length_log_prob` and
    `step_length_sample` for the correctly truncated density and sampler.
    """
    return dist.Gamma(concentration=concentration, rate=1.0 / scale)


def step_length_log_prob(
    d: jax.Array,
    concentration: float = 3.25,
    scale: float = 25.0,
    mobility: float = 216.0,
) -> jax.Array:
    """Log-probability of the truncated Gamma step-length distribution.

    Evaluates log p(d) for Gamma(concentration, rate=1/scale) restricted to
    [0, mobility].  Returns -inf for d outside [0, mobility].
    """
    rate = 1.0 / scale
    base_lp = dist.Gamma(concentration=concentration, rate=rate).log_prob(d)
    # CDF(mobility) = P(concentration, mobility * rate)
    log_cdf_high = jnp.log(jax.scipy.special.gammainc(concentration, mobility * rate))
    in_range = (d >= 0.0) & (d <= mobility)
    return jnp.where(in_range, base_lp - log_cdf_high, -jnp.inf)


def step_length_sample(
    key: jax.Array,
    sample_shape: tuple[int, ...] = (),
    concentration: float = 3.25,
    scale: float = 25.0,
    mobility: float = 216.0,
) -> jax.Array:
    """Sample from Gamma(concentration, rate=1/scale) truncated to [0, mobility].

    Uses the inverse-CDF method: u ~ Uniform(0, CDF(mobility)), then
    d = icdf(u) via tfp.math.igammainv.
    """
    rate = 1.0 / scale
    cdf_high = jax.scipy.special.gammainc(concentration, mobility * rate)
    u = jax.random.uniform(key, shape=sample_shape, minval=0.0, maxval=cdf_high)
    # igammainv(a, p) solves gammainc(a, x) = p for x; then d = x / rate
    d = tfp.math.igammainv(concentration, u) / rate
    return d


def turning_angle_log_prob(
    delta_phi: jax.Array,
    std: float = 0.4,
    normal_weight: float = 0.99,
) -> jax.Array:
    """Log-probability of the turning-angle mixture distribution.

    The distribution is:
        normal_weight * TruncatedNormal(0, std, [-pi, pi])
        + (1 - normal_weight) * Uniform(-pi, pi)

    Valid for delta_phi in [-pi, pi].
    """
    log_p_normal = dist.TruncatedNormal(
        loc=0.0, scale=std, low=-jnp.pi, high=jnp.pi
    ).log_prob(delta_phi)
    log_p_uniform = dist.Uniform(low=-jnp.pi, high=jnp.pi).log_prob(delta_phi)
    return jnp.logaddexp(
        jnp.log(normal_weight) + log_p_normal,
        jnp.log(1.0 - normal_weight) + log_p_uniform,
    )


def turning_angle_sample(
    key: jax.Array,
    sample_shape: tuple[int, ...] = (),
    std: float = 0.4,
    normal_weight: float = 0.99,
) -> jax.Array:
    """Sample from the turning-angle mixture distribution.

    Draws a Bernoulli to select the component (TruncatedNormal vs Uniform)
    and returns a sample in [-pi, pi].
    """
    k1, k2, k3 = jax.random.split(key, 3)
    use_normal = jax.random.bernoulli(k1, p=normal_weight, shape=sample_shape)
    normal_sample = dist.TruncatedNormal(
        loc=0.0, scale=std, low=-jnp.pi, high=jnp.pi
    ).sample(k2, sample_shape)
    uniform_sample = dist.Uniform(low=-jnp.pi, high=jnp.pi).sample(k3, sample_shape)
    return jnp.where(use_normal, normal_sample, uniform_sample)


def wrap_angle(phi: jax.Array) -> jax.Array:
    """Wrap angle to the interval [-pi, pi]."""
    return (phi + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
