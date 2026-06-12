"""Tests for the acoustic observation model."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from champlain_dsx.environment import environment_from_arrays
from champlain_dsx.observation import detection_probability, make_observation_model

ALPHA, BETA, GAMMA = 0.9039, -0.0021, 7000.0


def _open_env():
    # A large all-water grid so line of sight always holds.
    mask = np.ones((50, 50), dtype=np.float32)
    depth = np.full((50, 50), 30.0, dtype=np.float32)
    return environment_from_arrays(mask, mask, depth, x_min=0.0, y_max=50000.0, res=1000.0)


def test_probability_at_zero_distance_matches_logistic_intercept():
    env = _open_env()
    state = jnp.array([25000.0, 25000.0, 0.0])
    receivers = jnp.array([[25000.0, 25000.0]])
    p = detection_probability(state, receivers, env, ALPHA, BETA, GAMMA)
    expected = 1.0 / (1.0 + np.exp(-ALPHA))
    np.testing.assert_allclose(float(p[0]), expected, rtol=1e-5)


def test_probability_decays_with_distance_and_cuts_off_beyond_range():
    env = _open_env()
    state = jnp.array([0.0, 25000.0, 0.0])
    receivers = jnp.array([[3000.0, 25000.0], [8000.0, 25000.0]])
    p = detection_probability(state, receivers, env, ALPHA, BETA, GAMMA)
    # Within range, detection probability falls with distance.
    p_near = 1.0 / (1.0 + np.exp(-(ALPHA + BETA * 3000.0)))
    np.testing.assert_allclose(float(p[0]), p_near, rtol=1e-5)
    # Beyond gamma (8000 > 7000) the probability is exactly zero.
    assert float(p[1]) == 0.0


def test_line_of_sight_blocks_detection():
    # Water on the left and right halves, a land strip down the middle column.
    mask = np.ones((20, 20), dtype=np.float32)
    mask[:, 10] = 0.0
    depth = np.where(mask > 0, 30.0, np.nan).astype(np.float32)
    env = environment_from_arrays(mask, mask, depth, x_min=0.0, y_max=20000.0, res=1000.0)
    # State left of the strip, receiver right of it; midpoint falls on land.
    state = jnp.array([5000.0, 10000.0, 0.0])
    receivers = jnp.array([[15500.0, 10000.0]])
    p = detection_probability(state, receivers, env, ALPHA, BETA, GAMMA)
    assert float(p[0]) == 0.0


def test_observation_logprob_is_finite_and_penalises_impossible_detections():
    env = _open_env()
    receivers = jnp.array([[25000.0, 25000.0], [0.0, 0.0]])
    model = make_observation_model(env, receivers, ALPHA, BETA, GAMMA)
    state = jnp.array([25000.0, 25000.0, 0.0])
    d = model(state, None, 0.0)
    # Detection at the co-located receiver, non-detection at the far one.
    lp_consistent = float(d.log_prob(jnp.array([1.0, 0.0])))
    assert np.isfinite(lp_consistent)
    # A detection at the far (zero-probability) receiver is strongly penalised.
    lp_impossible = float(d.log_prob(jnp.array([1.0, 1.0])))
    assert lp_impossible < lp_consistent - 50.0


def test_observation_model_is_vmappable():
    env = _open_env()
    receivers = jnp.array([[25000.0, 25000.0], [10000.0, 10000.0]])
    model = make_observation_model(env, receivers, ALPHA, BETA, GAMMA)
    states = jr.uniform(jr.PRNGKey(0), (16, 3), minval=0.0, maxval=50000.0)
    lp = jax.vmap(lambda s: model(s, None, 0.0).log_prob(jnp.array([1.0, 0.0])))(states)
    assert lp.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(lp)))
