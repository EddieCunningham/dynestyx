"""Tests for the movement and initial-condition distributions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from champlain_dsx.distributions import CorrelatedRandomWalk, UniformOnWater
from champlain_dsx.environment import synthetic_champlain


def _env():
    return synthetic_champlain(res=400.0)


def test_uniform_on_water_lands_on_navigable_cells():
    env = _env()
    init = UniformOnWater(env.water_cell_centres(), env.res)
    keys = jr.split(jr.PRNGKey(0), 500)
    samples = jax.vmap(init._sample_one)(keys)
    nav = jax.vmap(env.is_navigable)(samples[:, 0], samples[:, 1])
    assert float(jnp.mean(nav)) > 0.99
    # Heading within (-pi, pi].
    assert float(jnp.max(jnp.abs(samples[:, 2]))) <= np.pi + 1e-5


def _crw(env, state):
    return CorrelatedRandomWalk(
        state=jnp.asarray(state),
        env=env,
        step_shape=jnp.float32(3.25),
        step_scale=jnp.float32(25.0),
        mobility=jnp.float32(216.0),
        sigma_heading=jnp.float32(0.4),
        mix_uniform_weight=jnp.float32(0.01),
        n_attempts=50,
    )


def test_transition_keeps_particles_on_water():
    env = _env()
    # Start at a known navigable centre near the middle of the lake.
    centres = np.asarray(env.water_cell_centres())
    start = centres[len(centres) // 2]
    state = jnp.array([start[0], start[1], 0.3])
    d = _crw(env, state)
    keys = jr.split(jr.PRNGKey(1), 2000)
    nxt = jax.vmap(d._sample_one)(keys)
    nav = jax.vmap(env.is_navigable)(nxt[:, 0], nxt[:, 1])
    assert float(jnp.mean(nav)) > 0.995


def test_step_length_respects_mobility_and_mean():
    env = _env()
    centres = np.asarray(env.water_cell_centres())
    start = centres[len(centres) // 2]
    state = jnp.array([start[0], start[1], 0.0])
    d = _crw(env, state)
    keys = jr.split(jr.PRNGKey(2), 4000)
    nxt = jax.vmap(d._sample_one)(keys)
    step = jnp.sqrt((nxt[:, 0] - state[0]) ** 2 + (nxt[:, 1] - state[1]) ** 2)
    # No accepted move exceeds the mobility bound.
    assert float(jnp.max(step)) <= 216.0 + 1e-3
    # Mean step length is in the right ballpark for Gamma(3.25, 25) truncated at
    # 216 (untruncated mean is 81.25). Stuck particles contribute zero steps, so
    # the empirical mean sits a little below.
    assert 40.0 < float(jnp.mean(step)) < 90.0


def test_transition_is_jittable_and_vmappable_like_the_filter():
    env = _env()
    centres = jnp.asarray(env.water_cell_centres())
    states = jnp.concatenate(
        [centres[:8], jnp.zeros((8, 1))], axis=-1
    )  # (8, 3) with heading 0

    def make_and_sample(x, key):
        return _crw(env, x)._sample_one(key)

    out = jax.jit(jax.vmap(make_and_sample))(states, jr.split(jr.PRNGKey(3), 8))
    assert out.shape == (8, 3)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_pytree_roundtrip_preserves_fields():
    env = _env()
    d = _crw(env, jnp.array([1000.0, 2000.0, 0.1]))
    leaves, treedef = jax.tree_util.tree_flatten(d)
    d2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert d2.n_attempts == 50
    np.testing.assert_allclose(np.asarray(d2.state), np.asarray(d.state))
    assert float(d2.mobility) == 216.0
