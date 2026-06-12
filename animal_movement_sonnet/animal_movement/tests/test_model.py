"""Tests for the acoustic telemetry CRW model."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Smoother
from dynestyx.inference.smoother_configs import PFSmootherConfig

from animal_movement.domain import Domain, make_synthetic_lake
from animal_movement.model import (
    ALPHA_DET,
    BETA_DET,
    GAMMA_DET,
    CRWStep,
    _detection_prob,
    animal_movement_model,
)


@pytest.fixture
def lake():
    return make_synthetic_lake()


# ── CRWStep ─────────────────────────────────────────────────────────────────


def test_crwstep_sample_shape():
    prev = jnp.array([5000.0, 10000.0, 0.5])
    d = CRWStep(prev, k=3.25, theta=25.0, mobility=216.0, sigma_phi=0.4)
    key = jr.PRNGKey(0)
    s = d.sample(key)
    assert s.shape == (3,)


def test_crwstep_sample_batch_shape():
    prev = jnp.array([5000.0, 10000.0, 0.5])
    d = CRWStep(prev, k=3.25, theta=25.0, mobility=216.0, sigma_phi=0.4)
    key = jr.PRNGKey(0)
    s = d.sample(key, sample_shape=(20,))
    assert s.shape == (20, 3)


def test_crwstep_step_length_within_bounds():
    """Sampled step lengths must stay below mobility."""
    prev = jnp.array([5000.0, 10000.0, 0.0])
    d = CRWStep(prev, k=3.25, theta=25.0, mobility=216.0, sigma_phi=0.4)
    key = jr.PRNGKey(1)
    samples = d.sample(key, sample_shape=(1000,))
    new_pos = samples[:, :2]
    distances = jnp.linalg.norm(new_pos - prev[:2], axis=-1)
    assert jnp.all(distances <= 216.0 + 1e-4)


def test_crwstep_log_prob_raises():
    prev = jnp.array([0.0, 0.0, 0.0])
    d = CRWStep(prev, k=3.25, theta=25.0, mobility=216.0, sigma_phi=0.4)
    with pytest.raises(NotImplementedError):
        d.log_prob(jnp.zeros(3))


# ── Detection probability ────────────────────────────────────────────────────


def test_detection_prob_at_zero_distance():
    """At distance 0, p = sigmoid(alpha) which should be close to 1."""
    receiver_locs = jnp.array([[0.0, 0.0]])
    pos = jnp.array([0.0, 0.0])
    p = _detection_prob(pos, receiver_locs)
    expected = jnp.array([1.0 / (1.0 + jnp.exp(-ALPHA_DET))])
    assert jnp.allclose(p, expected, atol=1e-5)


def test_detection_prob_zero_beyond_range():
    """Detection probability must be exactly 0 beyond gamma."""
    receiver_locs = jnp.array([[0.0, 0.0]])
    pos = jnp.array([GAMMA_DET + 1.0, 0.0])
    p = _detection_prob(pos, receiver_locs)
    assert p[0] == 0.0


def test_detection_prob_shape(lake):
    pos = jnp.array([lake.width / 2, lake.height / 2])
    p = _detection_prob(pos, lake.receiver_locs)
    assert p.shape == (lake.n_receivers,)
    assert jnp.all((p >= 0.0) & (p <= 1.0))


def test_detection_prob_decreasing_with_distance():
    """p should decrease monotonically with distance up to gamma."""
    receiver_locs = jnp.array([[0.0, 0.0]])
    distances = jnp.linspace(0.0, GAMMA_DET - 1.0, 50)
    positions = jnp.stack([distances, jnp.zeros(50)], axis=-1)
    probs = jax_vmap_detection(positions, receiver_locs)
    diffs = jnp.diff(probs[:, 0])
    assert jnp.all(diffs <= 0.0)


def jax_vmap_detection(positions, receiver_locs):
    import jax
    return jax.vmap(lambda p: _detection_prob(p, receiver_locs))(positions)


# ── Domain ───────────────────────────────────────────────────────────────────


def test_domain_receiver_count(lake):
    assert lake.n_receivers == 15  # 3 cols x 5 rows


def test_domain_receivers_inside_lake(lake):
    locs = lake.receiver_locs
    assert jnp.all(locs[:, 0] >= 0) and jnp.all(locs[:, 0] <= lake.width)
    assert jnp.all(locs[:, 1] >= 0) and jnp.all(locs[:, 1] <= lake.height)


def test_domain_region_count(lake):
    assert lake.region_boundaries.shape == (4,)  # 5 regions -> 4 internal boundaries


# ── Simulator smoke test ─────────────────────────────────────────────────────


def test_simulate_produces_correct_shapes(lake):
    """Forward simulation must produce states of shape (T, 3) and observations (T, K)."""
    T = 20
    obs_times = jnp.arange(T, dtype=float)

    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            animal_movement_model,
            params={},
            num_samples=1,
            exclude_deterministic=False,
        )
        out = pred(jr.PRNGKey(0), domain=lake, predict_times=obs_times)

    states = out["f_states"]      # (num_samples, n_sim, T, 3)
    obs = out["f_observations"]   # (num_samples, n_sim, T, K)
    assert states.shape == (1, 1, T, 3)
    assert obs.shape == (1, 1, T, lake.n_receivers)
    assert jnp.all(jnp.isfinite(states))


def test_simulate_observations_are_binary(lake):
    T = 10
    obs_times = jnp.arange(T, dtype=float)
    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            animal_movement_model,
            params={},
            num_samples=1,
            exclude_deterministic=False,
        )
        out = pred(jr.PRNGKey(1), domain=lake, predict_times=obs_times)
    obs = out["f_observations"][0, 0]
    assert jnp.all((obs == 0) | (obs == 1))


# ── PF smoother smoke test ───────────────────────────────────────────────────


def test_pf_smoother_runs_and_returns_particles(lake):
    """Particle smoother must run without error and return finite particles."""
    T = 15
    obs_times = jnp.arange(T, dtype=float)

    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            animal_movement_model,
            params={},
            num_samples=1,
            exclude_deterministic=False,
        )
        out = pred(jr.PRNGKey(2), domain=lake, predict_times=obs_times)

    obs_values = out["f_observations"][0, 0]

    smoother_config = PFSmootherConfig(
        n_particles=100,
        pf_backward_sampling_method="tracing",
        pf_n_smoother_particles=50,
        record_smoothed_particles=True,
        record_smoothed_states_mean=True,
    )

    with DiscreteTimeSimulator(n_simulations=1):
        with Smoother(smoother_config=smoother_config):
            smoother_pred = Predictive(
                animal_movement_model,
                params={},
                num_samples=1,
                exclude_deterministic=False,
            )
            smoother_out = smoother_pred(
                jr.PRNGKey(3),
                domain=lake,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    particles = smoother_out["f_smoothed_particles"]
    # shape: (num_samples, T, n_smoother_particles, state_dim)
    assert particles.shape[1] == T
    assert particles.shape[2] == 50
    assert particles.shape[3] == 3
    assert jnp.all(jnp.isfinite(particles))
