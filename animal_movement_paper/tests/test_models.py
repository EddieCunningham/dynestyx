"""Tests for animal_movement_paper.models."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

from dynestyx.inference.filter_configs import PFConfig
from dynestyx.inference.filters import Filter
import dynestyx as dsx

from animal_movement_paper.models import (
    CorrelatedRandomWalkDistribution,
    CorrelatedRandomWalkTransition,
    TelemetryObservationModel,
    animal_movement_model,
)

KEY = jax.random.PRNGKey(1)
STATE = jnp.array([5000.0, 5000.0, 0.5])
RECEIVERS = jnp.array([
    [3000.0, 3000.0],
    [7000.0, 3000.0],
    [5000.0, 7000.0],
])


class TestCorrelatedRandomWalkDistribution:
    def test_sample_shape(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        s = crw.sample(KEY)
        assert s.shape == (3,)

    def test_sample_batch_shape(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        s = crw.sample(KEY, sample_shape=(10,))
        assert s.shape == (10, 3)

    def test_log_prob_shape_scalar(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        s = crw.sample(KEY)
        lp = crw.log_prob(s)
        assert lp.shape == ()

    def test_log_prob_finite_for_valid_sample(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        s = crw.sample(KEY)
        lp = crw.log_prob(s)
        assert jnp.isfinite(lp)

    def test_log_prob_consistent_across_samples(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        keys = jax.random.split(KEY, 20)
        samples = jax.vmap(lambda k: crw.sample(k))(keys)
        log_probs = jax.vmap(lambda s: crw.log_prob(s))(samples)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_event_shape(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        assert crw.event_shape == (3,)

    def test_batch_shape_scalar_state(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        assert crw.batch_shape == ()

    def test_mobility_constraint_respected(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        keys = jax.random.split(KEY, 500)
        samples = jax.vmap(lambda k: crw.sample(k))(keys)
        displacements = jnp.sqrt(jnp.sum((samples[:, :2] - STATE[:2]) ** 2, axis=-1))
        assert jnp.all(displacements <= 216.0 + 1e-3)

    def test_log_prob_neginf_beyond_mobility(self):
        crw = CorrelatedRandomWalkDistribution(state=STATE)
        # Construct a state far beyond mobility
        far_state = STATE + jnp.array([5000.0, 5000.0, 0.0])
        lp = crw.log_prob(far_state)
        assert jnp.isinf(lp) and lp < 0


class TestCorrelatedRandomWalkTransition:
    def test_returns_distribution(self):
        transition = CorrelatedRandomWalkTransition()
        d = transition(STATE, None, 0.0, 1.0)
        assert isinstance(d, dist.Distribution)

    def test_returned_distribution_is_crw(self):
        transition = CorrelatedRandomWalkTransition()
        d = transition(STATE, None, 0.0, 1.0)
        assert isinstance(d, CorrelatedRandomWalkDistribution)

    def test_sample_has_correct_shape(self):
        transition = CorrelatedRandomWalkTransition()
        d = transition(STATE, None, 0.0, 1.0)
        s = d.sample(KEY)
        assert s.shape == (3,)

    def test_parameters_passed_through(self):
        transition = CorrelatedRandomWalkTransition(mobility=100.0)
        d = transition(STATE, None, 0.0, 1.0)
        assert d.mobility == 100.0


class TestTelemetryObservationModel:
    def test_returns_distribution(self):
        obs = TelemetryObservationModel(receiver_locations=RECEIVERS)
        d = obs(STATE, None, 0.0)
        assert isinstance(d, dist.Distribution)

    def test_log_prob_shape(self):
        obs = TelemetryObservationModel(receiver_locations=RECEIVERS)
        d = obs(STATE, None, 0.0)
        y = jnp.array([1.0, 0.0, 1.0])
        lp = d.log_prob(y)
        assert lp.shape == ()

    def test_log_prob_finite(self):
        obs = TelemetryObservationModel(receiver_locations=RECEIVERS)
        d = obs(STATE, None, 0.0)
        y = jnp.array([1.0, 0.0, 1.0])
        lp = d.log_prob(y)
        assert jnp.isfinite(lp)

    def test_detection_probability_decreases_with_distance(self):
        # Single receiver at origin; closer animal should have higher prob.
        receiver = jnp.array([[0.0, 0.0]])
        obs = TelemetryObservationModel(receiver_locations=receiver)
        close_state = jnp.array([500.0, 0.0, 0.0])
        far_state = jnp.array([3000.0, 0.0, 0.0])
        d_close = obs(close_state, None, 0.0)
        d_far = obs(far_state, None, 0.0)
        # Compare log_prob of detection (y=1) for close vs far
        lp_close = d_close.log_prob(jnp.array([1.0]))
        lp_far = d_far.log_prob(jnp.array([1.0]))
        assert lp_close > lp_far

    def test_zero_prob_beyond_max_range(self):
        receiver = jnp.array([[0.0, 0.0]])
        obs = TelemetryObservationModel(receiver_locations=receiver, max_range=1000.0)
        far_state = jnp.array([5000.0, 0.0, 0.0])
        d = obs(far_state, None, 0.0)
        # Detection probability should be 0, so log_prob of y=1 should be very small
        lp = d.log_prob(jnp.array([1.0]))
        assert lp < -50.0

    def test_no_detection_possible_for_all_zero_obs(self):
        obs = TelemetryObservationModel(receiver_locations=RECEIVERS)
        d = obs(STATE, None, 0.0)
        y_all_zeros = jnp.zeros(3)
        lp = d.log_prob(y_all_zeros)
        assert jnp.isfinite(lp)

    def test_observation_dim_matches_num_receivers(self):
        obs = TelemetryObservationModel(receiver_locations=RECEIVERS)
        d = obs(STATE, None, 0.0)
        assert d.event_shape == (3,)


class TestAnimalMovementModel:
    def test_model_builds(self):
        model = animal_movement_model(RECEIVERS)
        assert model is not None

    def test_state_dim(self):
        model = animal_movement_model(RECEIVERS)
        assert model.state_dim == 3

    def test_observation_dim(self):
        model = animal_movement_model(RECEIVERS)
        assert model.observation_dim == 3

    def test_particle_filter_runs(self):
        """Run a minimal particle filter on synthetic data."""
        n_steps = 10
        n_receivers = 3
        obs_times = jnp.arange(n_steps, dtype=float)
        # All-zero observations (no detections)
        obs_values = jnp.zeros((n_steps, n_receivers))

        pf_config = PFConfig(
            n_particles=50,
            record_filtered_particles=True,
            record_filtered_log_weights=True,
        )

        def model(obs_times=None, obs_values=None):
            dynamics = animal_movement_model(RECEIVERS)
            return dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

        with Filter(filter_config=pf_config):
            trace = numpyro.infer.Predictive(model, num_samples=1)(
                KEY, obs_times=obs_times, obs_values=obs_values
            )

        # Confirm filtered particles were recorded and have expected shape
        assert "f_filtered_particles" in trace
        particles = trace["f_filtered_particles"]
        # Shape: (1, T, N, state_dim) with 1 sample from Predictive
        assert particles.shape[-1] == 3
        assert particles.shape[-2] == 50
