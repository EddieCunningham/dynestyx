"""Tests for animal_movement_paper.distributions."""

import jax
import jax.numpy as jnp
import pytest

from animal_movement_paper.distributions import (
    step_length_log_prob,
    step_length_sample,
    turning_angle_log_prob,
    turning_angle_sample,
    wrap_angle,
)

KEY = jax.random.PRNGKey(0)


class TestStepLength:
    def test_sample_in_range(self):
        samples = step_length_sample(KEY, sample_shape=(1000,))
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 216.0)

    def test_sample_shape_scalar(self):
        s = step_length_sample(KEY, sample_shape=())
        assert s.shape == ()

    def test_sample_shape_batch(self):
        s = step_length_sample(KEY, sample_shape=(4, 5))
        assert s.shape == (4, 5)

    def test_log_prob_shape_scalar(self):
        lp = step_length_log_prob(jnp.array(80.0))
        assert lp.shape == ()

    def test_log_prob_shape_batch(self):
        d = jnp.array([10.0, 50.0, 100.0, 200.0])
        lp = step_length_log_prob(d)
        assert lp.shape == (4,)

    def test_log_prob_neginf_beyond_mobility(self):
        lp = step_length_log_prob(jnp.array(300.0))
        assert jnp.isinf(lp) and lp < 0

    def test_log_prob_neginf_at_boundary(self):
        lp = step_length_log_prob(jnp.array(216.0))
        assert jnp.isfinite(lp)

    def test_log_prob_finite_inside_range(self):
        d = jnp.linspace(1.0, 215.0, 50)
        lp = step_length_log_prob(d)
        assert jnp.all(jnp.isfinite(lp))

    def test_log_prob_normalization(self):
        # Numerical integral over [0, 216] should integrate to ~1 (check via logsumexp).
        d_grid = jnp.linspace(0.01, 215.99, 5000)
        delta = d_grid[1] - d_grid[0]
        lp = step_length_log_prob(d_grid)
        total = jnp.sum(jnp.exp(lp)) * delta
        assert abs(total - 1.0) < 0.01


class TestTurningAngle:
    def test_sample_in_range(self):
        angles = turning_angle_sample(KEY, sample_shape=(1000,))
        assert jnp.all(angles >= -jnp.pi)
        assert jnp.all(angles <= jnp.pi)

    def test_sample_shape_scalar(self):
        a = turning_angle_sample(KEY, sample_shape=())
        assert a.shape == ()

    def test_sample_shape_batch(self):
        a = turning_angle_sample(KEY, sample_shape=(3, 4))
        assert a.shape == (3, 4)

    def test_log_prob_shape_scalar(self):
        lp = turning_angle_log_prob(jnp.array(0.0))
        assert lp.shape == ()

    def test_log_prob_shape_batch(self):
        phi = jnp.linspace(-jnp.pi, jnp.pi, 10)
        lp = turning_angle_log_prob(phi)
        assert lp.shape == (10,)

    def test_log_prob_finite_inside_range(self):
        phi = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 100)
        lp = turning_angle_log_prob(phi)
        assert jnp.all(jnp.isfinite(lp))

    def test_log_prob_normalization(self):
        phi_grid = jnp.linspace(-jnp.pi + 1e-4, jnp.pi - 1e-4, 5000)
        delta = phi_grid[1] - phi_grid[0]
        lp = turning_angle_log_prob(phi_grid)
        total = jnp.sum(jnp.exp(lp)) * delta
        assert abs(total - 1.0) < 0.02

    def test_concentrated_near_zero(self):
        # The normal component (weight 0.99) has std=0.4, so the distribution
        # should assign much more mass near 0 than near +-pi.
        lp_near_zero = turning_angle_log_prob(jnp.array(0.0))
        lp_near_pi = turning_angle_log_prob(jnp.array(jnp.pi - 0.1))
        assert lp_near_zero > lp_near_pi


class TestWrapAngle:
    def test_identity_inside_range(self):
        phi = jnp.array(1.0)
        assert jnp.allclose(wrap_angle(phi), phi)

    def test_wraps_above_pi(self):
        phi = jnp.array(jnp.pi + 0.5)
        wrapped = wrap_angle(phi)
        assert jnp.abs(wrapped) <= jnp.pi

    def test_wraps_below_neg_pi(self):
        phi = jnp.array(-jnp.pi - 0.5)
        wrapped = wrap_angle(phi)
        assert jnp.abs(wrapped) <= jnp.pi
