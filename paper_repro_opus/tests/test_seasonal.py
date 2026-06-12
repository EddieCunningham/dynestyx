"""Tests for the periodic B-spline seasonal basis."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from seasonal import periodic_bspline_basis


def test_partition_of_unity():
    t = jnp.linspace(0.0, 1.0, 137)
    B = periodic_bspline_basis(t, nbasis=6, degree=3, period=1.0)
    sums = B.sum(axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-10), sums


def test_nonnegative():
    t = jnp.linspace(0.0, 3.0, 500)
    B = periodic_bspline_basis(t, nbasis=6, degree=3, period=1.0)
    assert jnp.all(B >= -1e-12), B.min()


def test_periodicity():
    t = jnp.linspace(0.0, 1.0, 50)
    B0 = periodic_bspline_basis(t, nbasis=6, degree=3, period=1.0)
    B1 = periodic_bspline_basis(t + 1.0, nbasis=6, degree=3, period=1.0)
    B2 = periodic_bspline_basis(t + 5.0, nbasis=6, degree=3, period=1.0)
    assert jnp.allclose(B0, B1, atol=1e-10)
    assert jnp.allclose(B0, B2, atol=1e-10)


def test_shape_and_count():
    t = jnp.linspace(0.0, 1.0, 11)
    B = periodic_bspline_basis(t, nbasis=6, degree=3, period=1.0)
    assert B.shape == (11, 6)


def test_smoothness_no_jumps():
    # A cubic basis is C^2, so consecutive values should be close on a fine grid.
    t = jnp.linspace(0.0, 1.0, 2000)
    B = periodic_bspline_basis(t, nbasis=6, degree=3, period=1.0)
    jumps = jnp.abs(jnp.diff(B, axis=0)).max()
    assert jumps < 0.02, jumps


if __name__ == "__main__":
    test_partition_of_unity()
    test_nonnegative()
    test_periodicity()
    test_shape_and_count()
    test_smoothness_no_jumps()
    print("all seasonal tests passed")
