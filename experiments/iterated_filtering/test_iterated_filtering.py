"""Tests for the IF1 iterated-filtering experiment.

The suite has three layers. Deterministic unit tests pin down the cooling schedule, the
parameter bijector, the weighted moment computation, and the parameter update formula against
hand computations. A linear-Gaussian test checks that IF1 reaches the exact maximum-likelihood
estimate, where that estimate is found by maximizing the Kalman-filter marginal likelihood on a
grid. Two nonlinear tests check parameter recovery on a stochastic volatility model and a
discrete Lorenz 63 model.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import compute_cuthbert_filter

from experiments.iterated_filtering import models as M
from experiments.iterated_filtering.iterated_filtering import (
    BoxTransform,
    IF1Config,
    IF1Inference,
    _cooling,
    _eq21_update,
    _if1_moments,
    run_iterated_filtering,
)


# ---------------------------------------------------------------------------
# Deterministic unit tests
# ---------------------------------------------------------------------------


def test_cooling_schedule():
    rw = jnp.array([0.5, 0.2])
    alpha = 0.9
    for m in [0, 1, 5, 17]:
        got = _cooling(rw, alpha, jnp.asarray(m))
        np.testing.assert_allclose(np.asarray(got), np.asarray(rw) * alpha**m, rtol=1e-12)


def test_box_transform_roundtrip_and_bounds():
    t = BoxTransform(lower=jnp.array([-0.7, 0.0]), upper=jnp.array([0.7, 1.0]))
    c = jnp.array([0.3, 0.85])
    u = t.inverse(c)
    np.testing.assert_allclose(np.asarray(t.forward(u)), np.asarray(c), atol=1e-10)
    # Forward keeps every value strictly inside the box for finite inputs, and maps the
    # saturating limits onto the closed boundary.
    moderate = jnp.array([[-8.0, -8.0], [8.0, 8.0]])
    fwd = jax.vmap(t.forward)(moderate)
    assert np.all(np.asarray(fwd)[:, 0] > -0.7) and np.all(np.asarray(fwd)[:, 0] < 0.7)
    assert np.all(np.asarray(fwd)[:, 1] > 0.0) and np.all(np.asarray(fwd)[:, 1] < 1.0)
    sat = jax.vmap(t.forward)(jnp.array([[-jnp.inf, -jnp.inf], [jnp.inf, jnp.inf]]))
    np.testing.assert_allclose(np.asarray(sat), np.array([[-0.7, 0.0], [0.7, 1.0]]))


def test_box_transform_identity_component():
    inf = jnp.array([jnp.inf])
    t = BoxTransform(lower=-inf, upper=inf)
    u = jnp.array([3.14])
    np.testing.assert_allclose(np.asarray(t.forward(u)), np.asarray(u), atol=1e-12)
    np.testing.assert_allclose(np.asarray(t.inverse(u)), np.asarray(u), atol=1e-12)


def test_if1_moments_match_numpy():
    key = jr.PRNGKey(0)
    theta = jr.normal(key, (500, 2))
    log_w = jr.normal(jr.PRNGKey(1), (500,))
    wnorm = jax.nn.softmax(log_w)
    theta_hat_F, V_P = _if1_moments(wnorm, theta)
    # Filter mean is the importance-weighted mean.
    np.testing.assert_allclose(
        np.asarray(theta_hat_F),
        np.average(np.asarray(theta), axis=0, weights=np.asarray(wnorm)),
        rtol=1e-10,
    )
    # Prediction covariance is the equal-weight covariance.
    expected = np.cov(np.asarray(theta).T, bias=True)
    np.testing.assert_allclose(np.asarray(V_P), expected, atol=1e-10)


def test_eq21_update_scalar_closed_form():
    # p = 1: theta_next = theta_m + V_P[0] * sum_n (diff_n / V_P_n), diff_0 = hat_0 - theta_m.
    theta_m = jnp.array([0.2])
    theta_hat_F = jnp.array([[0.5], [0.7], [0.65]])
    V_P = jnp.array([[[0.4]], [[0.3]], [[0.5]]])
    ridge = 1e-6
    got = _eq21_update(theta_m, theta_hat_F, V_P, ridge)
    Vp = np.asarray(V_P)[:, 0, 0] + ridge
    hats = np.asarray(theta_hat_F)[:, 0]
    prev = np.concatenate([[0.2], hats[:-1]])
    diffs = hats - prev
    expected = 0.2 + Vp[0] * np.sum(diffs / Vp)
    np.testing.assert_allclose(np.asarray(got)[0], expected, rtol=1e-9)


def test_eq21_update_diagonal_two_params():
    theta_m = jnp.array([0.0, 1.0])
    theta_hat_F = jnp.array([[0.1, 1.2], [0.3, 1.1]])
    V_P = jnp.stack([jnp.diag(jnp.array([0.5, 0.2])), jnp.diag(jnp.array([0.4, 0.3]))])
    got = _eq21_update(theta_m, theta_hat_F, V_P, 0.0)
    # Each coordinate decouples for diagonal V_P.
    for d in range(2):
        Vp = np.asarray(V_P)[:, d, d]
        hats = np.asarray(theta_hat_F)[:, d]
        prev = np.concatenate([[np.asarray(theta_m)[d]], hats[:-1]])
        diffs = hats - prev
        expected = np.asarray(theta_m)[d] + Vp[0] * np.sum(diffs / Vp)
        np.testing.assert_allclose(np.asarray(got)[d], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Linear-Gaussian correctness against the exact MLE
# ---------------------------------------------------------------------------


def _exact_alpha_mle(obs_times, obs_values, grid):
    def mll(a):
        ll, _ = compute_cuthbert_filter(
            M.make_lti_model(jnp.array([a])),
            KFConfig(),
            obs_times=obs_times,
            obs_values=obs_values,
        )
        return ll

    lls = jnp.array([mll(float(a)) for a in grid])
    return float(grid[int(jnp.argmax(lls))])


def test_linear_gaussian_recovers_exact_mle():
    times = jnp.arange(0.0, 150.0, 1.0)
    obs, _ = M.simulate_pomp(M.make_lti_model, jnp.array([0.4]), jr.PRNGKey(0), times)

    grid = jnp.linspace(-0.69, 0.69, 120)
    alpha_mle = _exact_alpha_mle(times, obs, grid)

    cfg = IF1Config(n_iterations=40, n_particles=1000, cooling_fraction=0.95)
    res = run_iterated_filtering(
        M.make_lti_model,
        cfg,
        jr.PRNGKey(1),
        init_theta=jnp.array([0.0]),
        obs_times=times,
        obs_values=obs,
        rw_sd=jnp.array([0.5]),
        transform=M.lti_transform(),
        control_dim=0,
    )
    assert abs(float(res.theta[0]) - alpha_mle) < 0.04
    # The likelihood estimate improves from start to finish.
    early = float(jnp.mean(res.loglik_history[:5]))
    late = float(jnp.mean(res.loglik_history[-5:]))
    assert late > early


# ---------------------------------------------------------------------------
# Nonlinear parameter recovery
# ---------------------------------------------------------------------------


def test_stochastic_volatility_recovers_phi():
    times = jnp.arange(0.0, 200.0, 1.0)
    true_phi = 0.9
    obs, _ = M.simulate_pomp(
        M.make_stoch_vol_model, jnp.array([true_phi]), jr.PRNGKey(0), times
    )
    cfg = IF1Config(n_iterations=40, n_particles=2000, cooling_fraction=0.95)
    res = IF1Inference(cfg, M.make_stoch_vol_model, M.stoch_vol_transform()).run(
        jr.PRNGKey(2),
        init_theta=jnp.array([0.3]),
        obs_times=times,
        obs_values=obs,
        rw_sd=jnp.array([0.4]),
        control_dim=0,
    )
    phi = float(res.theta[0])
    # The persistence is high and the likelihood surface in phi is flat, so the tolerance is
    # loose. The estimate must still land in the upper range and improve the likelihood.
    assert 0.7 < phi < 0.97
    assert float(jnp.mean(res.loglik_history[-5:])) > float(jnp.mean(res.loglik_history[:5]))


def test_discrete_l63_recovers_rho():
    times = jnp.arange(0.0, 3.0, 0.05)
    obs, _ = M.simulate_pomp(M.make_l63_model, jnp.array([28.0]), jr.PRNGKey(0), times)
    cfg = IF1Config(n_iterations=40, n_particles=4000, cooling_fraction=0.9)
    res = run_iterated_filtering(
        M.make_l63_model,
        cfg,
        jr.PRNGKey(3),
        init_theta=jnp.array([20.0]),
        obs_times=times,
        obs_values=obs,
        rw_sd=jnp.array([0.6]),
        transform=M.l63_transform(),
        control_dim=0,
    )
    assert abs(float(res.theta[0]) - 28.0) < 4.0
