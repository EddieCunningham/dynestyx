"""Demonstration driver for the IF1 iterated-filtering experiment.

Running this script estimates a parameter in three models and writes a figure for each one.
The linear-Gaussian case also overlays the exact Kalman marginal likelihood so the IF1 estimate
can be read off against the true likelihood maximum. Figures are written next to this file.
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import compute_cuthbert_filter

from experiments.iterated_filtering import models as M
from experiments.iterated_filtering.iterated_filtering import (
    IF1Config,
    run_iterated_filtering,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _plot_traces(theta_history, loglik_history, true_value, title, path, reference=None):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(theta_history[:, 0], marker="o", ms=3)
    ax0.axhline(true_value, color="k", ls="--", label="true")
    if reference is not None:
        ax0.axhline(reference, color="C3", ls=":", label="exact MLE")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel("parameter estimate")
    ax0.set_title(title)
    ax0.legend()
    ax1.plot(loglik_history, marker="o", ms=3)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("filter log-likelihood estimate")
    ax1.set_title("log-likelihood per iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def demo_linear_gaussian():
    times = jnp.arange(0.0, 150.0, 1.0)
    obs, _ = M.simulate_pomp(M.make_lti_model, jnp.array([0.4]), jr.PRNGKey(0), times)

    grid = jnp.linspace(-0.69, 0.69, 140)
    lls = jnp.array(
        [
            compute_cuthbert_filter(
                M.make_lti_model(jnp.array([float(a)])),
                KFConfig(),
                obs_times=times,
                obs_values=obs,
            )[0]
            for a in grid
        ]
    )
    alpha_mle = float(grid[int(jnp.argmax(lls))])

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
    alpha_hat = float(res.theta[0])
    print(
        f"[linear-Gaussian] true alpha=0.400  exact MLE={alpha_mle:.3f}  IF1={alpha_hat:.3f}"
    )

    _plot_traces(
        res.theta_history,
        res.loglik_history,
        0.4,
        "linear-Gaussian: alpha",
        os.path.join(_HERE, "demo_linear_gaussian.png"),
        reference=alpha_mle,
    )

    # Likelihood curve with the IF1 estimate marked.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(jnp.asarray(grid), jnp.asarray(lls), color="C0")
    ax.axvline(alpha_mle, color="C3", ls=":", label=f"exact MLE = {alpha_mle:.3f}")
    ax.axvline(alpha_hat, color="C1", ls="--", label=f"IF1 = {alpha_hat:.3f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("exact Kalman marginal log-likelihood")
    ax.set_title("linear-Gaussian likelihood")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, "demo_linear_gaussian_likelihood.png"), dpi=120)
    plt.close(fig)


def demo_stochastic_volatility():
    times = jnp.arange(0.0, 200.0, 1.0)
    obs, _ = M.simulate_pomp(M.make_stoch_vol_model, jnp.array([0.9]), jr.PRNGKey(0), times)
    cfg = IF1Config(n_iterations=40, n_particles=2000, cooling_fraction=0.95)
    res = run_iterated_filtering(
        M.make_stoch_vol_model,
        cfg,
        jr.PRNGKey(2),
        init_theta=jnp.array([0.3]),
        obs_times=times,
        obs_values=obs,
        rw_sd=jnp.array([0.4]),
        transform=M.stoch_vol_transform(),
        control_dim=0,
    )
    print(f"[stochastic volatility] true phi=0.900  IF1={float(res.theta[0]):.3f}")
    _plot_traces(
        res.theta_history,
        res.loglik_history,
        0.9,
        "stochastic volatility: phi",
        os.path.join(_HERE, "demo_stochastic_volatility.png"),
    )


def demo_lorenz63():
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
    print(f"[discrete Lorenz 63] true rho=28.00  IF1={float(res.theta[0]):.2f}")
    _plot_traces(
        res.theta_history,
        res.loglik_history,
        28.0,
        "discrete Lorenz 63: rho",
        os.path.join(_HERE, "demo_lorenz63.png"),
    )


if __name__ == "__main__":
    demo_linear_gaussian()
    demo_stochastic_volatility()
    demo_lorenz63()
    print(f"Figures written to {_HERE}")
