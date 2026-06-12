# Iterated filtering (IF1) with dynestyx

This experiment implements the IF1 algorithm of Ionides, Bretó and King, "Inference for
nonlinear dynamical systems" (PNAS 2006), with the moment-matching update stated in the
Annals of Statistics 2011 follow-up. IF1 is a plug-and-play maximum-likelihood estimator for a
partially observed Markov process. It needs only the ability to simulate the latent transition
and evaluate the observation density, which a dynestyx `DynamicalModel` provides through
`.sample()` and `.log_prob()`.

## The algorithm

IF1 estimates a parameter vector `theta` by iterating a bootstrap particle filter in which
every particle carries its own copy of `theta`. Across the observation times each particle's
parameter performs a Gaussian random walk, and the random-walk standard deviation is cooled
geometrically across iterations.

At iteration `m` with current estimate `theta_m` and cooling level `sigma_m = rw_sd * alpha^m`:

1. Spread `J` particles around `theta_m` and draw each particle's initial state.
2. For each observation time `n`, perturb each particle's parameter by `sigma_m`, propagate the
   state under that particle's own parameter, weight by the observation density, record the
   weighted filter mean `theta_hat^F_n` and the equal-weight prediction covariance `V^P_n` of
   the predicted parameters, then resample.
3. Update
   `theta_{m+1} = theta_m + V^P_1 sum_n (V^P_n)^{-1} (theta_hat^F_n - theta_hat^F_{n-1})`
   with `theta_hat^F_0 = theta_m`.

The perturbation and the update run on an unconstrained scale. A bijector maps unconstrained
parameters to the constrained scale that `make_model` expects, so the random walk never leaves
the feasible set.

## Files

- `iterated_filtering.py` — the IF1 core: `IF1Config`, `IF1Result`, `IF1Inference`,
  `run_iterated_filtering`, the `BoxTransform` bijector, and the internal filter. It reuses the
  dynestyx model contract and the systematic resampling routine from cuthbertlib. It builds a
  dedicated instrumented filter rather than the package filter, because IF1 needs the weighted
  parameter moments of the predicted particles before resampling and because each particle must
  propagate under its own parameter.
- `models.py` — model factories `make_lti_model` (free `alpha`), `make_stoch_vol_model`
  (free `phi`), `make_l63_model` (free `rho`), their bijectors, and `simulate_pomp` for drawing
  a trajectory and its observations through the same contract the filter uses.
- `test_iterated_filtering.py` — deterministic unit tests, a linear-Gaussian test against the
  exact Kalman MLE, and nonlinear recovery tests.
- `run_demo.py` — runs all three demonstrations and writes figures.

## Running

From the repository root:

```
uv run pytest experiments/iterated_filtering/test_iterated_filtering.py -q
uv run python -m experiments.iterated_filtering.run_demo
```

The demo prints the recovered parameter next to the truth, prints the exact MLE for the
linear-Gaussian case, and writes the figures `demo_linear_gaussian.png`,
`demo_linear_gaussian_likelihood.png`, `demo_stochastic_volatility.png`, and
`demo_lorenz63.png`.

## What the demonstrations show

- **Linear-Gaussian.** The exact MLE is available from the Kalman marginal likelihood. IF1
  converges to it. The likelihood figure marks the IF1 estimate on the exact likelihood curve.
- **Stochastic volatility.** A non-Gaussian observation density that genuinely requires the
  particle filter. IF1 recovers the AR(1) persistence on a flat likelihood surface.
- **Discrete Lorenz 63.** Nonlinear chaotic dynamics observed through a single noisy
  coordinate. IF1 recovers the drift parameter `rho` from a short trajectory.
