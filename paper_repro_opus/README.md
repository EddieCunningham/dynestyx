# Cholera reproduction: inference for nonlinear dynamical systems

A dynestyx reproduction of Ionides, Bretó and King, "Inference for nonlinear
dynamical systems," PNAS 103 (2006) 18438. The paper introduced maximum
likelihood via iterated filtering and applied it to cholera mortality in Dhaka,
Bangladesh.

## What is here

- `seasonal.py` builds the periodic cubic B-spline basis for seasonal
  transmission.
- `cholera.py` defines the stochastic SIRS dynamics, the custom Euler-Maruyama
  transition with a self-resetting monthly death accumulator, and the
  heteroscedastic count measurement. `THETA_STAR` holds the paper's Table 1
  parameters.
- `model.py` wires the dynamics into a NumPyro model function for dynestyx.
- `inference.py` provides simulation, data loading, the particle-filter
  marginal likelihood, profile sweeps, and a particle marginal
  Metropolis-Hastings sampler.
- `cholera_iterated_filtering.ipynb` is the narrative notebook with all results.
- `data/dhaka_cholera.csv` is the historical monthly mortality and
  census-interpolated population, from the `dacca` dataset of the `pomp` R
  package.
- `tests/` holds unit tests for the seasonal basis and the dynamics.

## Model

The latent state is a continuous-time stochastic SIRS process with three Erlang
immune classes, environmental noise on the susceptible-to-infected channel, and
seasonally forced transmission. Time is measured in months. The state vector
carried by the filter is `[S, I, R1, R2, R3, M]`, where `M` holds the cholera
deaths over the most recent month. Each monthly transition runs twenty
Euler-Maruyama substeps and recomputes `M` from zero, so the death accumulator
resets at every observation. The bootstrap particle filter proposes from this
transition and scores with the measurement density, so no transition density is
needed.

## Results

On data simulated at the Table 1 parameters, the particle-filter likelihood is
maximized at the generating values. Profile likelihoods recover the seasonal
amplitude, the environmental noise, and the reservoir strength. The reservoir
profile reproduces the likelihood-ratio evidence for a non-human reservoir. A
pseudo-marginal sampler returns posteriors over the environmental noise and the
measurement overdispersion that cover the truth. The same parameters generate
epidemics that match the historical Dhaka record in magnitude and seasonal
shape.

## Running

```
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 cholera_iterated_filtering.ipynb
uv run python -m pytest tests -q
```
