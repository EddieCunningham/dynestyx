# champlain_dsx

A minimal, tested [dynestyx](../dynestyx) port of the acoustic-telemetry
geolocation state-space model from Lavender et al., *Big data analysis of animal
movements in aquatic ecosystems with acoustic telemetry* (bioRxiv,
10.64898/2026.06.01.729394). The original analysis tracks lake trout in Lake
Champlain with the `patter` framework. This port keeps the model and swaps the
real lake raster for a small synthetic lake so the whole pipeline runs and tests
end to end.

## The model

The latent state is a location and a heading, `s_t = (x, y, phi)`. Time is
discrete.

- **Initial condition (eqn 3).** Location uniform over water cells, heading
  uniform on `(-pi, pi)`.
- **Movement (eqns 4 to 6).** A land-truncated correlated random walk. A step
  length is drawn from a Gamma truncated at the mobility bound and a turning
  angle from a `0.99 Normal + 0.01 Uniform` mixture. The next location and
  heading follow equation 4. A proposed step that lands on land is rejected and
  redrawn.
- **Observation (eqns 7 to 9).** One independent Bernoulli detection per
  receiver. Detection probability decays logistically with distance, is zero
  beyond a maximum range `gamma`, and is zero without a line of sight to the
  receiver. The line of sight is approximated by whether the receiver-to-animal
  midpoint is on water, following the paper's Julia source.

All parameters are fixed at the paper's "best" values
(`k=3.25, theta=25, mobility=216, sigma=0.4, alpha=0.904, beta=-0.002,
gamma=7000`). Inference targets the latent states.

## Mapping to dynestyx

The movement step is a deterministic function of two random variables, so its
density in `(x, y, phi)` is degenerate. The bootstrap particle filter and the
genealogy-tracing particle smoother need only to **sample** the transition, which
is what the model provides.

| paper | dynestyx |
| --- | --- |
| movement process `f(s_t \| s_{t-1})` | `state_evolution` returning `_CRWTransition` (sampling only) |
| detection likelihood `f(y_t \| s_t)` | `observation_model` returning `Independent(Bernoulli, 1)` |
| initial state `f(s_1)` | `UniformOnWater` initial condition |
| particle filter | `PFConfig` (bootstrap particle filter) |
| two-filter smoothing of `f(s_t \| y_{1:T})` | `PFSmootherConfig` (particle smoother, tracing backward pass) |

## Layout

```
animal_movement_opus/
  champlain_dsx/
    geography.py   synthetic water mask, grid lookups, line of sight, regions
    model.py       initial condition, CRW transition, detection model, build_model
    inference.py   simulate_dataset, run_filter, run_smoother
    occupancy.py   occurrence maps and regional residency
  tests/           geography, model, and end-to-end recovery tests
  build_notebook.py  regenerates demo.ipynb
  demo.ipynb       runnable walkthrough with plots
```

## Run

Everything runs in the parent project's uv environment.

```bash
# tests (the recovery tests take a couple of minutes)
uv run python -m pytest animal_movement_opus/tests -q

# regenerate and execute the demo notebook
cd animal_movement_opus
uv run python build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace demo.ipynb
```

## Correctness signal

On simulated data the smoother reconstructs the animal's space use. The
occurrence map concentrates around the true track, the 95% occupancy area is a
small part of the lake, and regional residency is recovered to within a few
percentage points. This matches the accuracy the paper reports for its
simulation analysis, where the Mean Weighted Occupancy Error had a median of
0.7%. The same model function drives simulation, filtering and smoothing, so a
real raster and a real detection series would slot into the same pipeline.
```python
from champlain_dsx import (
    make_synthetic_lake, default_receivers, build_model,
    simulate_dataset, run_smoother, residency_from_track, residency_from_particles,
)
import jax.random as jr, jax.numpy as jnp, numpy as np

geo = make_synthetic_lake()
recv = default_receivers(geo)
model = build_model(geo, recv)
data = simulate_dataset(model, n_steps=400, key=jr.PRNGKey(0), min_detections=20)
sm = run_smoother(model, data["obs_times"], data["obs_values"], jr.PRNGKey(1))

res_true = residency_from_track(geo, jnp.asarray(data["states"][:, :2]))
res_inf = residency_from_particles(geo, sm["particles"][..., :2], sm["log_weights"])
print(np.round(np.c_[res_true, res_inf], 3))
```
