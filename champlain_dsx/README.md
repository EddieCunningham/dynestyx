# champlain_dsx

A JAX port of the acoustic-telemetry geolocation state-space model of Lavender
et al., built on the dynestyx framework. The model reconstructs animal
trajectories in Lake Champlain from sparse acoustic detections. It reproduces
the paper's simulation analysis on the lake's real geometry.

## The model

The latent state is `s_t = (x, y, phi)`, a projected position in metres and a
heading in radians. The posterior over trajectories factorises into a movement
prior and an observation likelihood,

    f(s_{1:T} | y_{1:T}) ∝ f(s_1) ∏_t f(s_t | s_{t-1}) ∏_t f(y_t | s_t).

### Movement model (prior)

The transition is a land-truncated correlated random walk. A step length is
drawn from a Gamma distribution truncated above by a mobility bound. A turning
angle is drawn from a mixture of a truncated Normal and a uniform escape
component. The proposed position is rejected when it leaves navigable water.

    d_t      ~ Gamma(k, theta) truncated to (0, mobility]
    dphi_t   ~ 0.99 * TruncatedNormal(0, sigma, -pi, pi) + 0.01 * Uniform(-pi, pi)
    phi_t    = phi_{t-1} + dphi_t
    x_t      = x_{t-1} + d_t cos(phi_t)
    y_t      = y_{t-1} + d_t sin(phi_t)

The land-truncated density is intractable. The bootstrap particle filter and the
tracing particle smoother consume the transition through `sample` alone, so the
density is never needed. This is implemented in
[distributions.py](champlain_dsx/distributions.py) as a NumPyro distribution
whose `sample` performs fixed-attempt rejection and whose `log_prob` raises.

### Observation model (likelihood)

Each receiver reports a detection or non-detection at every step. Detection
probability declines logistically with distance, drops to zero beyond a maximum
range, and drops to zero when the straight-line midpoint falls on land.

    p(s_t, r_k) = logistic(alpha + beta |s_t - r_k|)   if |s_t - r_k| < gamma and line of sight
                = 0                                      otherwise
    y_{t,k} ~ Bernoulli(p(s_t, r_k))

Detections at distinct receivers are conditionally independent given the state.
See [observation.py](champlain_dsx/observation.py).

### Inference

Inference uses the bootstrap particle filter `PFConfig` for the filtering
distribution and the tracing particle smoother `PFSmootherConfig` for the
marginal smoothing distribution, matching the particle algorithms in the paper.
Smoothed particles and weights form an occupancy distribution over the lake grid,
the discrete analogue of the paper's occurrence probability. See
[inference.py](champlain_dsx/inference.py).

### Default parameters

The defaults in [model.py](champlain_dsx/model.py) are the paper's best
parameterisation. Step length is `Gamma(3.25, 25)` truncated at `216` m. The
turning angle uses `sigma = 0.4`. Detection uses `alpha = 0.9039`,
`beta = -0.0021`, and `gamma = 7000` m. The paper calibrates these from
telemetry and range-test data and fixes them during trajectory reconstruction,
so the model infers states with parameters held fixed.

## The environment

[environment.py](champlain_dsx/environment.py) holds a navigability mask, an
open-water mask, and a depth field on one affine grid. The summer option drops
cells shallower than the 20 m thermal threshold the paper uses.

The real Lake Champlain geometry is reconstructed from two public Vermont Open
Geodata layers, the lake outline polygon and the bathymetry point cloud, both in
EPSG:32145. The proprietary survey rasters in the paper are not public, so this
open reconstruction stands in for them. Acquire the raster with

    uv run python champlain_dsx/scripts/acquire_champlain_map.py --res 200

This writes `data/champlain_map.npz`. A self-contained synthetic lake is
available through `synthetic_champlain` when the download is not run.

## Running

Tests.

    uv run pytest champlain_dsx/tests -q

Simulation study on the real lake.

    uv run python champlain_dsx/scripts/run_simulation_study.py --map data/champlain_map.npz

Simulation study on the synthetic lake.

    uv run python champlain_dsx/scripts/run_simulation_study.py --map synthetic

The study simulates trajectories and detections, reconstructs each with the
filter and the smoother, reports recovery accuracy and posterior calibration,
and saves a figure of one reconstruction with its occupancy distribution.

## Scope

The geometry, movement model, observation model, and particle algorithms follow
the paper. Detections are simulated rather than read from the private fish
archive, which matches the paper's simulation analysis. Receivers are placed on
a grid over open water as a stand-in for the real mooring array.

## Layout

- [environment.py](champlain_dsx/environment.py) raster lake, masks, line of sight
- [distributions.py](champlain_dsx/distributions.py) movement transition and initial condition
- [observation.py](champlain_dsx/observation.py) acoustic detection model
- [model.py](champlain_dsx/model.py) dynestyx model assembly and parameters
- [simulate.py](champlain_dsx/simulate.py) receiver placement and simulation
- [inference.py](champlain_dsx/inference.py) filter, smoother, occupancy maps
- [metrics.py](champlain_dsx/metrics.py) recovery and calibration metrics
- [scripts/](scripts/) map acquisition and the simulation study
- [tests/](tests/) unit and end-to-end tests
