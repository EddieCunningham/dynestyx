"""Tests for the model components: initial condition, movement, observation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from champlain_dsx import (
    DetectionModel,
    MoveModel,
    MoveParams,
    ObsParams,
    UniformOnWater,
    build_model,
    default_receivers,
    make_synthetic_lake,
)
from champlain_dsx.model import _CRWTransition


@pytest.fixture(scope="module")
def lake():
    geo = make_synthetic_lake(nx=60, ny=60, cell=300.0, n_regions=3)
    return geo, default_receivers(geo)


def test_initial_condition_samples_on_water(lake):
    geo, _ = lake
    ic = UniformOnWater(geo.water_cell_centres(), geo.cell)
    keys = jr.split(jr.PRNGKey(0), 300)
    samples = jax.vmap(ic._sample_one)(keys)
    assert samples.shape == (300, 3)
    assert bool(geo.is_water(samples[:, 0], samples[:, 1]).all())
    phi = samples[:, 2]
    assert bool((phi >= -jnp.pi).all()) and bool((phi <= jnp.pi).all())


def test_crw_steps_stay_on_water_and_respect_mobility(lake):
    geo, _ = lake
    params = MoveParams()
    # Start from a water cell near the centre.
    centres = geo.water_cell_centres()
    start = jnp.array([centres[len(centres) // 2][0], centres[len(centres) // 2][1], 0.3])
    trans = _CRWTransition(start, geo, params)
    keys = jr.split(jr.PRNGKey(1), 500)
    nexts = jax.vmap(trans._sample_one)(keys)
    # All proposals that moved must land on water.
    on_water = geo.is_water(nexts[:, 0], nexts[:, 1])
    assert bool(on_water.all())
    # Step length never exceeds the mobility bound.
    step = jnp.sqrt((nexts[:, 0] - start[0]) ** 2 + (nexts[:, 1] - start[1]) ** 2)
    assert float(step.max()) <= params.mobility + 1e-3
    # Heading stays wrapped.
    assert bool((jnp.abs(nexts[:, 2]) <= jnp.pi + 1e-5).all())


def test_crw_mean_step_matches_truncated_gamma(lake):
    geo, _ = lake
    # Use a wide-open patch so land rejection rarely fires, then the empirical
    # mean step should sit near the truncated Gamma mean.
    params = MoveParams()
    cx, cy = geo.width / 2.0, geo.height / 2.0
    start = jnp.array([cx, cy + 0.35 * geo.height, 0.0])
    trans = _CRWTransition(start, geo, params)
    keys = jr.split(jr.PRNGKey(2), 4000)
    nexts = jax.vmap(trans._sample_one)(keys)
    step = jnp.sqrt((nexts[:, 0] - start[0]) ** 2 + (nexts[:, 1] - start[1]) ** 2)
    # Truncated Gamma(3.25, 25) on [0, 216] has mean ~76 m.
    assert 55.0 < float(step.mean()) < 95.0


def test_detection_probability_truncation_and_los(lake):
    geo, recv = lake
    obs = DetectionModel(geo, recv, ObsParams())
    # A point right at a receiver: probability is the logistic intercept and the
    # receiver sees itself (zero distance, trivial line of sight).
    r = recv[0]
    p_at = obs.detection_prob(r[0], r[1])
    assert 0.0 < float(p_at[0]) <= jax.nn.sigmoid(jnp.array(ObsParams().alpha)) + 1e-6
    # Probability decays with distance: a point 1 km away has lower probability.
    p_far = obs.detection_prob(r[0] + 1000.0, r[1])
    # (only compare the same receiver index 0 when still in water/LOS)
    if float(p_far[0]) > 0:
        assert float(p_far[0]) < float(p_at[0])
    # Beyond gamma the probability is exactly zero.
    p_beyond = jax.nn.sigmoid(
        jnp.array(ObsParams().alpha + ObsParams().beta * (ObsParams().gamma + 10.0))
    )
    obs_far = obs.detection_prob(r[0] + ObsParams().gamma + 10.0, r[1])
    assert float(obs_far[0]) == 0.0


def test_detection_prob_zero_without_line_of_sight():
    # Place a single receiver north of the peninsula and read the detection
    # probability at a water point directly south of it. The straight path
    # crosses land, so the line-of-sight term must zero the probability, even
    # though the same distance with a clear path gives a positive probability.
    geo = make_synthetic_lake(nx=100, ny=100, cell=200.0)
    cx, cy = geo.width / 2.0, geo.height / 2.0
    y_pen = cy - 0.06 * geo.height
    receiver = jnp.array([[cx - 0.2 * geo.width, y_pen + 0.12 * geo.height]])
    south = jnp.array([cx - 0.2 * geo.width, y_pen - 0.12 * geo.height])
    assert bool(geo.is_water(south[0], south[1]))
    assert not bool(geo.line_of_sight(south[0], south[1], receiver[0, 0], receiver[0, 1]))

    obs = DetectionModel(geo, receiver, ObsParams())
    p_blocked = obs.detection_prob(south[0], south[1])
    assert float(p_blocked[0]) == 0.0

    # A clear path of comparable distance to the receiver gives positive prob.
    near = jnp.array([receiver[0, 0], receiver[0, 1] - geo.cell])
    assert bool(geo.is_water(near[0], near[1]))
    p_clear = obs.detection_prob(near[0], near[1])
    assert float(p_clear[0]) > 0.0


def test_dynamical_model_infers_dims(lake):
    geo, recv = lake
    from numpyro.handlers import seed, trace

    model = build_model(geo, recv)
    # Trace once under a simulator so dynestyx builds the DynamicalModel and we
    # can read the inferred dimensions back off the observation site.
    obs_times = jnp.arange(5)
    from dynestyx import DiscreteTimeSimulator
    from numpyro.infer import Predictive

    with DiscreteTimeSimulator():
        out = Predictive(model, num_samples=1, exclude_deterministic=False)(
            jr.PRNGKey(0), predict_times=obs_times
        )
    assert out["fish_states"].shape[-1] == 3
    assert out["fish_observations"].shape[-1] == recv.shape[0]


def test_simulated_states_stay_on_water(lake):
    geo, recv = lake
    from champlain_dsx import simulate_dataset

    model = build_model(geo, recv)
    data = simulate_dataset(model, n_steps=150, key=jr.PRNGKey(3))
    states = data["states"]
    on_water = np.asarray(
        geo.is_water(jnp.asarray(states[:, 0]), jnp.asarray(states[:, 1]))
    )
    assert on_water.all()
    assert data["obs_values"].shape == (150, recv.shape[0])
