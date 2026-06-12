"""End-to-end inference tests: the particle filter and smoother recover truth.

These are the correctness signals for the port. On simulated data the smoother
should reconstruct regional residency close to the truth and localise the animal
to a small part of the lake, mirroring the paper's simulation analysis.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from champlain_dsx import (
    build_model,
    default_receivers,
    make_synthetic_lake,
    occupancy_area_cells,
    occurrence_from_particles,
    residency_from_particles,
    residency_from_track,
    run_filter,
    run_smoother,
    simulate_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    geo = make_synthetic_lake(nx=60, ny=60, cell=300.0, n_regions=3)
    recv = default_receivers(geo)
    model = build_model(geo, recv)
    data = simulate_dataset(
        model, n_steps=200, key=jr.PRNGKey(0), min_detections=20
    )
    return geo, recv, model, data


def test_simulated_dataset_has_signal(dataset):
    _, _, _, data = dataset
    assert data["n_detections"] >= 20


def test_filter_returns_finite_loglik(dataset):
    _, _, model, data = dataset
    res = run_filter(
        model,
        data["obs_times"],
        data["obs_values"],
        jr.PRNGKey(1),
        n_particles=1500,
    )
    assert np.isfinite(res["marginal_loglik"])
    assert res["filtered_mean"].shape == (200, 3)


def test_smoother_recovers_residency(dataset):
    geo, _, model, data = dataset
    res = run_smoother(
        model,
        data["obs_times"],
        data["obs_values"],
        jr.PRNGKey(2),
        n_particles=2000,
        n_smoother_particles=300,
    )
    parts = res["particles"][..., :2]
    lw = res["log_weights"]

    res_true = np.asarray(residency_from_track(geo, jnp.asarray(data["states"][:, :2])))
    res_inf = np.asarray(residency_from_particles(geo, parts, lw))

    # Residency vectors are valid probability vectors.
    assert np.isclose(res_inf.sum(), 1.0, atol=1e-4)
    # The mean absolute residency error stays small, in the spirit of the
    # paper's Mean Weighted Occupancy Error (well under ten percentage points).
    mae = np.mean(np.abs(res_true - res_inf))
    assert mae < 0.1, f"residency MAE too large: {mae:.3f}"
    # The animal is localised to a modest part of the lake, not smeared across it.
    occ = occurrence_from_particles(geo, parts, lw)
    n_water = int(geo.water.sum())
    assert occupancy_area_cells(occ, 0.95) < 0.5 * n_water


def test_smoothed_mean_tracks_truth(dataset):
    geo, _, model, data = dataset
    res = run_smoother(
        model,
        data["obs_times"],
        data["obs_values"],
        jr.PRNGKey(3),
        n_particles=2000,
        n_smoother_particles=300,
    )
    sm = res["smoothed_mean"][:, :2]
    truth = data["states"][:, :2]
    rmse = float(np.sqrt(np.mean((sm - truth) ** 2)))
    # Localisation error is on the order of the detection-decay length scale,
    # comfortably below the lake extent.
    assert rmse < 0.25 * geo.width
