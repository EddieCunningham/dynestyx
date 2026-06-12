"""Simulation study for the Lake Champlain geolocation model.

The study mirrors the paper's simulation analysis. It draws trajectories from
the movement model, generates acoustic detections from the observation model,
and reconstructs each trajectory with the bootstrap particle filter and the
tracing particle smoother. It then reports recovery accuracy and posterior
calibration, and saves a figure of one reconstructed trajectory with its
occupancy distribution.

Run on the real lake (after acquiring the map) or on the synthetic fallback:

    uv run python champlain_dsx/scripts/run_simulation_study.py --map data/champlain_map.npz
    uv run python champlain_dsx/scripts/run_simulation_study.py --map synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from champlain_dsx.environment import load_champlain, synthetic_champlain
from champlain_dsx.inference import (
    occupancy_map,
    run_filter,
    run_smoother,
    smoothed_mean,
)
from champlain_dsx.metrics import (
    credible_region_coverage,
    occupancy_total_variation,
    position_rmse,
    spatial_uncertainty,
    true_occupancy_map,
)
from champlain_dsx.model import ChamplainParameters, make_model
from champlain_dsx.simulate import grid_receivers, simulate


def load_environment(map_arg: str, coarsen: int, summer: bool):
    if map_arg == "synthetic":
        return synthetic_champlain(res=400.0, summer=summer)
    return load_champlain(map_arg, summer=summer, coarsen=coarsen)


def run_study(args):
    env = load_environment(args.map, args.coarsen, args.summer)
    receivers = grid_receivers(env, spacing=args.receiver_spacing)
    model = make_model(env, receivers, ChamplainParameters())
    times = jnp.arange(args.steps, dtype=jnp.float32)

    key = jr.PRNGKey(args.seed)
    k_sim, k_inf = jr.split(key)
    sim = simulate(model, times, k_sim, n_sim=args.n_traj)

    print(
        f"Environment {env.height}x{env.width} at {env.res:.0f} m, "
        f"{int(env.nav_mask.sum())} navigable cells, {receivers.shape[0]} receivers"
    )
    print(f"Simulated {args.n_traj} trajectories of {args.steps} steps\n")

    rows = []
    inf_keys = jr.split(k_inf, args.n_traj)
    for i in range(args.n_traj):
        obs = sim["observations"][i]
        truth = sim["states"][i]
        n_det = int(obs.sum())

        fout = run_filter(model, times, obs, inf_keys[i], n_particles=args.n_particles)
        sout = run_smoother(model, times, obs, inf_keys[i], n_particles=args.n_particles)

        filt_rmse = position_rmse(truth, np.asarray(fout["f_filtered_states_mean"][0]))
        smooth_rmse = position_rmse(truth, smoothed_mean(sout))

        parts = np.asarray(sout["f_smoothed_particles"][0])
        lw = np.asarray(sout["f_smoothed_log_weights"][0])
        spread = spatial_uncertainty(parts, lw)
        coverage = credible_region_coverage(env, parts, lw, truth, level=0.9)
        est_map = occupancy_map(env, parts, lw)
        tv = occupancy_total_variation(est_map, true_occupancy_map(env, truth))

        rows.append(
            dict(
                trajectory=i,
                detections=n_det,
                filter_rmse_m=filt_rmse,
                smoother_rmse_m=smooth_rmse,
                spread_m=spread,
                coverage_90=coverage,
                occupancy_tv=tv,
            )
        )
        print(
            f"  traj {i:2d}: det={n_det:4d}  filt_rmse={filt_rmse:7.0f}  "
            f"smooth_rmse={smooth_rmse:7.0f}  spread={spread:7.0f}  "
            f"cov90={coverage:.2f}  tv={tv:.2f}"
        )

    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    print("\nMedian over trajectories")
    print(f"  filter RMSE   : {np.median(arr['filter_rmse_m']):.0f} m")
    print(f"  smoother RMSE : {np.median(arr['smoother_rmse_m']):.0f} m")
    print(f"  posterior spread : {np.median(arr['spread_m']):.0f} m")
    print(f"  90% coverage  : {np.median(arr['coverage_90']):.2f}")
    print(f"  occupancy TV  : {np.median(arr['occupancy_tv']):.2f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    _save_figure(env, receivers, sim, model, times, inf_keys, args, out_dir)
    print(f"\nWrote {out_dir / 'metrics.json'} and {out_dir / 'reconstruction.png'}")


def _save_figure(env, receivers, sim, model, times, inf_keys, args, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick the trajectory with the most detections for a clear illustration.
    counts = sim["observations"].sum(axis=(1, 2))
    i = int(np.argmax(counts))
    truth = sim["states"][i]
    sout = run_smoother(model, times, sim["observations"][i], inf_keys[i],
                        n_particles=args.n_particles)
    parts = np.asarray(sout["f_smoothed_particles"][0])
    lw = np.asarray(sout["f_smoothed_log_weights"][0])
    est_map = occupancy_map(env, parts, lw)
    est = smoothed_mean(sout)

    extent = [env.x_min, env.x_max, env.y_min, env.y_max]
    water = np.asarray(env.water_mask)

    fig, ax = plt.subplots(figsize=(5, 11))
    ax.imshow(water, extent=extent, origin="upper", cmap="Blues", alpha=0.35)
    shown = np.where(est_map > 0, est_map, np.nan)
    ax.imshow(shown, extent=extent, origin="upper", cmap="viridis")
    ax.plot(truth[:, 0], truth[:, 1], color="white", lw=1.2, label="true path")
    ax.plot(est[:, 0], est[:, 1], color="red", lw=1.0, ls="--", label="smoothed mean")
    ax.scatter(receivers[:, 0], receivers[:, 1], c="black", s=6, marker="^",
               label="receivers")
    ax.set_title("Smoothed occupancy and reconstructed path")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="data/champlain_map.npz",
                        help="Path to an acquired .npz map, or 'synthetic'.")
    parser.add_argument("--coarsen", type=int, default=3)
    parser.add_argument("--summer", action="store_true")
    parser.add_argument("--n-traj", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--n-particles", type=int, default=3000)
    parser.add_argument("--receiver-spacing", type=float, default=4000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
