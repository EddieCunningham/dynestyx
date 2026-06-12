"""A JAX port of the Lavender et al. acoustic-telemetry geolocation model.

The package reconstructs animal trajectories from sparse acoustic detections
using the dynestyx state-space framework. It pairs a land-truncated correlated
random walk movement model with a Bernoulli acoustic observation model and runs
inference with a bootstrap particle filter and a tracing particle smoother.
"""

from champlain_dsx.distributions import CorrelatedRandomWalk, UniformOnWater
from champlain_dsx.environment import (
    LakeEnvironment,
    environment_from_arrays,
    synthetic_champlain,
)
from champlain_dsx.inference import (
    occupancy_map,
    run_filter,
    run_smoother,
    smoothed_mean,
)
from champlain_dsx.model import ChamplainParameters, build_dynamics, make_model
from champlain_dsx.observation import detection_probability, make_observation_model
from champlain_dsx.simulate import grid_receivers, random_receivers, simulate

__all__ = [
    "LakeEnvironment",
    "environment_from_arrays",
    "synthetic_champlain",
    "CorrelatedRandomWalk",
    "UniformOnWater",
    "detection_probability",
    "make_observation_model",
    "ChamplainParameters",
    "build_dynamics",
    "make_model",
    "grid_receivers",
    "random_receivers",
    "simulate",
    "run_filter",
    "run_smoother",
    "occupancy_map",
    "smoothed_mean",
]
