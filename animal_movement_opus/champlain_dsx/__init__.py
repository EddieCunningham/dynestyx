"""A minimal dynestyx port of the acoustic-telemetry geolocation state-space model.

This package reproduces the model of Lavender et al. (Big data analysis of animal
movements in aquatic ecosystems with acoustic telemetry) on a small synthetic
lake. The latent state is a two-dimensional location with a heading. Movement is
a land-truncated correlated random walk. Observations are per-receiver Bernoulli
detections with a logistic distance-decay and a line-of-sight requirement.
Inference targets the latent states through a bootstrap particle filter and a
particle smoother.
"""

from .geography import Geography, default_receivers, make_synthetic_lake
from .model import (
    DetectionModel,
    MoveModel,
    MoveParams,
    ObsParams,
    UniformOnWater,
    build_model,
)
from .inference import run_filter, run_smoother, simulate_dataset
from .occupancy import (
    occupancy_area_cells,
    occurrence_from_particles,
    occurrence_map,
    residency_from_particles,
    residency_from_track,
)

__all__ = [
    "Geography",
    "make_synthetic_lake",
    "default_receivers",
    "build_model",
    "MoveModel",
    "DetectionModel",
    "MoveParams",
    "ObsParams",
    "UniformOnWater",
    "occurrence_map",
    "occurrence_from_particles",
    "residency_from_track",
    "residency_from_particles",
    "occupancy_area_cells",
    "simulate_dataset",
    "run_filter",
    "run_smoother",
]
