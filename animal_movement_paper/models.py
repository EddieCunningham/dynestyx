"""State-space model for acoustic telemetry animal tracking.

Implements the correlated random walk model from Lavender et al. for
reconstructing lake trout trajectories from sparse acoustic detections.
The state is (x, y, phi) where (x, y) is the 2D position in meters and
phi is the heading in radians.  Observations are binary detection vectors
from an array of acoustic receivers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions import constraints

from dynestyx.models.core import DiscreteTimeStateEvolution, DynamicalModel, ObservationModel

from animal_movement_paper.distributions import (
    step_length_log_prob,
    step_length_sample,
    turning_angle_log_prob,
    turning_angle_sample,
    wrap_angle,
)


class CorrelatedRandomWalkDistribution(dist.Distribution):
    """Distribution over the next state (x, y, phi) given the current state.

    The transition samples a step length d from a truncated Gamma and a turning
    angle delta_phi from a mixture of a truncated Normal and a Uniform, then
    updates the state deterministically:

        phi_new  = phi_old + delta_phi
        x_new    = x_old + d * cos(phi_new)
        y_new    = y_old + d * sin(phi_new)

    The log_prob inverts this map to recover (d, delta_phi) and applies the
    Jacobian correction for the polar-to-Cartesian change of variables.
    """

    arg_constraints: dict = {}
    support = constraints.real_vector

    def __init__(
        self,
        state: Float[Array, " 3"],
        concentration: float = 3.25,
        scale: float = 25.0,
        mobility: float = 216.0,
        turn_std: float = 0.4,
        turn_normal_weight: float = 0.99,
        validate_args=None,
    ):
        self.state = state
        self.concentration = concentration
        self.scale = scale
        self.mobility = mobility
        self.turn_std = turn_std
        self.turn_normal_weight = turn_normal_weight
        batch_shape = jnp.shape(state)[:-1]
        super().__init__(
            batch_shape=batch_shape, event_shape=(3,), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        k1, k2 = jax.random.split(key)
        d = step_length_sample(
            k1,
            sample_shape=sample_shape,
            concentration=self.concentration,
            scale=self.scale,
            mobility=self.mobility,
        )
        delta_phi = turning_angle_sample(
            k2,
            sample_shape=sample_shape,
            std=self.turn_std,
            normal_weight=self.turn_normal_weight,
        )
        x_old = self.state[..., 0]
        y_old = self.state[..., 1]
        phi_old = self.state[..., 2]
        phi_new = phi_old + delta_phi
        x_new = x_old + d * jnp.cos(phi_new)
        y_new = y_old + d * jnp.sin(phi_new)
        return jnp.stack([x_new, y_new, phi_new], axis=-1)

    def log_prob(self, value):
        x_old = self.state[..., 0]
        y_old = self.state[..., 1]
        phi_old = self.state[..., 2]
        x_new = value[..., 0]
        y_new = value[..., 1]
        phi_new = value[..., 2]

        dx = x_new - x_old
        dy = y_new - y_old
        d = jnp.sqrt(dx**2 + dy**2)
        delta_phi = wrap_angle(phi_new - phi_old)

        log_p_d = step_length_log_prob(
            d,
            concentration=self.concentration,
            scale=self.scale,
            mobility=self.mobility,
        )
        log_p_phi = turning_angle_log_prob(
            delta_phi,
            std=self.turn_std,
            normal_weight=self.turn_normal_weight,
        )
        # Jacobian: the 2x2 block of d(dx,dy)/d(d,delta_phi) has det d,
        # so the density on (x_new,y_new,phi_new) acquires a factor of 1/d.
        log_jacobian = -jnp.log(jnp.maximum(d, 1e-10))
        return log_p_d + log_p_phi + log_jacobian


class CorrelatedRandomWalkTransition(DiscreteTimeStateEvolution):
    """Discrete-time correlated random walk state evolution.

    Returns a `CorrelatedRandomWalkDistribution` for the next state given
    the current state.  Time arguments are accepted but unused: the step
    length and turning angle parameters are constant across time steps.
    """

    concentration: float
    scale: float
    mobility: float
    turn_std: float
    turn_normal_weight: float

    def __init__(
        self,
        concentration: float = 3.25,
        scale: float = 25.0,
        mobility: float = 216.0,
        turn_std: float = 0.4,
        turn_normal_weight: float = 0.99,
    ):
        self.concentration = concentration
        self.scale = scale
        self.mobility = mobility
        self.turn_std = turn_std
        self.turn_normal_weight = turn_normal_weight

    def __call__(self, x, u, t_now, t_next):
        return CorrelatedRandomWalkDistribution(
            state=x,
            concentration=self.concentration,
            scale=self.scale,
            mobility=self.mobility,
            turn_std=self.turn_std,
            turn_normal_weight=self.turn_normal_weight,
        )


class TelemetryObservationModel(ObservationModel):
    """Acoustic telemetry observation model.

    At each time step the observation is a binary vector y in {0,1}^K where K
    is the number of receivers.  Each element is an independent Bernoulli with
    detection probability given by a logistic function of the distance from
    the animal to that receiver:

        p(s, r_k) = sigmoid(logit_intercept - logit_slope * |s - r_k|)
                    * 1[|s - r_k| < max_range]

    Parameter values are estimated from Lake Ontario range tests (Lavender et al.).
    Note: the sign convention uses a positive intercept and positive slope so
    that probability decreases with distance.  Verify against the paper PDF if
    applying to real data.
    """

    receiver_locations: Float[Array, "K 2"]
    logit_intercept: float
    logit_slope: float
    max_range: float

    def __init__(
        self,
        receiver_locations: Float[Array, "K 2"],
        logit_intercept: float = 0.9039,
        logit_slope: float = 0.0021,
        max_range: float = 7000.0,
    ):
        self.receiver_locations = receiver_locations
        self.logit_intercept = logit_intercept
        self.logit_slope = logit_slope
        self.max_range = max_range

    def __call__(self, x, u, t):
        pos = x[..., :2]
        dists = jnp.sqrt(
            jnp.sum((pos[..., None, :] - self.receiver_locations) ** 2, axis=-1)
        )
        probs = jax.nn.sigmoid(
            self.logit_intercept - self.logit_slope * dists
        )
        in_range = dists < self.max_range
        probs = probs * in_range
        return dist.Independent(dist.Bernoulli(probs=probs), 1)


def animal_movement_model(
    receiver_locations: Float[Array, "K 2"],
    study_area_bounds: Float[Array, "4"] | None = None,
    concentration: float = 3.25,
    scale: float = 25.0,
    mobility: float = 216.0,
    turn_std: float = 0.4,
    turn_normal_weight: float = 0.99,
    logit_intercept: float = 0.9039,
    logit_slope: float = 0.0021,
    max_range: float = 7000.0,
) -> DynamicalModel:
    """Build the animal movement DynamicalModel.

    Args:
        receiver_locations: (K, 2) array of receiver (x, y) positions in meters.
        study_area_bounds: (4,) array [x_min, y_min, x_max, y_max] in meters.
            Defines the uniform initial distribution over positions.
            Defaults to a 10 km x 10 km square centered at the origin.
        concentration: Gamma concentration (shape) for step lengths.
        scale: Gamma scale for step lengths (mean = concentration * scale meters).
        mobility: Upper truncation for step lengths in meters.
        turn_std: Standard deviation (radians) for the TruncatedNormal turning angle component.
        turn_normal_weight: Mixture weight for the TruncatedNormal component.
        logit_intercept: Intercept in the logistic detection probability.
        logit_slope: Distance coefficient (positive) in the logistic detection probability.
        max_range: Maximum detection range in meters; receivers beyond this distance
            have zero detection probability.

    Returns:
        A DynamicalModel with state dimension 3 (x, y, phi) and observation
        dimension K (one binary detection per receiver).
    """
    if study_area_bounds is None:
        study_area_bounds = jnp.array([0.0, 0.0, 10000.0, 10000.0])

    x_min, y_min, x_max, y_max = (
        study_area_bounds[0],
        study_area_bounds[1],
        study_area_bounds[2],
        study_area_bounds[3],
    )

    initial_condition = dist.Independent(
        dist.Uniform(
            low=jnp.array([x_min, y_min, -jnp.pi]),
            high=jnp.array([x_max, y_max, jnp.pi]),
        ),
        reinterpreted_batch_ndims=1,
    )

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=CorrelatedRandomWalkTransition(
            concentration=concentration,
            scale=scale,
            mobility=mobility,
            turn_std=turn_std,
            turn_normal_weight=turn_normal_weight,
        ),
        observation_model=TelemetryObservationModel(
            receiver_locations=receiver_locations,
            logit_intercept=logit_intercept,
            logit_slope=logit_slope,
            max_range=max_range,
        ),
    )
