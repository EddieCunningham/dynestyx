"""Periodic cubic B-spline seasonal basis in JAX.

The cholera transmission rate in Ionides, Breto and King (2006) is forced by a
log-linear combination of a periodic B-spline basis,

    log beta(t) = sum_k b_k s_k(t),

where the s_k form a periodic, nonnegative partition of unity over the annual
period. This module builds that basis with `nbasis` cubic B-splines.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def _cardinal_bspline(u: Float[Array, "..."], degree: int) -> Float[Array, "..."]:
    """Cardinal B-spline of `degree` on the uniform knots 0, 1, 2, ...

    Supported on [0, degree + 1). The Cox-de Boor recursion needs the previous
    degree at u, u - 1, u - 2, ..., so the routine carries the whole stack of
    shifted values and contracts it one level at a time.
    """
    # Stack of degree-0 indicators evaluated at u, u-1, ..., u-degree.
    shifts = jnp.arange(degree + 1)
    v = u[..., None] - shifts  # (..., degree + 1)
    b = jnp.where((v >= 0.0) & (v < 1.0), 1.0, 0.0)

    for d in range(1, degree + 1):
        # b currently holds N_{d-1}(u), N_{d-1}(u-1), ..., one per remaining shift.
        w = u[..., None] - jnp.arange(b.shape[-1] - 1)  # arguments for the new level
        lower = b[..., :-1]  # N_{d-1}(w)
        upper = b[..., 1:]  # N_{d-1}(w - 1)
        b = (w / d) * lower + ((d + 1.0 - w) / d) * upper

    return b[..., 0]


def periodic_bspline_basis(
    t: Float[Array, "..."],
    nbasis: int = 6,
    degree: int = 3,
    period: float = 1.0,
) -> Float[Array, "... nbasis"]:
    """Evaluate `nbasis` periodic B-splines of `degree` at times `t`.

    The knots are uniformly spaced by `period / nbasis`. Each basis function is
    a cardinal B-spline wrapped to the period, so the family is periodic,
    nonnegative, and sums to one at every `t`.

    Returns an array whose trailing axis indexes the basis functions.
    """
    t = jnp.asarray(t)
    dx = period / nbasis
    u = jnp.mod(t, period) / dx  # knot-index coordinate within one period
    n_images = jnp.arange(-1, 2)  # wrapped support that runs off either end

    def basis_j(j):
        shifted = u[..., None] - j - n_images * nbasis  # (..., n_images)
        vals = _cardinal_bspline(shifted, degree)
        return vals.sum(axis=-1)

    cols = [basis_j(j) for j in range(nbasis)]
    return jnp.stack(cols, axis=-1)
