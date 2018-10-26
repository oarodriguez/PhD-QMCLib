from math import atan, cos, cosh, fabs, pi, sin, sinh, sqrt, tan, tanh
from typing import Tuple

from numba import jit
from scipy.optimize import brentq


@jit(nopython=True, cache=True)
def gs_one_body_func(z, v0, r, e0, k1, kp1):
    """"""
    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    if z_a < z_cell:
        # Zero potential region.
        return cosh(kp1 * (z_cell - 1. + 0.5 * z_b))
    else:
        # Region where the potential is zero.
        cf = sqrt(1 + v0 / e0 * sinh(0.5 * sqrt(v0 - e0) * z_b) ** 2.0)
        return cf * cos(k1 * (z_cell - 0.5 * z_a))


@jit(nopython=True, cache=True)
def gs_one_body_func_log_dz(z, v0, r, e0, k1, kp1):
    """"""
    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    if z_a < z_cell:
        # Region with nonzero potential.
        return kp1 * tanh(kp1 * (z_cell - 1. + 0.5 * z_b))
    else:
        # Region where the potential is zero.
        return -k1 * tan(k1 * (z_cell - 0.5 * z_a))


@jit(nopython=True, cache=True)
def gs_one_body_func_log_dz2(z, v0, r, e0, k1, kp1):
    """"""
    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    return v0 - e0 if z_a < z_cell else -e0


@jit(nopython=True, cache=True)
def unpack_gs_obf_params(params_array):
    """"""
    # Seems unnecessary, but we need to return a tuple if we want to
    # unpack the tuples in a function call.
    v0, r, e0, k1, kp1 = params_array
    return v0, r, e0, k1, kp1


@jit(nopython=True, cache=True)
def phonon_two_body_func(rz, rm, scs, k2, beta, r_off, am):
    """"""
    # Two-body term.
    if rz < fabs(rm):
        return am * cos(k2 * (rz - r_off))
    else:
        return sin(pi * rz / scs) ** beta


@jit(nopython=True, cache=True)
def phonon_two_body_func_log_dz(rz, rm, scs, k2, beta, r_off, am):
    """"""
    # Two-body term.
    if rz < fabs(rm):
        return -k2 * tan(k2 * (rz - r_off))
    else:
        return (pi / scs) * beta / (tan(pi * rz / scs))


@jit(nopython=True, cache=True)
def phonon_two_body_func_log_dz2(rz, rm, scs, k2, beta, r_off, am):
    """"""
    # Two-body term.
    if rz < fabs(rm):
        return -k2 * k2
    else:
        return (pi / scs) ** 2 * beta * (
                (beta - 1) / (tan(pi * rz / scs) ** 2) - 1
        )


@jit(nopython=True, cache=True)
def unpack_tbf_params(params_array):
    """"""
    # Seems unnecessary, but we need to return a tuple if we want to
    # unpack the tuples in a function call.
    rm, scs, k2, beta, r_off, am = params_array
    return rm, scs, k2, beta, r_off, am


def two_body_func_match_params(gn: float,
                               nop: int,
                               rm: float,
                               scs: float) -> (
        Tuple[float, float, float, float]):
    """Calculate the unknown constants that join the two pieces of the
    two-body functions of the Jastrow trial function at the point `zm_var`.
    The parameters are a function of the boson interaction magnitude `g`
    and the average linear density `boson_number` of the system.

    :param gn: The magnitude of the interaction
                                 between bosons.
    :param nop: The density of bosons in the simulation box.
    :param rm: The point where both the pieces of the
                           function must be joined.
    :param scs:
    :return: The two body momentum that match the functions.
    """
    if not fabs(rm) <= fabs(scs / 2):
        raise ValueError("parameter value 'rm' out of domain")

    if gn == 0:
        return 0., 0., 1 / 2 * scs, 1.

    # Convert interaction energy to Lieb gamma.
    lgm = 0.5 * (scs / nop) ** 2 * gn

    # Following equations require rm in simulation box units.
    rm /= scs

    def _nonlinear_equation(k2rm, *args):
        a1d = args[0]
        beta_rm = tan(pi * rm) / pi if k2rm == 0 else (
                k2rm / pi * (rm - k2rm * a1d * tan(k2rm)) * tan(pi * rm) /
                (k2rm * a1d + rm * tan(k2rm))
        )

        # Equality of the local energy at `rm`.
        fn2d_rm_eq = (
                (k2rm * sin(pi * rm)) ** 2 +
                (pi * beta_rm * cos(pi * rm)) ** 2 -
                pi ** 2 * beta_rm * rm
        )

        return fn2d_rm_eq

    # The one-dimensional scattering length.
    # ❗ NOTICE ❗: Here has to appear a two factor in order to be
    # consistent with the Lieb-Liniger theory.
    a1d = 2.0 / (lgm * nop)

    k2rm = brentq(_nonlinear_equation, 0, pi / 2, args=(a1d,))

    beta_rm = (
            k2rm / pi * (rm - k2rm * a1d * tan(k2rm)) * tan(pi * rm) /
            (k2rm * a1d + rm * tan(k2rm))
    )

    k2 = k2rm / rm
    k2r_off = atan(1 / (k2 * a1d))

    beta = beta_rm / rm
    r_off = k2r_off / k2
    am = sin(pi * rm) ** beta / cos(k2rm - k2r_off)

    # The coefficient `am` is fixed by the rest of the parameters.
    # am = sin(pi * rm) ** beta / cos(k2 * (rm - r_off))
    # Return momentum and length in units of lattice period.
    # return k2, beta, r_off, am
    return k2 / scs, beta, r_off * scs, am
