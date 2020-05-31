from math import copysign, fabs

from numba import jit
from numpy import random as random

__all__ = [
    'min_distance',
    'numba_seed',
    'recast_to_supercell',
    'sign'
]


@jit(nopython=True)
def numba_seed(seed):
    """Seeds the numba RNG

    :param seed:
    :return:
    """
    random.seed(seed)


@jit(nopython=True)
def sign(v):
    """Retrieves the sign of a floating point number.

    :param v:
    :return:
    """
    return copysign(1., v)


@jit(nopython=True)
def min_distance(z_i, z_j, sc_size):
    """Calculates the minimum distance between the particle at
    ``z_i`` and all of the images of the particle at ``z_j``,
    including this. The minimum distance is always less than
    half of the size of the simulation supercell ``sc_size``.

    :param z_i:
    :param z_j:
    :param sc_size:
    :return:
    """
    sc_half = 0.5 * sc_size
    z_ij = z_i - z_j
    if fabs(z_ij) > sc_half:
        # Take the image.
        return -sc_half + (z_ij + sc_half) % sc_size
    return z_ij


@jit(nopython=True)
def recast_to_supercell(z, z_min, z_max):
    """Gets the position of the particle at ``z`` within the simulation
    supercell with boundaries ``z_min`` y ``z_max``. If the particle is
    outside the supercell, it returns the position of its closest image.

    :param z:
    :param z_min:
    :param z_max:
    :return:
    """
    sc_size = (z_max - z_min)
    return z_min + (z - z_min) % sc_size
