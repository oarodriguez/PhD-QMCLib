import math
from functools import partial

import mpmath as mp
from scipy.optimize import brentq


def energy_relation(lattice_depth: float,
                    lattice_ratio: float,
                    energy: float,
                    momentum: float,
                    ctx: object = math) -> float:
    """Evaluates the equation that relates the energy of the ideal Bose
    gas and the momentum of the bosons.

    :param lattice_depth: The potential magnitude.
    :param lattice_ratio: The ratio width/separation of two consecutive
                            barriers of th e potential.
    :param energy: The energy of the bosons.
    :param momentum: The momentum of the bosons.
    :param ctx: The context where the mathematical functions that determine
                the relation live in. By default it corresponds to the
                ``math`` module.
    :return: The value of the equation for the given parameters.
    """
    v0 = lattice_depth
    r = lattice_ratio
    ez = energy
    ks = momentum

    # Shortcuts.
    sin = ctx.sin
    cos = ctx.cos
    sinh = ctx.sinh
    cosh = ctx.cosh
    sqrt = ctx.sqrt

    # @formatter:off
    if ez == 0:
        return (
            1 / (2 * (1 + r)) * sqrt(v0) * sinh(r / (1 + r) * sqrt(v0)) +
            cosh(r / (1 + r) * sqrt(v0)) - cos(ks)
        )
    if ez == v0:
        return (
            -r * sqrt(v0) / (2 * (1 + r)) * sin(sqrt(v0) / (1 + r)) +
            cos(sqrt(v0) / (1 + r)) - cos(ks)
        )
    return (
        (v0 - 2 * ez) / (2 * sqrt(ez * (v0 - ez))) * sinh(
            r / (1 + r) * sqrt(v0 - ez)) * sin(sqrt(ez) / (1 + r)) +
        cosh(r / (1 + r) * sqrt(v0 - ez)) * cos(sqrt(ez) / (1 + r)) - cos(ks)
    )
    # @formatter:on


def eigen_energy(lattice_depth, lattice_ratio):
    """Calculates the ground state energy per particle of an ideal Bose gas
    within a multi-rods structure modeled through a Kronig-Penney potential.

    :param lattice_depth: The magnitude of the external potential.
    :param lattice_ratio: The relation width/separation of the potential
                            barriers.
    :return: The ground state energy per boson of the system.
    """
    v0 = lattice_depth
    r = lattice_ratio

    try:
        # First find a root with machine precision.
        func = partial(energy_relation, v0, r, momentum=0)
        root = brentq(func, 0, min(v0, (1 + r) ** 2 * math.pi ** 2))

        # Use arbitrary precision.
        mp_solver = partial(mp.findroot, verify=False)

    except OverflowError:
        # Use an arbitrary precision, root-bracketing method.
        root = (0, min(v0, (1 + r) ** 2 * mp.pi ** 2))
        mp_solver = partial(mp.findroot, solver='illinois', verify=False)

    func = partial(energy_relation, v0, r, momentum=0, ctx=mp)
    root = mp_solver(func, root)

    return mp.chop(root)
