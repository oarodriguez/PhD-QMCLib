from math import atan, cos, cosh, fabs, pi, sin, sinh, sqrt, tan, tanh
from typing import NamedTuple

import attr
import numpy as np
from cached_property import cached_property
from numba import jit
from numpy import random
from scipy.optimize import brentq

from my_research_libs import ideal, qmc_base
from my_research_libs.qmc_base.utils import min_distance

__all__ = [
    'CoreFuncs',
    'Spec',
    'core_funcs'
]

# Some alias..
DIST_RAND = qmc_base.jastrow.SysConfDistType.RANDOM
DIST_REGULAR = qmc_base.jastrow.SysConfDistType.REGULAR


class SpecNT(qmc_base.jastrow.SpecNT, NamedTuple):
    """The model `Spec` as a named tuple."""
    lattice_depth: float
    lattice_ratio: float
    interaction_strength: float
    boson_number: int
    supercell_size: float
    tbf_contact_cutoff: float
    well_width: float
    barrier_width: float
    is_free: bool
    is_ideal: bool


class OBFSpecNT(qmc_base.jastrow.OBFSpecNT, NamedTuple):
    """One-body function parameters."""
    lattice_depth: float
    lattice_ratio: float
    well_width: float
    barrier_width: float
    param_e0: float
    param_k1: float
    param_kp1: float


class TBFSpecNT(qmc_base.jastrow.OBFSpecNT, NamedTuple):
    """Two-body function parameters."""
    supercell_size: float
    tbf_contact_cutoff: float
    param_k2: float
    param_beta: float
    param_r_off: float
    param_am: float


class CFCSpecNT(qmc_base.jastrow.CFCSpecNT, NamedTuple):
    """"""
    # Does nothing, only for type hints
    model_spec: SpecNT
    obf_spec: OBFSpecNT
    tbf_spec: TBFSpecNT


# NOTE: slots=True avoids adding more attributes
# NOTE: Use repr=False if we want instances that can be serialized (pickle)
@attr.s(auto_attribs=True, frozen=True)
class Spec(qmc_base.jastrow.Spec):
    """The parameters of the Bloch-Phonon QMC model.

    Defines the parameters and related properties of a quantum system in a
    Multi-Rods periodic structure with repulsive, contact interactions,
    with a trial wave function of the Bijl-Jastrow type.
    """
    #: The lattice depth of the potential.
    lattice_depth: float

    #: The ratio of the barriers width between the wells width.
    lattice_ratio: float

    #: The magnitude of the interaction strength between two bosons.
    interaction_strength: float

    #: The number of bosons.
    boson_number: int

    #: The size of the QMC simulation box.
    supercell_size: float

    # TODO: We need a better documentation for this attribute.
    #: The variational parameter of the two-body functions.
    tbf_contact_cutoff: float

    #: Functions to calculate the main physical properties of a model.
    phys_funcs: 'PhysicalFuncs' = attr.ib(init=False, cmp=False, repr=False)

    # TODO: Implement improved __init__.

    def __attrs_post_init__(self):
        """"""
        # NOTE: Should we use a new CoreFuncs instance?
        physical_funcs = PhysicalFuncs(self)
        super().__setattr__('phys_funcs', physical_funcs)

    @property
    def boundaries(self):
        """The boundaries of the QMC simulation box."""
        sc_size = self.supercell_size
        return 0., 1. * sc_size

    @property
    def well_width(self):
        """The width of the lattice wells."""
        r = self.lattice_ratio
        return 1 / (1 + r)

    @property
    def barrier_width(self):
        """The width of the lattice barriers."""
        r = self.lattice_ratio
        return r / (1 + r)

    @property
    def is_free(self):
        """"""
        v0 = self.lattice_depth
        r = self.lattice_ratio
        if v0 <= 1e-10:
            return True
        elif r <= 1e-10:
            return True
        else:
            return False

    @property
    def is_ideal(self):
        """"""
        gn = self.interaction_strength
        if gn <= 1e-10:
            return True
        else:
            return False

    @property
    def sys_conf_shape(self):
        """The shape of the array/buffer that stores the configuration
        of the particles (positions, velocities, etc.)
        """
        # NOTE: Should we allocate space for the DRIFT_SLOT?
        # TODO: Fix NUM_SLOTS if DRIFT_SLOT becomes really unnecessary.
        nop = self.boson_number
        return len(self.sys_conf_slots), nop

    def init_get_sys_conf(self, dist_type=DIST_RAND, offset=None):
        """Creates and initializes a system configuration with the
        positions of the particles arranged in the order specified
        by ``dist_type`` argument.

        :param dist_type:
        :param offset:
        :return:
        """
        nop = self.boson_number
        sc_size = self.supercell_size
        z_min, z_max = self.boundaries
        sys_conf_slots = qmc_base.jastrow.SysConfSlot
        sys_conf = self.get_sys_conf_buffer()
        pos_slot = sys_conf_slots.pos
        offset = offset or 0.

        if dist_type is DIST_RAND:
            spread = sc_size * random.random_sample(nop)
        elif dist_type is DIST_REGULAR:
            spread = np.linspace(0, sc_size, nop, endpoint=False)
        else:
            raise ValueError("unrecognized '{}' dist_type".format(dist_type))

        sys_conf[pos_slot, :] = z_min + (offset + spread) % sc_size
        return sys_conf

    @property
    def as_nt(self):
        """"""
        # NOTE: Keep the order in which the attributes were defined.
        return SpecNT(self.lattice_depth,
                      self.lattice_ratio,
                      self.interaction_strength,
                      self.boson_number,
                      self.supercell_size,
                      self.tbf_contact_cutoff,
                      self.well_width,
                      self.barrier_width,
                      self.is_free,
                      self.is_ideal)

    @property
    def obf_spec_nt(self):
        """

        :return:
        """
        v0 = self.lattice_depth
        r = self.lattice_ratio
        e0 = float(ideal.eigen_energy(v0, r))
        k1, kp1 = sqrt(e0), sqrt(v0 - e0)

        obf_spec_nt = OBFSpecNT(self.lattice_depth,
                                self.lattice_ratio,
                                self.well_width,
                                self.barrier_width,
                                param_e0=e0,
                                param_k1=k1,
                                param_kp1=kp1)
        return obf_spec_nt

    @property
    def tbf_spec_nt(self):
        """

        :return:
        """
        gn = self.interaction_strength
        nop = self.boson_number
        sc_size = self.supercell_size
        rm = self.tbf_contact_cutoff

        if not fabs(rm) <= fabs(sc_size / 2):
            raise ValueError("parameter value 'rm' out of domain")

        if gn == 0:
            tbf_spec_nt = TBFSpecNT(self.supercell_size,
                                    self.tbf_contact_cutoff,
                                    param_k2=0.,
                                    param_beta=0.,
                                    param_r_off=1 / 2 * sc_size,
                                    param_am=1.0)
            return tbf_spec_nt

        # Convert interaction energy to Lieb gamma.
        lgm = 0.5 * (sc_size / nop) ** 2 * gn

        # Following equations require rm in simulation box units.
        rm /= sc_size

        # noinspection PyShadowingNames
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
        tbf_spec_nt = TBFSpecNT(self.supercell_size,
                                self.tbf_contact_cutoff,
                                param_k2=k2 / sc_size,
                                param_beta=beta,
                                param_r_off=r_off * sc_size,
                                param_am=am)
        return tbf_spec_nt

    @property
    def cfc_spec_nt(self):
        """"""
        self_spec = self.as_nt
        obf_spec = self.obf_spec_nt
        tbf_spec = self.tbf_spec_nt
        return CFCSpecNT(self_spec, obf_spec, tbf_spec)


@jit(nopython=True)
def _one_body_func(z: float, spec: OBFSpecNT) -> float:
    """One-body function.

    :param z:
    :param spec:
    :return:
    """
    v0 = spec.lattice_depth
    r = spec.lattice_ratio
    e0 = spec.param_e0
    k1 = spec.param_k1
    kp1 = spec.param_kp1

    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    if z_a < z_cell:
        # Zero potential region.
        return cosh(kp1 * (z_cell - 1. + 0.5 * z_b))
    else:
        # Region where the potential is zero.
        cf = sqrt(1 + v0 / e0 * sinh(0.5 * sqrt(v0 - e0) * z_b) ** 2.0)
        return cf * cos(k1 * (z_cell - 0.5 * z_a))


@jit(nopython=True)
def _one_body_func_log_dz(z: float, spec: OBFSpecNT) -> float:
    """One-body function logarithmic derivative.

    :param z:
    :param spec:
    :return:
    """
    r = spec.lattice_ratio
    k1 = spec.param_k1
    kp1 = spec.param_kp1

    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    if z_a < z_cell:
        # Region with nonzero potential.
        return kp1 * tanh(kp1 * (z_cell - 1. + 0.5 * z_b))
    else:
        # Region where the potential is zero.
        return -k1 * tan(k1 * (z_cell - 0.5 * z_a))


@jit(nopython=True)
def _one_body_func_log_dz2(z: float, spec: OBFSpecNT) -> float:
    """One-body function second logarithmic derivative.

    :param z:
    :param spec:
    :return:
    """
    v0 = spec.lattice_depth
    r = spec.lattice_ratio
    e0 = spec.param_e0

    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    return v0 - e0 if z_a < z_cell else -e0


@jit(nopython=True)
def _two_body_func(rz: float, spec: TBFSpecNT) -> float:
    """Two-body function.

    :param rz:
    :param spec:
    :return:
    """
    sc_size = spec.supercell_size
    rm = spec.tbf_contact_cutoff
    k2 = spec.param_k2
    beta = spec.param_beta
    r_off = spec.param_r_off
    am = spec.param_am

    # Two-body term.
    if rz < fabs(rm):
        return am * cos(k2 * (rz - r_off))
    else:
        return sin(pi * rz / sc_size) ** beta


@jit(nopython=True)
def _two_body_func_log_dz(rz: float, spec: TBFSpecNT) -> float:
    """Two-body function logarithmic derivative.

    :param rz:
    :param spec:
    :return:
    """
    sc_size = spec.supercell_size
    rm = spec.tbf_contact_cutoff
    k2 = spec.param_k2
    beta = spec.param_beta
    r_off = spec.param_r_off

    # Two-body term.
    if rz < fabs(rm):
        return -k2 * tan(k2 * (rz - r_off))
    else:
        return (pi / sc_size) * beta / (tan(pi * rz / sc_size))


@jit(nopython=True)
def _two_body_func_log_dz2(rz: float, spec: TBFSpecNT) -> float:
    """Two-body function logarithmic derivative.

    :param rz:
    :param spec:
    :return:
    """
    sc_size = spec.supercell_size
    rm = spec.tbf_contact_cutoff
    k2 = spec.param_k2
    beta = spec.param_beta

    # Two-body term.
    if rz < fabs(rm):
        return -k2 * k2
    else:
        return (pi / sc_size) ** 2 * beta * (
                (beta - 1) / (tan(pi * rz / sc_size) ** 2) - 1
        )


@jit(nopython=True)
def _potential(z: float, spec: SpecNT) -> float:
    """Calculates the potential energy of the Bose gas due to the
     external potential.

     :param z: The current configuration of the positions of the
               particles.
     :param spec:
     :return:
    """
    v0 = spec.lattice_depth
    z_a = spec.well_width
    z_cell = z % 1.
    return v0 if z_a < z_cell else 0.


@jit(nopython=True)
def _real_distance(z_i, z_j, model_spec: SpecNT):
    """The real distance between two bosons.

    This routine takes into account the periodic boundary
    conditions of the QMC calculation.
    """
    sc_size = model_spec.supercell_size
    return min_distance(z_i, z_j, sc_size)


class CoreFuncs(qmc_base.jastrow.CoreFuncs):
    """Functions of a QMC model for a system with a trial wave function
    of the Bijl-Jastrow type.
    """

    def __init__(self):
        """"""
        super().__init__()

    @cached_property
    def real_distance(self):
        """The real distance between two bosons."""
        return _real_distance

    @cached_property
    def one_body_func(self):
        """The one-body function definition."""
        return _one_body_func

    @cached_property
    def two_body_func(self):
        """The two-body function definition."""
        return _two_body_func

    @cached_property
    def one_body_func_log_dz(self):
        """One-body function logarithmic derivative."""
        return _one_body_func_log_dz

    @cached_property
    def two_body_func_log_dz(self):
        """Two-body function logarithmic derivative."""
        return _two_body_func_log_dz

    @cached_property
    def one_body_func_log_dz2(self):
        """One-body function second logarithmic derivative."""
        return _one_body_func_log_dz2

    @cached_property
    def two_body_func_log_dz2(self):
        """Two-body function second logarithmic derivative."""
        return _two_body_func_log_dz2

    @cached_property
    def potential(self):
        """The external potential."""
        return _potential


# Common reference to all the core functions.
#
# These functions are general: they accept the core func spec as an
# argument.
core_funcs = CoreFuncs()


@attr.s(auto_attribs=True, frozen=True)
class PhysicalFuncs(qmc_base.jastrow.PhysicalFuncs):
    """Functions to calculate the main physical properties of the model."""

    spec: Spec
    core_funcs: CoreFuncs = attr.ib(init=False, cmp=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        # NOTE: Should we use a new CoreFuncs instance?
        super().__setattr__('core_funcs', core_funcs)