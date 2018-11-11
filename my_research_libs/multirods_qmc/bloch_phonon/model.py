import operator
from abc import ABCMeta
from enum import unique
from functools import reduce
from math import atan, cos, cosh, fabs, pi, sin, sinh, sqrt, tan, tanh
from typing import NamedTuple

import numpy as np
from attr import attrs
from cached_property import cached_property
from numba import jit
from numpy import random
from scipy.optimize import brentq

from my_research_libs import ideal, qmc_base
from my_research_libs.qmc_base.utils import min_distance
from my_research_libs.utils import get_random_rng_seed

__all__ = [
    'ArrayGUFunc',
    'ArrayGUPureFunc',
    'CoreFuncs',
    'EnergyGUFunc',
    'Spec',
    'ModelParams',
    'ModelVarParams',
    'Sampling',
    'UniformSampling',
    'ScalarGUFunc',
    'ScalarGUPureFunc',
    'WFGUFunc',
    'core_funcs'
]

# Some alias..
DIST_RAND = qmc_base.jastrow.SysConfDistType.RANDOM
DIST_REGULAR = qmc_base.jastrow.SysConfDistType.REGULAR


@unique
class ParamName(qmc_base.model.ParamNameEnum):
    """Enumerates the parameters of the model (Bijl-Jastrow type) of a
    quantum system in a Multi-Rods periodic structure with repulsive,
    contact interactions.
    """
    LATTICE_DEPTH = 'lattice_depth'
    LATTICE_RATIO = 'lattice_ratio'
    INTERACTION_STRENGTH = 'interaction_strength'
    BOSON_NUMBER = 'boson_number'
    SUPERCELL_SIZE = 'supercell_size'


class ModelParams(qmc_base.ParamsSet):
    """Represents the parameters of the model."""
    names = ParamName


@unique
class VarParamName(qmc_base.ParamNameEnum):
    """Enumerates the variational parameters of the wave function of the
    model (Bijl-Jastrow type) of a quantum system in a Multi-Rods periodic
    structure with repulsive, contact interactions.
    """
    TBF_CONTACT_CUTOFF = 'tbf_contact_cutoff'


class ModelVarParams(qmc_base.ParamsSet):
    """Represents the variational parameters of the model"""
    names = VarParamName


class SpecNT(NamedTuple):
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


class OBFSpecNT(NamedTuple):
    """One-body function parameters."""
    lattice_depth: float
    lattice_ratio: float
    well_width: float
    barrier_width: float
    param_e0: float
    param_k1: float
    param_kp1: float


class TBFSpecNT(NamedTuple):
    """Two-body function parameters."""
    supercell_size: float
    tbf_contact_cutoff: float
    param_k2: float
    param_beta: float
    param_r_off: float
    param_am: float


class CFCSpecNT(NamedTuple):
    """"""
    # Does nothing, only for type hints
    model_spec: SpecNT
    obf_spec: OBFSpecNT
    tbf_spec: TBFSpecNT


# NOTE: slots=True avoids adding more attributes
@attrs(auto_attribs=True, init=False, slots=True)
class Spec(qmc_base.jastrow.Spec):
    """The parameters of the Bloch-Phonon QMC model.

    Defines the parameters and related properties of a quantum system in a
    Multi-Rods periodic structure with repulsive, contact interactions,
    with a trial wave function of the Bijl-Jastrow type.
    """
    #
    lattice_depth: float
    lattice_ratio: float
    interaction_strength: float
    boson_number: int
    supercell_size: float
    tbf_contact_cutoff: float

    def __init__(self, lattice_depth: float,
                 lattice_ratio: float,
                 interaction_strength: float,
                 boson_number: float,
                 supercell_size: float,
                 tbf_contact_cutoff: float):
        """

        :param lattice_depth:
        :param lattice_ratio:
        :param interaction_strength:
        :param boson_number:
        :param supercell_size:
        :param tbf_contact_cutoff:
        """
        self.lattice_depth = lattice_depth
        self.lattice_ratio = lattice_ratio
        self.interaction_strength = interaction_strength
        self.boson_number = boson_number
        self.supercell_size = supercell_size
        self.tbf_contact_cutoff = tbf_contact_cutoff

    @property
    def boundaries(self):
        """"""
        sc_size = self.supercell_size
        return 0., 1. * sc_size

    @property
    def well_width(self):
        """"""
        r = self.lattice_ratio
        return 1 / (1 + r)

    @property
    def barrier_width(self):
        """"""
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

    @property
    def gufunc_args(self):
        """Concatenate the :attr:`Spec.cfc_spec_nt` tuples and returns
        a single tuple. Intended to be used with generalized universal
        functions (gufunc).
        """
        return reduce(operator.add, self.cfc_spec_nt)

    @property
    def core_funcs(self):
        """"""
        # TODO: Remove this method...
        return CoreFuncs()


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


class Sampling(qmc_base.jastrow.vmc.PBCSampling):
    """Sampling of the probability density of the Bloch-Phonon model with
    a multi-rods external potential.
    """

    @property
    def gen_args(self):
        """The arguments of the sampling generator function.

        :return:
        """
        names = self.params_cls.names
        rng_seed = self.params[names.RNG_SEED]
        if rng_seed is None:
            rng_seed = get_random_rng_seed()

        return (
            self.model_spec.cfc_spec_nt,
            self.ppf_args,
            self.params[names.INI_SYS_CONF],
            self.params[names.CHAIN_SAMPLES],
            self.params[names.BURN_IN_SAMPLES],
            rng_seed
        )


class UniformSampling(Sampling, qmc_base.jastrow.vmc.PBCUniformSampling):
    """"""
    pass


@jit(nopython=True, cache=True)
def _as_model_args(model_full_params):
    """Takes the model parameters from the array, and group them
    in tuples. The constructed tuples are returned to the caller so
    they an be user as arguments for a ``CoreFuncs`` function.
    """
    # TODO: Is there a better way to do this?
    v0 = model_full_params[0]
    r = model_full_params[1]
    gn = model_full_params[2]
    nop = model_full_params[3]
    sc_size = model_full_params[4]
    model_params = v0, r, gn, nop, sc_size

    #
    v0 = model_full_params[5]
    r = model_full_params[6]
    e0 = model_full_params[7]
    k1 = model_full_params[8]
    kp1 = model_full_params[9]
    obf_params = v0, r, e0, k1, kp1

    #
    rm = model_full_params[10]
    sc_size = model_full_params[11]
    k2 = model_full_params[12]
    beta = model_full_params[13]
    r_off = model_full_params[14]
    am = model_full_params[15]
    tbf_params = rm, sc_size, k2, beta, r_off, am

    # NOTICE: This way to access data may generate corrupted results.
    # v0, r, gn, nop, sc_size = model_full_params[0:5]
    # v0, r, e0, k1, kp1 = model_full_params[5:10]
    # rm, scs, k2, beta, r_off, am = model_full_params[10:16]
    #
    return model_params, obf_params, tbf_params


class ArrayGUFunc(qmc_base.jastrow.ArrayGUFunc, metaclass=ABCMeta):
    """"""

    signatures = ['void(f8[:,:],f8[:],f8[:],f8[:,:])']
    layout = '(ns,nop),(p1),(p2)->(ns,nop)'

    @cached_property
    def as_model_args(self):
        """Takes the model parameters from an array."""
        return _as_model_args

    def __init__(self, base_func, target=None):
        """

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)


class ScalarGUFunc(ArrayGUFunc, qmc_base.jastrow.ScalarGUFunc,
                   metaclass=ABCMeta):
    """"""
    signatures = ['void(f8[:,:],f8[:],f8[:],f8[:])']
    layout = '(ns,nop),(p1),(p2)->()'

    def __init__(self, base_func, target=None):
        """"""
        super().__init__(base_func, target)


class ArrayGUPureFunc(qmc_base.jastrow.ArrayGUPureFunc, metaclass=ABCMeta):
    """"""

    signatures = ['void(f8[:,:],f8[:],f8[:,:])']
    layout = '(ns,nop),(p1)->(ns,nop)'

    @cached_property
    def as_model_args(self):
        """"""
        return _as_model_args

    def __init__(self, base_func, target=None):
        """

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)


class ScalarGUPureFunc(ArrayGUPureFunc, qmc_base.jastrow.ScalarGUPureFunc,
                       metaclass=ABCMeta):
    """"""
    signatures = ['void(f8[:,:],f8[:],f8[:])']
    layout = '(ns,nop),(p1)->()'

    def __init__(self, base_func, target=None):
        """"""
        super().__init__(base_func, target)


class WFGUFunc(ScalarGUPureFunc):
    """Generalized version of the wave function."""
    pass


class EnergyGUFunc(ScalarGUFunc):
    """"""

    @property
    def as_func_args(self):
        """"""

        @jit(nopython=True, cache=True)
        def _as_func_args(func_params):
            """"""
            v0_ = func_params[0]
            r_ = func_params[1]
            gn_ = func_params[2]
            func_args_0 = v0_, r_, gn_

            return func_args_0,

        return _as_func_args


# Global object with most general model core functions.
core_funcs = CoreFuncs()
