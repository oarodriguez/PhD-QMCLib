import typing as t
from math import (atan, ceil, cos, cosh, fabs, pi, sin, sinh, sqrt, tan, tanh)
from os import cpu_count

import attr
import dask
import dask.bag as db
import numpy as np
from cached_property import cached_property
from numba import jit
from numpy import random
from scipy.optimize import brentq, differential_evolution

from phd_qmclib import ideal, qmc_base
from phd_qmclib.qmc_base.utils import min_distance
from phd_qmclib.util.attr import (
    int_converter, int_validator, opt_float_converter, opt_float_validator,
    opt_int_converter
)

__all__ = [
    'CFCSpec',
    'CoreFuncs',
    'CSWFOptimizer',
    'OBFParams',
    'Params',
    'PhysicalFuncs',
    'Spec',
    'TBFParams',
    'DIST_RAND',
    'DIST_REGULAR',
    'core_funcs'
]

# Some alias..
DIST_RAND = qmc_base.jastrow.SysConfDistType.RANDOM
DIST_REGULAR = qmc_base.jastrow.SysConfDistType.REGULAR


class Params(qmc_base.jastrow.Params, t.NamedTuple):
    """The model `Spec` as a named tuple."""
    # TODO: Add a more robust type system for the parameters.
    lattice_depth: float
    lattice_ratio: float
    interaction_strength: float
    boson_number: np.int64
    supercell_size: float
    tbf_contact_cutoff: float
    defect_magnitude: float
    defects_sep: int
    well_width: float
    barrier_width: float
    is_free: bool
    is_ideal: bool


class OBFParams(qmc_base.jastrow.OBFParams, t.NamedTuple):
    """One-body function parameters."""
    lattice_depth: float
    lattice_ratio: float
    well_width: float
    barrier_width: float
    param_e0: float
    param_k1: float
    param_kp1: float


class TBFParams(qmc_base.jastrow.TBFParams, t.NamedTuple):
    """Two-body function parameters."""
    supercell_size: float
    tbf_contact_cutoff: float
    param_k2: float
    param_beta: float
    param_r_off: float
    param_am: float


class CFCSpec(qmc_base.jastrow.CFCSpec, t.NamedTuple):
    """"""
    # Does nothing, only for type hints
    model_params: Params
    obf_params: OBFParams
    tbf_params: TBFParams


class CFCSpecAlt(qmc_base.jastrow.CFCSpecAlt, t.NamedTuple):
    """"""
    # Does nothing, only for type hints
    model_params: np.ndarray
    obf_params: np.ndarray
    tbf_params: np.ndarray


# noinspection PyUnusedLocal
def tbf_contact_cutoff_validator(model_inst: 'Spec',
                                 attribute: str,
                                 value: t.Any):
    """Validator for the ``tbf_contact_cutoff`` attribute.

    :param model_inst: The model instance.
    :param attribute: The validated attribute.
    :param value: The value of the attribute.
    :return:
    """
    sc_size = model_inst.supercell_size
    if not fabs(value) <= fabs(sc_size / 2):
        raise ValueError("parameter value 'rm' out of domain")


# noinspection PyUnusedLocal
def num_defects_validator(model_inst: 'Spec',
                          attribute: str,
                          value: t.Any):
    """Validator for the ``num_defects`` attribute.

    :param model_inst: The model instance.
    :param attribute: The validated attribute.
    :param value: The value of the attribute.
    :return:
    """
    num_defects = value
    if num_defects is not None:
        if num_defects < 0:
            raise ValueError("number of defects can't be negative")
        num_sites = int(ceil(model_inst.supercell_size))
        if num_defects and (num_sites % num_defects):
            raise ValueError(f"the specified number of defects "
                             f"({num_defects :d}) can't be evenly "
                             f"distributed in the lattice")


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
    lattice_depth: float = attr.ib(converter=float)

    #: The ratio of the barriers width between the wells width.
    lattice_ratio: float = attr.ib(converter=float)

    #: The magnitude of the interaction strength between two bosons.
    interaction_strength: float = attr.ib(converter=float)

    #: The number of bosons.
    boson_number: int = \
        attr.ib(converter=int_converter, validator=int_validator)

    #: The size of the QMC simulation box.
    supercell_size: float = attr.ib(converter=float)

    # TODO: We need a better documentation for this attribute.
    #: The variational parameter of the two-body functions.
    tbf_contact_cutoff: float = \
        attr.ib(converter=float, validator=tbf_contact_cutoff_validator)

    #: Number of defects, evenly spaced.
    num_defects: t.Optional[int] = \
        attr.ib(default=None, converter=opt_int_converter,
                validator=num_defects_validator)

    #: Magnitude for all the defects.
    defect_magnitude: t.Optional[float] = \
        attr.ib(default=None, converter=opt_float_converter,
                validator=opt_float_validator)

    # TODO: Implement improved __init__.
    def __attrs_post_init__(self):
        """Post initialization stage."""
        lattice_depth = self.lattice_depth
        num_defects = self.num_defects
        defect_magnitude = self.defect_magnitude
        if (defect_magnitude is None) and (num_defects is None):
            # Assume a regular lattice.
            object.__setattr__(self, 'defect_magnitude', lattice_depth)
            object.__setattr__(self, 'num_defects', 0)
        else:
            if num_defects is None:
                num_defects = 0
                defect_magnitude = lattice_depth
            else:
                # Assume as many full defects as num_defects
                defect_magnitude = \
                    defect_magnitude if num_defects else lattice_depth

            if defect_magnitude > lattice_depth:
                raise ValueError("Defect magnitude can't be greater than the "
                                 "lattice depth.")
            object.__setattr__(self, 'defect_magnitude', defect_magnitude)
            object.__setattr__(self, 'num_defects', num_defects)

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
        # noinspection PyTypeChecker
        return len(qmc_base.jastrow.SysConfSlot), nop

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
    def params(self):
        """"""
        num_sites = int(ceil(self.supercell_size))
        num_defects = self.num_defects
        defects_sep = 1 if not num_defects else int(num_sites // num_defects)

        # NOTE: Keep the order in which the attributes were defined.
        params = Params(self.lattice_depth,
                        self.lattice_ratio,
                        self.interaction_strength,
                        self.boson_number,
                        self.supercell_size,
                        self.tbf_contact_cutoff,
                        self.defect_magnitude,
                        defects_sep,
                        self.well_width,
                        self.barrier_width,
                        self.is_free,
                        self.is_ideal)
        return params

    @property
    def obf_params(self):
        """

        :return:
        """
        v0 = self.lattice_depth
        r = self.lattice_ratio
        e0 = float(ideal.eigen_energy(v0, r))
        k1, kp1 = sqrt(e0), sqrt(v0 - e0)

        obf_params = OBFParams(self.lattice_depth,
                               self.lattice_ratio,
                               self.well_width,
                               self.barrier_width,
                               param_e0=e0,
                               param_k1=k1,
                               param_kp1=kp1)
        return obf_params

    @property
    def tbf_params(self):
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
            tbf_params = TBFParams(self.supercell_size,
                                   self.tbf_contact_cutoff,
                                   param_k2=0.,
                                   param_beta=0.,
                                   param_r_off=1 / 2 * sc_size,
                                   param_am=1.0)
            return tbf_params

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

        # noinspection PyTypeChecker
        k2rm: float = brentq(_nonlinear_equation, 0, pi / 2, args=(a1d,))

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
        tbf_params = TBFParams(self.supercell_size,
                               self.tbf_contact_cutoff,
                               param_k2=k2 / sc_size,
                               param_beta=beta,
                               param_r_off=r_off * sc_size,
                               param_am=am)
        return tbf_params

    @property
    def cfc_spec(self):
        """"""
        return CFCSpec(self.params,
                       self.obf_params,
                       self.tbf_params)


@jit(nopython=True)
def _one_body_func(z: float, obf_params: OBFParams) -> float:
    """One-body function.

    :param z:
    :param obf_params:
    :return:
    """
    v0 = obf_params.lattice_depth
    r = obf_params.lattice_ratio
    e0 = obf_params.param_e0
    k1 = obf_params.param_k1
    kp1 = obf_params.param_kp1

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
def _one_body_func_log_dz(z: float, obf_params: OBFParams) -> float:
    """One-body function logarithmic derivative.

    :param z:
    :param obf_params:
    :return:
    """
    r = obf_params.lattice_ratio
    k1 = obf_params.param_k1
    kp1 = obf_params.param_kp1

    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    if z_a < z_cell:
        # Region with nonzero potential.
        return kp1 * tanh(kp1 * (z_cell - 1. + 0.5 * z_b))
    else:
        # Region where the potential is zero.
        return -k1 * tan(k1 * (z_cell - 0.5 * z_a))


@jit(nopython=True)
def _one_body_func_log_dz2(z: float, obf_params: OBFParams) -> float:
    """One-body function second logarithmic derivative.

    :param z:
    :param obf_params:
    :return:
    """
    v0 = obf_params.lattice_depth
    r = obf_params.lattice_ratio
    e0 = obf_params.param_e0

    z_cell = (z % 1.)
    z_a, z_b = 1 / (1 + r), r / (1 + r)
    return v0 - e0 if z_a < z_cell else -e0


@jit(nopython=True)
def _two_body_func(rz: float, tbf_params: TBFParams) -> float:
    """Two-body function.

    :param rz:
    :param tbf_params:
    :return:
    """
    sc_size = tbf_params.supercell_size
    rm = tbf_params.tbf_contact_cutoff
    k2 = tbf_params.param_k2
    beta = tbf_params.param_beta
    r_off = tbf_params.param_r_off
    am = tbf_params.param_am

    # Two-body term.
    if rz < fabs(rm):
        return am * cos(k2 * (rz - r_off))
    else:
        return sin(pi * rz / sc_size) ** beta


@jit(nopython=True)
def _two_body_func_log_dz(rz: float, tbf_params: TBFParams) -> float:
    """Two-body function logarithmic derivative.

    :param rz:
    :param tbf_params:
    :return:
    """
    sc_size = tbf_params.supercell_size
    rm = tbf_params.tbf_contact_cutoff
    k2 = tbf_params.param_k2
    beta = tbf_params.param_beta
    r_off = tbf_params.param_r_off

    # Two-body term.
    if rz < fabs(rm):
        return -k2 * tan(k2 * (rz - r_off))
    else:
        return (pi / sc_size) * beta / (tan(pi * rz / sc_size))


@jit(nopython=True)
def _two_body_func_log_dz2(rz: float, tbf_params: TBFParams) -> float:
    """Two-body function logarithmic derivative.

    :param rz:
    :param tbf_params:
    :return:
    """
    sc_size = tbf_params.supercell_size
    rm = tbf_params.tbf_contact_cutoff
    k2 = tbf_params.param_k2
    beta = tbf_params.param_beta

    # Two-body term.
    if rz < fabs(rm):
        return -k2 * k2
    else:
        return (pi / sc_size) ** 2 * beta * (
                (beta - 1) / (tan(pi * rz / sc_size) ** 2) - 1
        )


@jit(nopython=True)
def _potential(z: float, params: Params) -> float:
    """Calculates the potential energy of the Bose gas due to the
     external potential.

     :param z: The current configuration of the positions of the
               particles.
     :param params:
     :return:
    """
    v0 = params.lattice_depth
    z_a = params.well_width
    v0d = params.defect_magnitude
    defects_sep = params.defects_sep
    n_cell, z_cell = divmod(z, 1)
    if not (n_cell % defects_sep):
        # We are in a lattice defect.
        return v0d if z_a < z_cell else 0.
    else:
        return v0 if z_a < z_cell else 0.


@jit(nopython=True)
def _real_distance(z_i, z_j, params: Params):
    """The real distance between two bosons.

    This routine takes into account the periodic boundary
    conditions of the QMC calculation.
    """
    sc_size = params.supercell_size
    return min_distance(z_i, z_j, sc_size)


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(qmc_base.jastrow.CoreFuncs):
    """Functions of a QMC model for a system with a trial wave function
    of the Bijl-Jastrow type.
    """

    @cached_property
    def model_params_transform(self):
        """"""

        @jit(nopython=True)
        def _model_params_transform(model_params: Params):
            """

            :param model_params:
            :return:
            """
            return np.array(tuple(model_params), dtype=np.float64)

        return _model_params_transform

    @cached_property
    def obf_params_transform(self):
        """"""

        @jit(nopython=True)
        def _obf_params_transform(obf_params: OBFParams):
            """

            :param obf_params:
            :return:
            """
            return np.array(tuple(obf_params), dtype=np.float64)

        return _obf_params_transform

    @cached_property
    def tbf_params_transform(self):
        """"""

        @jit(nopython=True)
        def _tbf_params_transform(tbf_params: TBFParams):
            """

            :param tbf_params:
            :return:
            """
            return np.array(tuple(tbf_params), dtype=np.float64)

        return _tbf_params_transform

    @cached_property
    def model_params_reconstruct(self):
        """"""

        @jit(nopython=True)
        def _model_params_reconstruct(model_params: np.ndarray):
            """

            :param model_params:
            :return:
            """
            # Somewhat ugly function...
            # First, we have to transform the elements of the array.
            model_params = Params(lattice_depth=model_params[0],
                                  lattice_ratio=model_params[1],
                                  interaction_strength=model_params[2],
                                  boson_number=int(model_params[3]),
                                  supercell_size=model_params[4],
                                  tbf_contact_cutoff=model_params[5],
                                  defect_magnitude=model_params[6],
                                  defects_sep=int(model_params[7]),
                                  well_width=model_params[8],
                                  barrier_width=model_params[9],
                                  is_free=bool(model_params[10]),
                                  is_ideal=bool(model_params[11]))
            return model_params

        return _model_params_reconstruct

    @cached_property
    def obf_params_reconstruct(self):
        """"""

        @jit(nopython=True)
        def _obf_params_reconstruct(obf_params: np.ndarray):
            """

            :param obf_params:
            :return:
            """
            # OBF is a tuple of floats.
            return OBFParams(lattice_depth=obf_params[0],
                             lattice_ratio=obf_params[1],
                             well_width=obf_params[2],
                             barrier_width=obf_params[3],
                             param_e0=obf_params[4],
                             param_k1=obf_params[5],
                             param_kp1=obf_params[6])

        return _obf_params_reconstruct

    @cached_property
    def tbf_params_reconstruct(self):
        """"""

        @jit(nopython=True)
        def _tbf_params_reconstruct(tbf_params: np.ndarray):
            """

            :param tbf_params:
            :return:
            """
            # TBF is a tuple of floats.
            return TBFParams(supercell_size=tbf_params[0],
                             tbf_contact_cutoff=tbf_params[1],
                             param_k2=tbf_params[2],
                             param_beta=tbf_params[3],
                             param_r_off=tbf_params[4],
                             param_am=tbf_params[5])

        return _tbf_params_reconstruct

    @cached_property
    def cfc_spec_transform(self):
        """"""

        @jit(nopython=True)
        def _cfc_spec_transform(model_params: Params,
                                obf_params: OBFParams,
                                tbf_params: TBFParams):
            """

            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            model_params = np.array(tuple(model_params), dtype=np.float64)
            obf_params = np.array(tuple(obf_params), dtype=np.float64)
            tbf_params = np.array(tuple(tbf_params), dtype=np.float64)
            return CFCSpecAlt(model_params, obf_params, tbf_params)

        return _cfc_spec_transform

    @cached_property
    def cfc_spec_reconstruct(self):
        """"""

        @jit(nopython=True)
        def _cfc_spec_reconstruct(model_params: Params,
                                  obf_params: OBFParams,
                                  tbf_params: TBFParams):
            """

            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            # Somewhat ugly function...
            # First, we have to transform the elements of the array.
            model_params = Params(lattice_depth=model_params[0],
                                  lattice_ratio=model_params[1],
                                  interaction_strength=model_params[2],
                                  boson_number=int(model_params[3]),
                                  supercell_size=model_params[4],
                                  tbf_contact_cutoff=model_params[5],
                                  defect_magnitude=model_params[6],
                                  defects_sep=int(model_params[7]),
                                  well_width=model_params[8],
                                  barrier_width=model_params[9],
                                  is_free=bool(model_params[10]),
                                  is_ideal=bool(model_params[11]))

            # OBF is a tuple of floats.
            obf_params = \
                OBFParams(*(obf_params[idx] for idx in range(len(obf_params))))

            # TBF is a tuple of floats.
            tbf_params = \
                TBFParams(*(tbf_params[idx] for idx in range(len(tbf_params))))

            return CFCSpec(model_params, obf_params, tbf_params)

        return _cfc_spec_reconstruct

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

    cfc_spec_nt: CFCSpec

    @classmethod
    def from_model_spec(cls, model_spec: Spec):
        """Builds the core functions for the given model Spec."""
        return cls(model_spec.cfc_spec)

    @cached_property
    def core_funcs(self):
        """The core functions of the model."""
        return core_funcs


@attr.s(auto_attribs=True, frozen=True)
class CSWFOptimizer(qmc_base.jastrow.CSWFOptimizer):
    """Class to optimize the trial-wave function.

    It uses the correlated sampling method to minimize the variance of
    the local energy of the model.
    """

    #: The spec of the model.
    spec: Spec

    #: The system configurations used for the minimization process.
    sys_conf_set: np.ndarray = attr.ib(cmp=False)

    #: The initial wave function values. Used to calculate the weights.
    ini_wf_abs_log_set: np.ndarray = attr.ib(cmp=False)

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: t.Optional[float] = attr.ib(cmp=False, default=None)

    #: Use threads or multiple process.
    use_threads: bool = attr.ib(default=True, cmp=False)

    #: Number of threads or process to use.
    num_workers: int = attr.ib(default=cpu_count(), cmp=False)

    #: Display log messages or not.
    verbose: bool = attr.ib(default=False, cmp=False)

    @cached_property
    def sys_conf_set_db(self):
        """"""
        sys_conf_set = [sys_conf for sys_conf in self.sys_conf_set]
        return db.from_sequence(sys_conf_set)

    def update_spec(self, tbf_contact_cutoff: float):
        """Updates the model spec.

        :param tbf_contact_cutoff:
        :return:
        """
        # NOTE: We have to convert to float first (numba errors
        #  appear otherwise 🤔)
        tbf_contact_cutoff = float(tbf_contact_cutoff)
        return attr.evolve(self.spec, tbf_contact_cutoff=tbf_contact_cutoff)

    @cached_property
    def _threaded_func(self):
        """"""
        wf_abs_log = core_funcs.wf_abs_log
        energy = core_funcs.energy

        @jit(nopython=True, nogil=True)
        def __threaded_func(sys_conf: np.ndarray, cfc_spec: CFCSpec):
            """Evaluates the energy and wave function, and releases de GIL.

            :param sys_conf: The system configuration.
            :param cfc_spec: The spec of the core functions.
            :return:
            """
            wf_abs_log_v = \
                wf_abs_log(sys_conf, cfc_spec.model_params,
                           cfc_spec.obf_params, cfc_spec.tbf_params)
            energy_v = energy(sys_conf, cfc_spec.model_params,
                              cfc_spec.obf_params, cfc_spec.tbf_params)
            return wf_abs_log_v, energy_v

        return __threaded_func

    def wf_abs_log_and_energy_set(self, cfc_spec: CFCSpec) -> \
            t.Tuple[np.ndarray, np.ndarray]:
        """"""

        # The function to execute in parallel. It return lists.
        func = self._threaded_func
        sys_conf_set_db = self.sys_conf_set_db
        db_results = sys_conf_set_db.map(func, cfc_spec=cfc_spec).compute()
        wf_abs_log_set, energies_set = list(zip(*db_results))

        # We need arrays.
        wf_abs_log_set = np.array(wf_abs_log_set)
        energies_set = np.array(energies_set)

        return wf_abs_log_set, energies_set

    @property
    def principal_function_bounds(self):
        """Boundaries of the variables of the principal function."""

        sc_size = self.spec.supercell_size
        sc_size_lower_bound = 5e-2
        sc_size_upper_bound = (0.5 - 5e-3) * sc_size

        func_bounds = [(sc_size_lower_bound, sc_size_upper_bound)]
        return func_bounds

    @property
    def dask_config(self):
        """Updates the ``dask`` configuration and returns the object.

        The config object is updated to honor the optimizer attributes
        ``num_workers`` and ``use_threads``.

        :return: The updated ``dask`` configuration object.
        """
        num_workers = self.num_workers
        use_threads = self.use_threads
        scheduler = 'threads' if use_threads else 'processes'
        return dask.config.set(scheduler=scheduler, num_workers=num_workers)

    def exec(self):
        """Starts the variance minimization process.

        :return: The initial spec, updated with the value of the
            ``tbf_contact_cutoff`` parameter that minimizes the variance,
            i.e., that optimizes the trial wave function.
        """
        verbose = self.verbose
        bounds = self.principal_function_bounds
        with self.dask_config:
            # Realize the minimization process.
            opt_params = differential_evolution(self.principal_function,
                                                bounds=bounds, disp=verbose)

        opt_tbf_contact_cutoff, = opt_params.x
        return self.update_spec(opt_tbf_contact_cutoff)
