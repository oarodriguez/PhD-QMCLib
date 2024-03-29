from math import exp, fabs, pi, sin, tan

import attr
import numba as nb
from cached_property import cached_property
from numpy import random

from phd_qmclib.qmc_base import jastrow, utils
from phd_qmclib.qmc_base.jastrow import SysConfSlot
from phd_qmclib.util.attr import Record


@attr.s(auto_attribs=True, frozen=True)
class Params(jastrow.Params, Record):
    """"""
    boson_number: int
    supercell_size: float
    is_free: bool
    is_ideal: bool


@attr.s(auto_attribs=True, frozen=True)
class OBFParams(jastrow.OBFParams, Record):
    """"""
    supercell_size: float


@attr.s(auto_attribs=True, frozen=True)
class TBFParams(jastrow.TBFParams, Record):
    """"""
    supercell_size: float


@attr.s(auto_attribs=True)
class Spec(jastrow.Spec):
    """A simple spec for testing."""

    boson_number: int

    supercell_size: float

    @property
    def is_free(self):
        return False

    @property
    def is_ideal(self):
        return False

    @property
    def params(self):
        spec_params = Params(self.boson_number, self.supercell_size,
                             self.is_free, self.is_ideal)
        return spec_params

    @property
    def obf_params(self):
        return OBFParams(self.supercell_size)

    @property
    def tbf_params(self):
        return TBFParams(self.supercell_size)

    @property
    def boundaries(self):
        sc_size = self.supercell_size
        return 0., 1. * sc_size

    @property
    def sys_conf_shape(self):
        """"""
        nop = self.boson_number
        # noinspection PyTypeChecker
        return len(SysConfSlot), nop

    def init_get_sys_conf(self):
        """Creates and initializes a system configuration with the
        positions of the particles arranged in the order specified
        by ``dist_type`` argument.

        :return:
        """
        nop = self.boson_number
        sc_size = self.supercell_size
        z_min, z_max = self.boundaries
        sys_conf = self.get_sys_conf_buffer()
        pos_slot = SysConfSlot.pos
        spread = sc_size * random.random_sample(nop)
        sys_conf[pos_slot, :] = z_min + spread % sc_size
        return sys_conf

    @property
    def var_params_bounds(self):
        return None

    @property
    def cfc_spec(self):
        return jastrow.CFCSpec(self.params.as_record(),
                               self.obf_params.as_record(),
                               self.tbf_params.as_record())

    @property
    def phys_funcs(self):
        raise NotImplementedError


class CoreFuncs(jastrow.CoreFuncs):
    """Concrete core functions for testing."""

    @cached_property
    def is_free(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _is_free(model_params):
            """"""
            return True

        return _is_free

    @cached_property
    def is_ideal(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _is_ideal(model_params):
            """"""
            return True

        return _is_ideal

    @cached_property
    def real_distance(self):
        """"""
        min_distance = utils.min_distance

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _real_distance(z_i, z_j, model_params: jastrow.Params):
            """The real distance between two bosons."""
            sc_size = model_params.supercell_size
            return min_distance(z_i, z_j, sc_size)

        return _real_distance

    @cached_property
    def one_body_func(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func(z: float, obf_params: OBFParams):
            """"""
            sc_size = obf_params.supercell_size
            uz = 0.25 / sc_size * z
            return exp(-uz ** 2)

        return _one_body_func

    @cached_property
    def two_body_func(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func(rz: float, tbf_params: TBFParams):
            """"""
            sc_size = tbf_params.supercell_size
            return sin(pi * fabs(rz) / sc_size)

        return _two_body_func

    @cached_property
    def one_body_func_log_dz(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func_log_dz(z: float, obf_params: OBFParams):
            """"""
            sc_size = obf_params.supercell_size
            uz = 0.25 / sc_size * z
            return - 2. * uz * (0.25 / sc_size)

        return _one_body_func_log_dz

    @cached_property
    def two_body_func_log_dz(self):
        """"""
        sign = utils.sign

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func_log_dz(rz, tbf_params: TBFParams):
            """"""
            sc_size = tbf_params.supercell_size
            sgn = sign(rz)
            return pi / sc_size / tan(fabs(rz)) * sgn

        return _two_body_func_log_dz

    @cached_property
    def one_body_func_log_dz2(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func_log_dz2(z, obf_params: OBFParams):
            """"""
            sc_size = obf_params.supercell_size
            uz = 0.25 * z / sc_size
            return (4. * uz ** 2. - 2.) * (0.25 / sc_size) ** 2

        return _one_body_func_log_dz2

    @property
    def two_body_func_log_dz2(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func_log_dz2(rz, tbf_params: TBFParams):
            """"""
            sc_size = tbf_params.supercell_size
            return - (pi / sc_size) ** 2

        return _two_body_func_log_dz2

    @property
    def potential(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _potential(z, model_params=None):
            """"""
            return 0.

        return _potential


def test_ith_wf_abs_log():
    """Testing the routines to calculate the wave function."""

    nop, sc_size = 100, 100
    model_params = Spec(nop, sc_size)
    sys_conf = model_params.init_get_sys_conf()
    cfc_spec = model_params.cfc_spec

    core_funcs = CoreFuncs()
    wf_v1 = core_funcs.ith_wf_abs_log(0, sys_conf, cfc_spec)
    wf_v2 = core_funcs.wf_abs_log(sys_conf, cfc_spec)
    wf_v3 = core_funcs.wf_abs(sys_conf, cfc_spec)

    print(f"* Function ith_wf_abs_log: {wf_v1:.3g}")
    print(f"* Function wf_abs_log: {wf_v2:.3g}")
    print(f"* Function wf_abs: {wf_v3:.3g}")


def test_drift_funcs():
    """Testing the routines to calculate the wave function."""

    nop, sc_size = 100, 100
    model_params = Spec(nop, sc_size)
    sys_conf = model_params.init_get_sys_conf()
    cfc_spec = model_params.cfc_spec

    core_funcs = CoreFuncs()
    drift_v1 = core_funcs.ith_drift(0, sys_conf, cfc_spec)
    drift_v2 = core_funcs.drift(sys_conf, cfc_spec)

    print(f"* Function ith_drift: {drift_v1:.8g}")
    print(f"* Function drift: {drift_v2}")


def test_energy_funcs():
    """Testing the routines to calculate the wave function."""

    nop, sc_size = 100, 100
    model_params = Spec(nop, sc_size)
    sys_conf = model_params.init_get_sys_conf()
    cfc_spec = model_params.cfc_spec

    core_funcs = CoreFuncs()
    e_r1 = core_funcs.ith_energy(0, sys_conf, cfc_spec)
    e_r2, _ = core_funcs.ith_energy_and_drift(0, sys_conf, cfc_spec)

    assert e_r1 == e_r2

    print(f"* Function <ith_energy>: {e_r1:.16g}")
    print(f"* Function <ith_energy_and_drift>: {(e_r2, _)}")


def test_obd_funcs():
    """Testing the functions to calculate the one-body density."""

    nop, sc_size = 100, 100
    model_params = Spec(nop, sc_size)
    sys_conf = model_params.init_get_sys_conf()
    cfc_spec = model_params.cfc_spec

    core_funcs = CoreFuncs()
    sz = model_params.supercell_size / nop
    obd_r1 = core_funcs.ith_one_body_density(0, sz, sys_conf, cfc_spec)
    obd_r2 = core_funcs.one_body_density(sz, sys_conf, cfc_spec)

    print(f"* Function <ith_one_body_density>: {obd_r1:.16g}")
    print(f"* Function <one_body_density>: {obd_r2:.16g}")


def test_sf_funcs():
    """Testing the functions to calculate the static structure factor."""

    nop, sc_size = 100, 100
    model_params = Spec(nop, sc_size)
    sys_conf = model_params.init_get_sys_conf()
    cfc_spec = model_params.cfc_spec

    core_funcs = CoreFuncs()
    kz = 2 * pi / sc_size
    sf_r1 = core_funcs.fourier_density(kz, sys_conf, cfc_spec)

    print(f"* Function <fourier_density>: {sf_r1}")
