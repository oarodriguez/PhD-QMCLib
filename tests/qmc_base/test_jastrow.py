import attr
import numba as nb
from cached_property import cached_property
from numpy import random
import numpy as np

from my_research_libs.qmc_base import jastrow, utils


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
    def as_named_tuple(self):
        return jastrow.SpecNT(self.boson_number,
                              self.supercell_size,
                              self.is_free,
                              self.is_ideal)

    @property
    def obf_spec_nt(self):
        return jastrow.OBFSpecNT()

    @property
    def tbf_spec_nt(self):
        return jastrow.TBFSpecNT()

    @property
    def boundaries(self):
        sc_size = self.supercell_size
        return 0., 1. * sc_size

    @property
    def sys_conf_shape(self):
        """"""
        nop = self.boson_number
        return len(self.sys_conf_slots), nop

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
        pos_slot = self.sys_conf_slots.pos
        spread = sc_size * random.random_sample(nop)
        sys_conf[pos_slot, :] = z_min + spread % sc_size
        return sys_conf

    @property
    def var_params_bounds(self):
        return None

    @property
    def cfc_spec_nt(self):
        return jastrow.CFCSpecNT(self.as_named_tuple,
                                 self.obf_spec_nt,
                                 self.tbf_spec_nt)

    @property
    def core_funcs(self) -> 'CoreFuncs':
        return CoreFuncs()


class CoreFuncs(jastrow.CoreFuncs):
    """Concrete core functions for testing."""

    @cached_property
    def is_free(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _is_free(model_spec):
            """"""
            return True

        return _is_free

    @cached_property
    def is_ideal(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _is_ideal(model_spec):
            """"""
            return True

        return _is_ideal

    @cached_property
    def real_distance(self):
        """"""
        min_distance = utils.min_distance

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _real_distance(z_i, z_j, model_spec: jastrow.SpecNT):
            """The real distance between two bosons."""
            sc_size = model_spec.supercell_size
            return min_distance(z_i, z_j, sc_size)

        return _real_distance

    @cached_property
    def one_body_func(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func(z, obf_spec=None):
            """"""
            return 1.

        return _one_body_func

    @cached_property
    def two_body_func(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func(rz, tbf_spec=None):
            """"""
            return random.rand()

        return _two_body_func

    @cached_property
    def one_body_func_log_dz(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func_log_dz(z, obf_spec=None):
            """"""
            return 0.

        return _one_body_func_log_dz

    @cached_property
    def two_body_func_log_dz(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func_log_dz(rz, tbf_spec):
            """"""
            return random.rand()

        return _two_body_func_log_dz

    @cached_property
    def one_body_func_log_dz2(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _one_body_func_log_dz2(z, obf_spec=None):
            """"""
            return 0.

        return _one_body_func_log_dz2

    @property
    def two_body_func_log_dz2(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _two_body_func_log_dz2(rz, tbf_spec=None):
            """"""
            return 0.

        return _two_body_func_log_dz2

    @property
    def potential(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, cache=True)
        def _potential(z, model_spec=None):
            """"""
            return 0.

        return _potential


def test_ith_wf_abs_log():
    """Testing the routines to calculate the wave function."""

    nop, sc_size = 100, 100
    model_spec = Spec(nop, sc_size)
    sys_conf = model_spec.init_get_sys_conf()
    cfc_spec = model_spec.cfc_spec_nt

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
    model_spec = Spec(nop, sc_size)
    sys_conf = model_spec.init_get_sys_conf()
    cfc_spec = model_spec.cfc_spec_nt

    core_funcs = CoreFuncs()
    drift_v1 = core_funcs.ith_drift(0, sys_conf, cfc_spec)
    drift_v2 = core_funcs.drift(sys_conf, cfc_spec)

    print(f"* Function ith_drift: {drift_v1:.8g}")
    print(f"* Function drift: {drift_v2}")

