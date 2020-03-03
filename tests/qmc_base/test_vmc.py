import typing as t

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from math import sqrt
from matplotlib import pyplot
from numpy.linalg import norm

from my_research_libs.qmc_base import vmc as vmc_udf, vmc_ndf
from my_research_libs.qmc_base.vmc import State


class WFParams(t.NamedTuple):
    """Represent the parameters of the of the gaussian pdf."""
    dims: int
    mu: float
    sigma: float


class TPFParams(vmc_ndf.TPFParams, t.NamedTuple):
    """Represents the transition probability function parameters."""
    dims: int
    sigma: float


class UTPFParams(vmc_udf.TPFParams, t.NamedTuple):
    """The uniform, transition probability function parameters."""
    dims: int
    move_spread: float


class SSFParams(vmc_udf.SSFParams, t.NamedTuple):
    """Static structure factor parameters."""
    assume_none: bool


class CFCSpec(vmc_udf.CFCSpec, t.NamedTuple):
    """Represent the spec of the core functions."""
    wf_params: WFParams
    tpf_params: TPFParams
    ssf_params: SSFParams


@attr.s(auto_attribs=True, frozen=True)
class NormalSampling(vmc_ndf.Sampling):
    """A spec to sampling the multidimensional gaussian."""

    dims: int

    mu: float
    sigma: float
    time_step: float
    rng_seed: int

    @property
    def wf_params(self):
        """"""
        wf_params = WFParams(self.dims, self.mu, self.sigma)
        return wf_params

    @property
    def tpf_params(self) -> TPFParams:
        """"""
        sigma = sqrt(self.time_step)
        tpf_params = TPFParams(self.dims, sigma)
        return tpf_params

    @property
    def ssf_params(self) -> SSFParams:
        """"""
        ssf_params = SSFParams(assume_none=True)
        return ssf_params

    @property
    def ssf_momenta(self):
        return None

    @property
    def cfc_spec(self) -> CFCSpec:
        """"""
        return CFCSpec(self.wf_params,
                       self.tpf_params,
                       self.ssf_params)

    def build_state(self, sys_conf: np.ndarray) -> State:
        """

        :param sys_conf:
        :return:
        """
        return self.core_funcs.init_prepare_state(sys_conf, self.cfc_spec)

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        return core_funcs


@nb.jit(nopython=True)
def _base_wf_abs_log(sys_conf: np.ndarray,
                     cfc_spec: CFCSpec):
    """

    :param sys_conf:
    :param cfc_spec:
    :return:
    """
    wf_params = cfc_spec.wf_params
    mean = wf_params.mu
    var = wf_params.sigma
    vn = norm(sys_conf - mean)
    return -vn ** 2 / (2 * var ** 2)


class CoreFuncs(vmc_ndf.CoreFuncs):
    """Functions to sample a multidimensional gaussian pdf."""

    @cached_property
    def wf_abs_log(self):
        """The logarithm of the Gaussian pdf."""

        @nb.jit(nopython=True)
        def _wf_abs_log(state_data: vmc_udf.StateData,
                        cfc_spec: CFCSpec):
            """

            :param state_data:
            :param cfc_spec:
            :return:
            """
            sys_conf = state_data.sys_conf
            return _base_wf_abs_log(sys_conf, cfc_spec)

        return _wf_abs_log

    @cached_property
    def energy(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.njit
        def _energy(step_idx: int,
                    state: vmc_udf.State,
                    cfc_spec: CFCSpec,
                    iter_props_array: vmc_udf.PropsData):
            """"""
            energy_set = iter_props_array.energy
            energy_set[step_idx] = 0.

        return _energy

    @property
    def one_body_density(self):
        return None

    @property
    def init_obd_est_data(self):
        return None

    @cached_property
    def fourier_density(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.njit
        def _fourier_density(step_idx: int,
                             state: vmc_udf.State,
                             cfc_spec: CFCSpec,
                             ssf_exec_data: vmc_udf.SSFExecData):
            return 0.

        return _fourier_density

    @cached_property
    def init_ssf_est_data(self):
        """"""

        # noinspection PyUnusedLocal
        @nb.njit
        def _init_ssf_est_data(num_steps_block: int,
                               cfc_spec: CFCSpec):
            """

            :param num_steps_block:
            :param cfc_spec:
            :return:
            """
            momenta = np.zeros((1,), dtype=np.float64)
            iter_ssf_array = np.zeros((num_steps_block, 1), dtype=np.float64)
            return vmc_udf.SSFExecData(momenta, iter_ssf_array)

        return _init_ssf_est_data

    @cached_property
    def init_state_data(self):
        """Initialize the data arrays for the VMC states generator."""

        # noinspection PyUnusedLocal
        @nb.njit
        def _init_state_data(base_shape: t.Tuple[int, ...],
                             cfc_spec: CFCSpec):
            """

            :param cfc_spec:
            :return:
            """
            num_dims = cfc_spec.wf_params.dims
            confs_shape = base_shape + (num_dims,)
            state_sys_conf = np.zeros(confs_shape, dtype=np.float64)
            return vmc_udf.StateData(state_sys_conf)

        return _init_state_data

    @cached_property
    def init_prepare_state(self):
        """

        :return:
        """
        move_stat = vmc_udf.STAT_REJECTED
        init_state_data = self.init_state_data
        build_state = self.build_state

        @nb.njit
        def _init_prepare_state(sys_conf: np.ndarray,
                                cfc_spec: CFCSpec):
            """

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            base_shape = ()
            state_data = init_state_data(base_shape, cfc_spec)
            wf_abs_log = _base_wf_abs_log(sys_conf, cfc_spec)
            state_data.sys_conf[:] = sys_conf[:]
            return build_state(state_data, wf_abs_log, move_stat)

        return _init_prepare_state

    @cached_property
    def build_state(self):
        """"""

        @nb.njit
        def _build_state(state_data: vmc_udf.StateData,
                         wf_abs_log: float,
                         move_stat: int):
            """

            :param state_data:
            :param wf_abs_log:
            :param move_stat:
            :return:
            """
            return vmc_udf.State(state_data.sys_conf,
                                 wf_abs_log, move_stat)

        return _build_state

    @cached_property
    def ith_sys_conf_tpf(self):
        """"""

        rand_displace = self.rand_displace

        @nb.jit(nopython=True)
        def _ith_sys_conf_ppf(i_: int, ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              tpf_params: TPFParams):
            """"""
            z_i = ini_sys_conf[i_]
            rnd_spread = rand_displace(tpf_params)
            prop_sys_conf[i_] = z_i + rnd_spread

        return _ith_sys_conf_ppf

    @cached_property
    def sys_conf_tpf(self):
        """Proposal probability function."""

        ith_sys_conf_tpf = self.ith_sys_conf_tpf

        # noinspection PyShadowingNames
        @nb.jit(nopython=True)
        def _sys_conf_tpf(actual_state_data: vmc_udf.StateData,
                          next_state_data: vmc_udf.StateData,
                          cfc_spec: CFCSpec):
            """Changes the current configuration of the system.

            :param actual_state_data:
            :param next_state_data:
            :param cfc_spec: The parameters of the function.
            """
            ini_sys_conf = actual_state_data.sys_conf
            prop_sys_conf = next_state_data.sys_conf
            tpf_params = cfc_spec.tpf_params
            scs = tpf_params.dims  # Number of dimensions
            for i_ in range(scs):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_params)

        return _sys_conf_tpf


core_funcs = CoreFuncs()

dims, mu, sigma = 10, 1, 1
l_scale = sigma / 2
time_step = l_scale ** 2
ini_sys_conf = np.random.random_sample(dims)
sampling = NormalSampling(dims, mu, sigma, time_step,
                          rng_seed=0)
ini_state = sampling.build_state(ini_sys_conf)
ini_state_data = \
    core_funcs.init_state_data_from_state(ini_state, sampling.cfc_spec)
next_state_data = \
    core_funcs.init_state_data_from_state(ini_state, sampling.cfc_spec)


def test_wf_abs_log():
    """Test the wave function."""
    wf_abs_log = core_funcs.wf_abs_log
    v = wf_abs_log(ini_state_data, sampling.cfc_spec)
    print(v)


def test_sys_conf_tpf():
    """Test the transition probability function."""
    ini_state_sys_conf = ini_state.sys_conf
    next_state_sys_conf = next_state_data.sys_conf
    sys_conf_tpf = core_funcs.sys_conf_tpf
    sys_conf_tpf(ini_state_data, next_state_data, sampling.cfc_spec)
    print(next_state_sys_conf - ini_state_sys_conf)


def test_sampling():
    """Test the sampling states."""
    ar = 0
    num_steps = 4096 * 128
    for cj_, data in enumerate(sampling.states(ini_state)):
        stat = data.move_stat
        ar += stat
        if cj_ + 1 >= num_steps:
            break

    ar /= num_steps
    chain_data = sampling.as_chain(num_steps, ini_state)
    sys_conf_set = chain_data.confs
    ar_ = chain_data.accept_rate

    assert sys_conf_set.shape == (num_steps, sampling.dims)
    assert ar == ar_
    print(f"Sampling acceptance rate: {ar:.5g}")

    ax = pyplot.gca()
    ax.hist(sys_conf_set[:, 0], bins=256, density=True)
    pyplot.show()
    print(sys_conf_set)


def test_blocks():
    """Test the sampling blocks of states."""
    num_blocks = 64
    num_steps_block = 4096
    blocks_enum: vmc_udf.T_E_SBlocksIter = \
        enumerate(sampling.blocks(num_steps_block, ini_state))

    for bj_, block_data in blocks_enum:
        ar = block_data.accept_rate
        print(f"Sampling acceptance rate: {ar:.5g}")
        if bj_ + 1 >= num_blocks:
            break


if __name__ == '__main__':
    test_sampling()
