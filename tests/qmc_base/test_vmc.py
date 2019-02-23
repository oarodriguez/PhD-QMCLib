import typing as t
from math import sqrt

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from matplotlib import pyplot
from numpy.linalg import norm

from my_research_libs.qmc_base import vmc as vmc_udf, vmc_ndf
from my_research_libs.qmc_base.vmc import State
from my_research_libs.util.attr import Record


@attr.s(auto_attribs=True, frozen=True)
class WFParams(Record):
    """Represent the parameters of the of the gaussian pdf."""
    dims: int
    mu: float
    sigma: float


@attr.s(auto_attribs=True, frozen=True)
class TPFParams(vmc_ndf.TPFParams, Record):
    """Represents the transition probability function parameters."""
    dims: int
    sigma: float


@attr.s(auto_attribs=True, frozen=True)
class UTPFParams(vmc_udf.TPFParams, Record):
    """The uniform, transition probability function parameters."""
    dims: int
    move_spread: float


class CFCSpec(vmc_udf.CFCSpec, t.NamedTuple):
    """Represent the spec of the core functions."""
    wf_params: WFParams
    tpf_params: TPFParams


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
        return wf_params.as_record()

    @property
    def tpf_params(self) -> TPFParams:
        """"""
        sigma = sqrt(self.time_step)
        tpf_params = TPFParams(self.dims, sigma)
        return tpf_params.as_record()

    @property
    def cfc_spec(self) -> CFCSpec:
        """"""
        return CFCSpec(self.wf_params,
                       self.tpf_params)

    def build_state(self, sys_conf: np.ndarray) -> State:
        # TODO: Implement
        pass

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        return core_funcs


def init_get_sys_conf(dims: int):
    """Generates an initial random configuration."""
    return np.random.random_sample(dims)


class CoreFuncs(vmc_ndf.CoreFuncs):
    """Functions to sample a multidimensional gaussian pdf."""

    @cached_property
    def wf_abs_log(self):
        """The logarithm of the Gaussian pdf."""

        @nb.jit(nopython=True)
        def _wf_abs_log(sys_conf: np.ndarray,
                        cfc_spec: CFCSpec):
            """

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            wf_params = cfc_spec.wf_params
            mu = wf_params.mu
            sigma = wf_params.sigma
            vn = norm(sys_conf - mu)
            return -vn ** 2 / (2 * sigma ** 2)

        return _wf_abs_log

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
        def _sys_conf_tpf(ini_sys_conf: np.ndarray,
                          prop_sys_conf: np.ndarray,
                          cfc_spec: CFCSpec):
            """Changes the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param cfc_spec: The parameters of the function.
            """
            tpf_params = cfc_spec.tpf_params
            scs = tpf_params.dims  # Number of dimensions
            for i_ in range(scs):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_params)

        return _sys_conf_tpf


core_funcs = CoreFuncs()

dims, mu, sigma = 10, 1, 1
l_scale = sigma / 2
time_step = l_scale ** 2
ini_sys_conf = init_get_sys_conf(dims)
sampling = NormalSampling(dims, mu, sigma, time_step,
                          rng_seed=0)


def test_wf_abs_log():
    """Test the wave function."""
    wf_abs_log = core_funcs.wf_abs_log
    v = wf_abs_log(ini_sys_conf, sampling.cfc_spec)
    print(v)


def test_sys_conf_tpf():
    """Test the transition probability function."""
    proc_sys_conf = np.zeros_like(ini_sys_conf)
    sys_conf_tpf = core_funcs.sys_conf_tpf
    sys_conf_tpf(ini_sys_conf, proc_sys_conf, sampling.cfc_spec)
    print(proc_sys_conf - ini_sys_conf)


def test_sampling():
    """Test the sampling states."""
    ar = 0
    num_steps = 4096 * 64
    for cj_, data in enumerate(sampling.states(ini_sys_conf)):
        stat = data.move_stat
        ar += stat
        if cj_ + 1 >= num_steps:
            break

    ar /= num_steps
    chain_data = sampling.as_chain(num_steps, ini_sys_conf)
    sys_conf_set = chain_data.confs
    ar_ = chain_data.accept_rate

    assert sys_conf_set.shape == (num_steps, sampling.dims)
    assert ar == ar_
    print(f"Sampling acceptance rate: {ar:.5g}")

    ax = pyplot.gca()
    ax.hist(sys_conf_set[:, 0], bins=100)
    pyplot.show()
    print(sys_conf_set)


def test_batches():
    """Test the sampling batches of states."""
    num_batches = 64
    num_steps_batch = 4096
    batches_enum: vmc_udf.T_E_SBatchesIter = \
        enumerate(sampling.batches(num_steps_batch, ini_sys_conf))

    ax = pyplot.gca()

    for bj_, batch_data in batches_enum:
        ar = batch_data.accept_rate
        sys_conf_set = batch_data.confs

        assert sys_conf_set.shape == (num_steps_batch, sampling.dims)
        print(f"Sampling acceptance rate: {ar:.5g}")

        ax.hist(sys_conf_set[:, 0], bins=100)
        if bj_ + 1 >= num_batches:
            break

    pyplot.show()
