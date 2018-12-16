from math import sqrt
from typing import NamedTuple

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from matplotlib import pyplot
from numpy.linalg import norm

from my_research_libs.qmc_base import vmc


class WFSpecNT(NamedTuple):
    """The parameters of the gaussian pdf."""
    dims: int
    mu: float
    sigma: float


class NTPFSpecNT(vmc.NTPFSpecNT, NamedTuple):
    """The gaussian, transition probability function parameters."""
    dims: int
    sigma: float


class UTPFSpecNT(vmc.UTPFSpecNT, NamedTuple):
    """The uniform, transition probability function parameters."""
    dims: int
    move_spread: float


@attr.s(auto_attribs=True, frozen=True)
class NormalSampling(vmc.NormalSampling):
    """A spec to sampling the multidimensional gaussian."""

    dims: int
    mu: float
    sigma: float
    time_step: float
    num_steps: int
    ini_sys_conf: np.ndarray
    rng_seed: int

    @property
    def wf_spec_nt(self):
        """"""
        return WFSpecNT(self.dims, self.mu, self.sigma)

    @property
    def tpf_spec_nt(self):
        """"""
        sigma = sqrt(self.time_step)
        return NTPFSpecNT(self.dims, sigma)

    @cached_property
    def core_funcs(self) -> 'NormalCoreFuncs':
        """The core functions of the sampling."""
        return NormalCoreFuncs()


def init_get_sys_conf(dims: int):
    """Generates an initial random configuration."""
    return np.random.random_sample(dims)


class CoreFuncs(vmc.CoreFuncs):
    """Functions to sample a multidimensional gaussian pdf."""

    @property
    def wf_abs_log(self):
        """The logarithm of the Gaussian pdf."""

        @nb.jit(nopython=True)
        def _wf_abs_log(sys_conf: np.ndarray, wf_spec: WFSpecNT):
            """"""
            mu = wf_spec.mu
            sigma = wf_spec.sigma
            vn = norm(sys_conf - mu)
            return -vn ** 2 / (2 * sigma ** 2)

        return _wf_abs_log

    @property
    def ith_sys_conf_tpf(self):
        """"""

        rand_displace = self.rand_displace

        @nb.jit(nopython=True)
        def _ith_sys_conf_ppf(i_: int, ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              tpf_spec: NTPFSpecNT):
            """"""
            z_i = ini_sys_conf[i_]
            rnd_spread = rand_displace(tpf_spec)
            prop_sys_conf[i_] = z_i + rnd_spread

        return _ith_sys_conf_ppf

    @property
    def sys_conf_tpf(self):
        """Proposal probability function."""

        ith_sys_conf_tpf = self.ith_sys_conf_tpf

        @nb.jit(nopython=True)
        def _sys_conf_ppf(ini_sys_conf: np.ndarray,
                          prop_sys_conf: np.ndarray,
                          tpf_spec: NTPFSpecNT):
            """Changes the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_spec: The parameters of the function.
            """
            scs = tpf_spec.dims  # Number of dimensions
            for i_ in range(scs):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_spec)

        return _sys_conf_ppf


class NormalCoreFuncs(CoreFuncs, vmc.NormalCoreFuncs):
    """Functions to sample a multidimensional Gaussian pdf."""
    pass


def test_core_funcs():
    """"""
    dims, mu, sigma = 10, 1, 1
    wf_spec = WFSpecNT(dims, mu, sigma)
    tpf_spec = UTPFSpecNT(dims, move_spread=0.5 * sigma)
    num_steps = 4096 * 64
    ini_sys_conf = init_get_sys_conf(dims)
    spec_nt = vmc.SpecNT(wf_spec, tpf_spec,
                         num_steps=num_steps,
                         ini_sys_conf=ini_sys_conf,
                         rng_seed=0)

    core_funcs = CoreFuncs()
    chain = core_funcs.as_chain(*spec_nt)
    sys_conf_chain = chain.sys_conf_chain

    assert sys_conf_chain.shape == (num_steps, dims)

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)


def test_normal_core_funcs():
    """"""
    dims, mu, sigma = 10, 1, 1
    wf_spec = WFSpecNT(dims, mu, sigma)
    tpf_spec = NTPFSpecNT(dims, 0.15 * sigma)
    ini_sys_conf = np.random.random_sample(dims)
    num_steps = 4096 * 64

    core_funcs = NormalCoreFuncs()
    spec_nt = vmc.SpecNT(wf_spec, tpf_spec,
                         num_steps=num_steps,
                         ini_sys_conf=ini_sys_conf,
                         rng_seed=0)
    chain = core_funcs.as_chain(*spec_nt)
    sys_conf_chain = chain.sys_conf_chain

    assert sys_conf_chain.shape == (num_steps, dims)

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)


def test_normal_sampling():
    """"""
    dims, mu, sigma = 10, 1, 1
    l_scale = sigma / 2
    time_step = l_scale ** 2
    ini_sys_conf = init_get_sys_conf(dims)
    num_steps = 4096 * 64
    sampling = NormalSampling(dims, mu, sigma, time_step,
                              num_steps=num_steps,
                              ini_sys_conf=ini_sys_conf,
                              rng_seed=0)

    ar = 0
    for data in sampling:
        sys_conf, wfv, stat = data
        ar += stat
    ar /= num_steps

    chain_data = sampling.as_chain()
    sys_conf_chain, wf_abs_log_chain, ar_ = chain_data

    assert sys_conf_chain.shape == (sampling.num_steps, sampling.dims)
    assert ar == ar_
    print(f"Sampling acceptance rate: {ar:.5g}")

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(sys_conf_chain)
