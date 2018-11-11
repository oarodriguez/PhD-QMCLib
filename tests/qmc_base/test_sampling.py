from typing import NamedTuple

import numba as nb
import numpy as np
from matplotlib import pyplot
from numpy.linalg import norm

from my_research_libs.qmc_base import vmc


class WFSpecNT(NamedTuple):
    """The parameters of the gaussian pdf."""
    dims: int
    mu: float
    sigma: float


class TPFSpecNT(vmc.TPFSpecNT, NamedTuple):
    """The gaussian, transition probability function parameters."""
    dims: int
    time_step: float


class UTPFSpecNT(vmc.UTPFSpecNT, NamedTuple):
    """The uniform, transition probability function parameters."""
    dims: int
    move_spread: float


class SamplingFuncs(vmc.SamplingFuncs):
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
                              tpf_spec: TPFSpecNT):
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
                          tpf_spec: TPFSpecNT):
            """Changes the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_spec: The parameters of the function.
            """
            scs = tpf_spec.dims  # Number of dimensions
            for i_ in range(scs):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_spec)

        return _sys_conf_ppf


class UniformSamplingFuncs(SamplingFuncs, vmc.UniformSamplingFuncs):
    """Functions to sample a multidimensional Gaussian pdf."""
    pass


def test_sampling_funcs():
    """"""
    dims, mu, sigma = 10, 1, 0.05
    wf_spec = WFSpecNT(dims, mu, sigma)
    tpf_spec = TPFSpecNT(dims, time_step=0.5 * sigma)
    ini_sys_conf = np.random.random_sample(dims)

    funcs = SamplingFuncs()
    chain = funcs.as_chain(wf_spec, tpf_spec, ini_sys_conf,
                           chain_samples=100000,
                           burn_in_samples=10000,
                           rng_seed=0)

    sys_conf_chain = chain.sys_conf_chain

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)


def test_uniform_sampling_funcs():
    """"""
    dims, mu, sigma = 10, 1, 1
    wf_spec = WFSpecNT(dims, mu, sigma)
    tpf_spec = UTPFSpecNT(dims, move_spread=0.5 * sigma)
    ini_sys_conf = np.random.random_sample(dims)

    funcs = UniformSamplingFuncs()
    chain = funcs.as_chain(wf_spec, tpf_spec, ini_sys_conf,
                           chain_samples=100000,
                           burn_in_samples=10000,
                           rng_seed=0)

    sys_conf_chain = chain.sys_conf_chain

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)
