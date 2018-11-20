from math import sqrt
from typing import NamedTuple

import attr
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
    sigma: float


class UTPFSpecNT(vmc.UTPFSpecNT, NamedTuple):
    """The uniform, transition probability function parameters."""
    dims: int
    move_spread: float


@attr.s(auto_attribs=True)
class Spec(vmc.Spec):
    """A spec to sampling the multidimensional gaussian."""

    dims: int
    mu: float
    sigma: float
    time_step: float
    chain_samples: int
    ini_sys_conf: np.ndarray
    burn_in_samples: int
    rng_seed: int

    @property
    def wf_spec_nt(self):
        """"""
        return WFSpecNT(self.dims, self.mu, self.sigma)

    @property
    def tpf_spec_nt(self):
        """"""
        sigma = sqrt(self.time_step)
        return TPFSpecNT(self.dims, sigma)

    @property
    def cfc_spec_nt(self):
        """"""
        return vmc.CFCSpecNT(self.wf_spec_nt,
                             self.tpf_spec_nt,
                             self.chain_samples,
                             self.ini_sys_conf,
                             self.burn_in_samples,
                             self.rng_seed)


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


class UniformCoreFuncs(CoreFuncs, vmc.UniformCoreFuncs):
    """Functions to sample a multidimensional Gaussian pdf."""
    pass


def test_core_funcs():
    """"""
    dims, mu, sigma = 10, 1, 0.05
    time_step = (0.5 * sigma) ** 2
    ini_sys_conf = init_get_sys_conf(dims)
    spec = Spec(dims, mu, sigma, time_step,
                chain_samples=100000,
                ini_sys_conf=ini_sys_conf,
                burn_in_samples=10000,
                rng_seed=0)

    core_funcs = CoreFuncs()
    chain = core_funcs.as_chain(*spec.cfc_spec_nt)
    sys_conf_chain = chain.sys_conf_chain

    assert sys_conf_chain.shape == (spec.chain_samples, spec.dims)

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)


def test_uniform_core_funcs():
    """"""
    dims, mu, sigma = 10, 1, 1
    wf_spec = WFSpecNT(dims, mu, sigma)
    tpf_spec = UTPFSpecNT(dims, move_spread=0.5 * sigma)
    ini_sys_conf = np.random.random_sample(dims)
    chain_samples = 100000

    funcs = UniformCoreFuncs()
    cfc_spec_nt = vmc.CFCSpecNT(wf_spec, tpf_spec, chain_samples=chain_samples,
                                ini_sys_conf=ini_sys_conf,
                                burn_in_samples=10000,
                                rng_seed=0)
    chain = funcs.as_chain(cfc_spec_nt)

    sys_conf_chain = chain.sys_conf_chain
    assert sys_conf_chain.shape == (chain_samples, dims)

    ax = pyplot.gca()
    ax.hist(sys_conf_chain[:, 0], bins=100)
    pyplot.show()
    print(chain)
