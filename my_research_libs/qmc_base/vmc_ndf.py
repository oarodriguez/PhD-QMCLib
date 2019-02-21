import typing as t
from abc import ABCMeta, abstractmethod

import numpy as np
from cached_property import cached_property
from numba import jit
from numpy import random as random

from . import vmc as vmc_udf


class WFSpecNT(t.NamedTuple):
    """The parameters of the trial wave function.

    We declare this class to help with typing and nothing more. A concrete
    spec should be implemented for every concrete model. It is recommended
    to inherit from this class to keep a logical sequence in the code.
    """
    pass


class TPFSpecNT(t.NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a normal (gaussian) distribution function.
    """
    #: The standard deviation of the normal distribution.
    sigma: float


class SpecNT(t.NamedTuple):
    """The parameters to realize a sampling."""
    wf_spec: WFSpecNT
    tpf_spec: TPFSpecNT
    ini_sys_conf: np.ndarray
    rng_seed: int


class Sampling(vmc_udf.SamplingBase):
    """Realizes a VMC sampling.

    Defines the parameters and methods to realize of a Variational Monte
    Carlo calculation. A normal distribution is used to generate random
    numbers.
    """
    __slots__ = ()

    #: The "time-step" (squared, average move spread) of the sampling.
    time_step: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    @property
    @abstractmethod
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        pass


@jit(nopython=True)
def rand_displace(tpf_spec: TPFSpecNT):
    """Generates a random number from a normal distribution with
    zero mean and a a standard deviation ``ppf_spec.move_spread``.
    """
    # Avoid `Untyped global name error` when executing the code in a
    # multiprocessing pool.
    # TODO: Make tests with the symbols imported globally
    normal = random.normal

    # NOTE: We may use the time-step approach.
    # Some papers suggest to use the same Gaussian proposal
    # probability function, but with a **time step** parameter,
    # which is equal to the variance of the proposal distribution.
    # sigma = sqrt(time_step)
    sigma = tpf_spec.sigma
    return normal(0, sigma)


class CoreFuncs(vmc_udf.CoreFuncs, metaclass=ABCMeta):
    """The core functions to realize a VMC sampling.

    These functions perform the sampling of the probability density of a QMC
    model using the Metropolis-Hastings algorithm. A normal distribution is
    used to generate random numbers.
    """

    @cached_property
    def rand_displace(self):
        """Generates a random number from a uniform distribution."""
        return rand_displace
