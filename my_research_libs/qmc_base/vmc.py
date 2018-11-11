"""
    my_research_libs.qmc_base.vmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Contains the basic classes and routines used to estimate the ground
    state properties of a quantum gas using the Variational Monte Carlo (VMC)
    technique.
"""

import math
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum
from typing import NamedTuple, Union

import numpy as np
from cached_property import cached_property
from numba import jit
from numpy import random as random

__all__ = [
    'RandDisplaceStat',
    'SamplingFuncs',
    'SamplingMeta',
    'TPFSpecNT',
    'UniformSamplingFuncs',
    'UTPFSpecNT',
    'WFSpecNT',
]


class PPFType(str, Enum):
    """The type of """
    UNIFORM = 'uniform'
    GAUSSIAN = 'gaussian'


PPF_UNIFORM = PPFType.UNIFORM
PPF_GAUSSIAN = PPFType.GAUSSIAN


class RandDisplaceStat(IntEnum):
    """"""
    REJECTED = 0
    ACCEPTED = 1


STAT_REJECTED = int(RandDisplaceStat.REJECTED)
STAT_ACCEPTED = int(RandDisplaceStat.ACCEPTED)


class WFSpecNT(NamedTuple):
    """The parameters of the trial wave function.

    We declare this class to help with typing and nothing more. A concrete
    spec should be implemented for every concrete model. It is recommended
    to inherit from this class to keep a logical sequence in the code.
    """
    pass


class TPFSpecNT(NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a normal (gaussian) distribution function.
    """
    time_step: float


class UTPFSpecNT(NamedTuple):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    move_spread: float


class SpecNT(NamedTuple):
    """The parameters to realize a sampling."""
    wf_spec: WFSpecNT
    tpf_spec: Union[TPFSpecNT, UTPFSpecNT]
    ini_sys_conf: np.ndarray
    chain_samples: int
    burn_in_samples: int
    rng_seed: int


class SamplingIterDataNT(NamedTuple):
    """The data yielded at every iteration of the VMC generator object."""
    sys_conf: np.ndarray
    wf_abs_log: float
    move_stat: int


class SamplingChainNT(NamedTuple):
    """The data of the Markov chain generated by the sampling."""
    sys_conf_chain: np.ndarray
    wf_abs_log_chain: np.ndarray
    accept_rate: float


# noinspection PyUnusedLocal
def _wf_abs_log_stub(sys_conf: np.ndarray, spec: WFSpecNT) -> float:
    """Stub for the probability density function (p.d.f.)."""
    pass


# noinspection PyUnusedLocal
def _ith_sys_conf_tpf_stub(i_: int, ini_sys_conf: np.ndarray,
                           prop_sys_conf: np.ndarray,
                           tpf_spec: Union[TPFSpecNT, UTPFSpecNT]) -> float:
    """Stub for the (i-th particle) transition probability function."""
    pass


# noinspection PyUnusedLocal
def _sys_conf_tpf_stub(ini_sys_conf: np.ndarray,
                       prop_sys_conf: np.ndarray,
                       tpf_spec: Union[TPFSpecNT, UTPFSpecNT]):
    """Stub for the transition probability function."""
    pass


class SamplingMeta(ABCMeta):
    """Metaclass for :class:`SamplingFuncs` abstract base class."""
    pass


class SamplingFuncs(metaclass=SamplingMeta):
    """The core functions to realize a VMC sampling.

    These functions perform the sampling of the probability density of a QMC
    model using the Metropolis-Hastings algorithm. A normal distribution
    to generate random numbers.
    """

    @property
    @abstractmethod
    def wf_abs_log(self):
        """The probability density function (p.d.f.) to sample."""
        pass

    @cached_property
    def rand_displace(self):
        """Generates a random number from a normal distribution."""

        @jit(nopython=True, cache=True)
        def _rand_displace(tpf_spec: TPFSpecNT):
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
            sigma = tpf_spec.time_step
            return normal(0, sigma)

        return _rand_displace

    @property
    @abstractmethod
    def ith_sys_conf_tpf(self):
        """The transition probability function applied to the ith particle."""
        pass

    @property
    @abstractmethod
    def sys_conf_tpf(self):
        """The transition probability function."""
        pass

    @cached_property
    def generator(self):
        """A generator object for the sampling configurations that follow
        the p.d.f.

        :return:
        """
        wf_abs_log = self.wf_abs_log
        sys_conf_tpf = self.sys_conf_tpf

        @jit(nopython=True, cache=True, nogil=True)
        def _generator(wf_spec: WFSpecNT,
                       tpf_spec: Union[TPFSpecNT, UTPFSpecNT],
                       ini_sys_conf: np.ndarray,
                       chain_samples: int,
                       burn_in_samples: int,
                       rng_seed: int):
            """Generator-based sampling of the probability density
            function. The Metropolis-Hastings algorithm is used to
            generate the Markov chain. Each time a new configuration is
            tested, the generator yields it, as well as the status of the
            test: if the move was accepted, the status is ``STAT_ACCEPTED``,
            otherwise is ``STAT_REJECTED``.

            :param wf_spec: A tuple with the parameters needed to
                evaluate the probability density function.
            :param tpf_spec: The parameters of the transition probability
                function.
            :param ini_sys_conf: The buffer to store the positions and
                drift velocities of the particles. It should contain
                the initial configuration of the particles.
            :param chain_samples: The maximum number of samples of the
                Markov chain.
            :param burn_in_samples: The number of initial samples to discard.
            :param rng_seed: The seed used to generate the random numbers.
            """
            # Do not allow invalid parameters.
            if not chain_samples >= 1:
                raise ValueError('chain_samples must be nonzero and positive')

            if not burn_in_samples >= 0:
                raise ValueError('burn_in_samples must be zero or positive')

            ncs = chain_samples
            bis = burn_in_samples

            # Avoid `Untyped global name error` when executing the code in a
            # multiprocessing pool.
            rand = random.rand
            log = math.log

            # Feed the numba random number generator with the given seed.
            random.seed(rng_seed)

            # Buffers
            main_conf = np.zeros_like(ini_sys_conf)
            aux_conf = np.zeros_like(ini_sys_conf)

            # Initial configuration
            actual_conf, next_conf = main_conf, aux_conf
            actual_conf[:] = ini_sys_conf[:]

            # Initial value of the p.d.f. and loop
            cj_ini, cj_end = 1, bis + ncs
            wf_abs_log_actual = wf_abs_log(actual_conf, wf_spec)

            if not bis:
                # Yield initial value.
                yield SamplingIterDataNT(actual_conf,
                                         wf_abs_log_actual,
                                         STAT_REJECTED)

            for cj_ in range(cj_ini, cj_end):

                # Just keep advancing...
                sys_conf_tpf(actual_conf, next_conf, tpf_spec)
                wf_abs_log_next = wf_abs_log(next_conf, wf_spec)
                move_stat = STAT_REJECTED

                # Metropolis condition
                if wf_abs_log_next > 0.5 * log(rand()) + wf_abs_log_actual:
                    # Accept the movement
                    # main_conf[:] = aux_conf[:]
                    actual_conf, next_conf = next_conf, actual_conf
                    wf_abs_log_actual = wf_abs_log_next
                    move_stat = STAT_ACCEPTED

                if cj_ < bis:
                    continue
                else:
                    # NOTICE: Using a tuple creates a performance hit?
                    yield SamplingIterDataNT(actual_conf,
                                             wf_abs_log_actual,
                                             move_stat)

        return _generator

    @cached_property
    def as_chain(self):
        """JIT-compiled function to generate a Markov chain with the
        sampling of the probability density function.

        :return: The JIT compiled function that execute the Monte Carlo
            integration.
        """
        generator = self.generator

        @jit(nopython=True, cache=True, nogil=True)
        def _as_chain(wf_spec: WFSpecNT,
                      tpf_spec: Union[TPFSpecNT, UTPFSpecNT],
                      ini_sys_conf: np.ndarray,
                      chain_samples: int,
                      burn_in_samples: int,
                      rng_seed: int):
            """Routine to samples the probability density function.

            :param wf_spec: A tuple with the parameters needed to
                evaluate the probability density function.
            :param tpf_spec: The parameters of the transition probability
                function.
            :param ini_sys_conf: The buffer to store the positions and
                drift velocities of the particles. It should contain
                the initial configuration of the particles.
            :param chain_samples: The maximum number of samples of the
                Markov chain.
            :param burn_in_samples: The number of initial samples to discard.
            :param rng_seed: The seed used to generate the random numbers.
            :return: An array with the Markov chain configurations, the values
                of the p.d.f. and the acceptance rate.
            """
            sampling_iter = generator(wf_spec,
                                      tpf_spec,
                                      ini_sys_conf,
                                      chain_samples,
                                      burn_in_samples,
                                      rng_seed)

            # TODO: What is better: allocate or pass a pre-allocated buffer?
            mcs = (chain_samples,) + ini_sys_conf.shape
            sys_conf_chain = np.zeros(mcs, dtype=np.float64)
            wf_abs_log_chain = np.zeros(chain_samples, dtype=np.float64)
            accepted = 0

            for cj_, iter_values in enumerate(sampling_iter):
                # Metropolis-Hastings iterator.
                # TODO: Test use of next() function.
                sys_conf, pdf_log, move_stat = iter_values
                sys_conf_chain[cj_, :] = sys_conf[:]
                wf_abs_log_chain[cj_] = pdf_log
                accepted += move_stat

            # TODO: Should we account burnout and transient moves?
            accept_rate = accepted / chain_samples
            return SamplingChainNT(sys_conf_chain,
                                   wf_abs_log_chain,
                                   accept_rate)

        return _as_chain


class UniformSamplingFuncs(SamplingFuncs, metaclass=ABCMeta):
    """The core functions to realize a VMC sampling.

    These functions perform the sampling of the probability density of a QMC
    model using the Metropolis-Hastings algorithm. A uniform distribution is
    used to generate random numbers.
    """

    @cached_property
    def rand_displace(self):
        """Generates a random number from a uniform distribution."""

        @jit(nopython=True, cache=True)
        def _rand_displace(tpf_spec: UTPFSpecNT):
            """Generates a random number from a uniform distribution.

            The number lies in the half-open interval
            ``[-0.5 * move_spread, 0.5 * move_spread)``, with
            ``move_spread = spec.move_spread``.

            :param tpf_spec:
            :return:
            """
            # Avoid `Untyped global name error` when executing the code in a
            # multiprocessing pool.
            rand = random.rand
            move_spread = tpf_spec.move_spread
            return (rand() - 0.5) * move_spread

        return _rand_displace
