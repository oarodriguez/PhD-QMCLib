"""
    my_research_libs.qmc_base.vmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Contains the basic classes and routines used to estimate the ground
    state properties of a quantum gas using the Variational Monte Carlo (VMC)
    technique.
"""

import math
from abc import ABCMeta, abstractmethod
from collections import Iterable
from enum import Enum, IntEnum, unique
from typing import Any, Mapping, Tuple

import numpy as np
from numba import jit
from numpy import random as random

from my_research_libs.utils import Cached, CachedMeta, cached_property
from . import model

__all__ = [
    'PBCSampling',
    'PBCUniformSampling',
    'RandDisplaceStat',
    'Sampling',
    'SamplingMeta',
    'SamplingParams',
    'UniformSampling',
    'UniformSamplingParams'
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


@unique
class SamplingParam(model.ParamNameEnum):
    """"""
    TIME_STEP = 'time_step'
    INI_SYS_CONF = 'ini_sys_conf'
    CHAIN_SAMPLES = 'chain_samples'
    BURN_IN_SAMPLES = 'burn_in_samples'
    RNG_SEED = 'rng_seed'


class SamplingParamDefault(Enum):
    """"""
    BURN_IN_SAMPLES = 0
    RNG_SEED = None


class SamplingParams(model.ParamsSet):
    """"""
    names = SamplingParam
    defaults = SamplingParamDefault


class SamplingMeta(CachedMeta):
    """Metaclass for :class:`Sampling` abstract base class."""
    pass


class Sampling(Iterable, Cached, metaclass=SamplingMeta):
    """The interface to realize a VMC sampling.

    Performs the sampling of the probability density of a QMC model
    using the Metropolis-Hastings algorithm. It uses a normal distribution
    to generate random numbers.
    """

    # The sampling parameters.
    params_cls = SamplingParams

    def __init__(self, params: Mapping[str, float]):
        """

        :param params:
        """
        # Dicts remember insertion order in Python 3.7 and beyond.
        super().__init__()
        self._params = self.params_cls(params)

    @property
    def params(self):
        """"""
        # Return a shallow copy
        return self._params

    def update_params(self, params: Mapping):
        """

        :param params:
        :return:
        """
        self._params = self.params_cls(self.params, **params)

    @property
    @abstractmethod
    def wf_abs_log(self):
        """The probability density function (p.d.f.) to sample."""
        pass

    @property
    @abstractmethod
    def ppf_args(self):
        """The set of parameters for the transition proposal
        probability function.
        """
        pass

    @property
    @abstractmethod
    def sys_conf_ppf(self):
        """The transition proposal probability function."""
        pass

    @cached_property
    def rand_displace(self):
        """Generates a random number from a normal distribution."""

        @jit(nopython=True, cache=True)
        def _rand_displace(sigma: float):
            """Generates a random number from a normal distribution with
            zero mean and a a standard deviation ``sigma``.
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
            return normal(0, sigma)

        return _rand_displace

    @property
    def args(self):
        """"""
        return tuple(self.params.values())

    @property
    @abstractmethod
    def gen_args(self):
        """

        :return:
        """
        pass

    @cached_property
    def generator(self):
        """A generator object for the sampling configurations that follow
        the p.d.f.

        :return:
        """
        wf_abs_log = self.wf_abs_log
        sys_conf_ppf = self.sys_conf_ppf

        @jit(nopython=True, cache=True, nogil=True)
        def _generator(wf_args: Tuple[Any, ...],
                       ppf_args: Tuple[Any, ...],
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

            :param wf_args: A tuple with the parameters needed to
                evaluate the probability density function.
            :param ppf_args: A tuple with the parameters that control
                the random configurations used to sample the p.d.f.
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
            wf_abs_log_actual: float = wf_abs_log(actual_conf, *wf_args)

            if not bis:
                # Yield initial value.
                yield actual_conf, wf_abs_log_actual, STAT_REJECTED

            for cj_ in range(cj_ini, cj_end):

                # Just keep advancing...
                sys_conf_ppf(actual_conf, next_conf, ppf_args)
                wf_abs_log_next: float = wf_abs_log(next_conf, *wf_args)
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
                    yield actual_conf, wf_abs_log_actual, move_stat

        return _generator

    @cached_property
    def _as_chain(self):
        """JIT-compiled function to generate a Markov chain with the
        sampling of the probability density function.

        :return: The JIT compiled function that execute the Monte Carlo
            integration.
        """
        generator = self.generator

        @jit(nopython=True, cache=True, nogil=True)
        def __as_chain(wf_args: Tuple[Any, ...],
                       ppf_args: Tuple[Any, ...],
                       ini_sys_conf: np.ndarray,
                       chain_samples: int,
                       burn_in_samples: int,
                       rng_seed: int):
            """Routine to samples the probability density function.

            :param wf_args: A tuple with the parameters needed to
                evaluate the probability density function.
            :param ppf_args: A tuple with the parameters that control
                the random configurations used to sample the p.d.f.
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
            sampling_iter = generator(wf_args,
                                      ppf_args,
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
            return sys_conf_chain, wf_abs_log_chain, accept_rate

        return __as_chain

    def as_chain(self):
        """Returns the sampling results as a whole."""
        return self._as_chain(*self.gen_args)

    def __iter__(self):
        """"""
        return self.generator(*self.gen_args)


class PBCSampling(Sampling, metaclass=ABCMeta):
    """Performs the sampling of the probability density of a QMC model using
    the Metropolis-Hastings algorithm, in a configuration space subject to
    periodic boundary conditions (PBC). It uses a normal distribution to
    generate random numbers.
    """

    # For a pdf with periodic boundary conditions.
    @property
    @abstractmethod
    def recast(self):
        """"""
        pass


@unique
class UniformSamplingParam(model.ParamNameEnum):
    """The parameters to realize a UniformSampling."""
    MOVE_SPREAD = 'move_spread'
    INI_SYS_CONF = 'ini_sys_conf'
    CHAIN_SAMPLES = 'chain_samples'
    BURN_IN_SAMPLES = 'burn_in_samples'
    RNG_SEED = 'rng_seed'


class UniformSamplingParamDefault(Enum):
    """"""
    BURN_IN_SAMPLES = 0
    RNG_SEED = None


class UniformSamplingParams(SamplingParams):
    """"""
    names = UniformSamplingParam
    defaults = UniformSamplingParamDefault


class UniformSampling(Sampling, metaclass=ABCMeta):
    """Performs the sampling of the probability density of a QMC model
    using the Metropolis-Hastings algorithm. It uses a uniform distribution
    to generate random numbers.
    """
    #
    params_cls = UniformSamplingParams

    @cached_property
    def rand_displace(self):
        """Generates a random number from a uniform distribution."""

        @jit(nopython=True, cache=True)
        def _rand_displace(move_spread: float):
            """Generates a random number from a uniform distribution in the
            half-open interval ``[-0.5 * move_spread, 0.5 * move_spread)``.

            :param move_spread:
            :return:
            """
            # Avoid `Untyped global name error` when executing the code in a
            # multiprocessing pool.
            rand = random.rand
            return (rand() - 0.5) * move_spread

        return _rand_displace


class PBCUniformSampling(UniformSampling, PBCSampling, metaclass=ABCMeta):
    """Performs the sampling of the probability density of a QMC model using
    the Metropolis-Hastings algorithm, in a configuration space subject to
    periodic boundary conditions (PBC). It uses a uniform distribution to
    generate random numbers.
    """
    # NOTE: Maybe this class is not necessary...
    pass
