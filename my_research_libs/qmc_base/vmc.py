"""
    my_research_libs.qmc_base.vmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Contains the basic classes and routines used to estimate the ground
    state properties of a quantum gas using the Variational Monte Carlo (VMC)
    technique.
"""
import enum
import math
import typing as t
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum

import numpy as np
from cached_property import cached_property
from numba import jit
from numpy import random as random

__all__ = [
    'CoreFuncs',
    'RandDisplaceStat',
    'Sampling',
    'CFCSpec',
    'IterProp',
    'State',
    'StateProp',
    'SamplingBatch',
    'SSFPartSlot',
    'T_E_SIter',
    'T_SIter',
    'T_SBatchesIter',
    'T_E_SBatchesIter',
    'TPFParams',
    'rand_displace'
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


@enum.unique
class SSFPartSlot(enum.IntEnum):
    """Contributions to the static structure factor."""

    #: Squared module of the Fourier density component.
    FDK_SQR_ABS = 0

    #: Real part of the Fourier density component.
    FDK_REAL = 1

    #: Imaginary part of the Fourier density component.
    FDK_IMAG = 2


class TPFParams:
    """Represent the parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    move_spread: float


class SSFParams:
    """Static structure factor parameters."""
    assume_none: bool


class SSFExecData(t.NamedTuple):
    """"""
    momenta: np.ndarray
    iter_ssf_array: np.ndarray


class CFCSpec(t.NamedTuple):
    """Represent the common spec of the core functions."""
    tpf_params: TPFParams
    ssf_params: t.Optional[SSFParams]


@enum.unique
class StateProp(str, enum.Enum):
    """The properties of a configuration."""
    WF_ABS_LOG = 'WF_ABS_LOG'
    MOVE_STAT = 'MOVE_STAT'


@enum.unique
class IterProp(str, enum.Enum):
    """The properties of a configuration."""
    WF_ABS_LOG = 'WF_ABS_LOG'
    ENERGY = 'ENERGY'
    MOVE_STAT = 'MOVE_STAT'


class State(t.NamedTuple):
    """The data yielded at every iteration of the VMC generator object."""
    sys_conf: np.ndarray
    wf_abs_log: float
    move_stat: int


class SamplingBatch(t.NamedTuple):
    """The data of the Markov chain generated by the sampling."""
    iter_props: np.ndarray
    iter_ssf: np.ndarray
    accept_rate: float
    last_state: t.Optional[State] = None


class SamplingConfsPropsBatch(t.NamedTuple):
    """The data of the Markov chain generated by the sampling."""
    confs: np.ndarray
    props: np.ndarray
    accept_rate: float
    last_state: t.Optional[State] = None


T_SIter = t.Iterator[State]
T_E_SIter = t.Iterator[t.Tuple[int, State]]
T_SBatchesIter = t.Iterator[SamplingBatch]
T_E_SBatchesIter = t.Iterator[t.Tuple[int, SamplingBatch]]


class SSFEstSpec:
    """Structure factor estimator spec."""

    #: Number of modes to evaluate the structure factor S(k).
    num_modes: int


class SamplingBase(metaclass=ABCMeta):
    """Realizes a VMC sampling.

    Defines the common parameters and methods to realize of a Variational
    Monte Carlo calculation.
    """
    __slots__ = ()

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    @property
    @abstractmethod
    def tpf_params(self) -> TPFParams:
        """Represent the parameters of the transition probability function."""
        pass

    @property
    @abstractmethod
    def ssf_params(self) -> SSFParams:
        """Represent the static structure factor parameters."""
        pass

    @property
    @abstractmethod
    def cfc_spec(self) -> CFCSpec:
        """The common spec of parameters of the core functions."""
        pass

    @property
    @abstractmethod
    def ssf_momenta(self):
        """Get the momenta to evaluate the static structure factor."""
        pass

    @abstractmethod
    def build_state(self, sys_conf: np.ndarray) -> State:
        """Builds a state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.

        :param sys_conf: The configuration of the state.
        """
        pass

    def as_chain(self, num_steps: int,
                 ini_state: State):
        """Returns the VMC sampling as an array object.

        :param num_steps: The number of states to generate.
        :param ini_state: The initial configuration of the sampling.
        """
        return self.core_funcs.as_chain(num_steps,
                                        ini_state,
                                        self.rng_seed,
                                        self.cfc_spec)

    def batches(self, num_steps_batch: int,
                ini_state: State):
        """

        :param num_steps_batch: The number of steps per batch to generate.
        :param ini_state: The initial configuration of the sampling.
        :return:
        """
        return self.core_funcs.batches(num_steps_batch,
                                       ini_state,
                                       self.rng_seed,
                                       self.cfc_spec)

    def states(self, ini_state: State) -> t.Iterator[State]:
        """Generator of VMC States.

        :param ini_state: The initial configuration of the sampling.
        """
        return self.core_funcs.generator(ini_state,
                                         self.rng_seed,
                                         self.cfc_spec)

    @property
    @abstractmethod
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        pass


class Sampling(SamplingBase, metaclass=ABCMeta):
    """Realizes a VMC sampling.

    Defines the parameters and methods to realize of a Variational Monte
    Carlo calculation. A uniform distribution is used to generate random
    numbers.
    """
    __slots__ = ()

    #: The spread magnitude of the random moves for the sampling.
    move_spread: float

    #: The seed of the pseudo-RNG used to explore the configuration space.
    rng_seed: t.Optional[int]

    #:
    ssf_est_spec: t.Optional[SSFEstSpec]


# Numpy dtype for the properties of a VMC sampling state.


states_props_dtype = np.dtype([
    (StateProp.WF_ABS_LOG.value, np.float64),
    (StateProp.MOVE_STAT.value, np.bool)
])

iter_props_dtype = np.dtype([
    (IterProp.WF_ABS_LOG.value, np.float64),
    (IterProp.ENERGY.value, np.float64),
    (IterProp.MOVE_STAT.value, np.bool)
])


# noinspection PyUnusedLocal
def _wf_abs_log_stub(sys_conf: np.ndarray,
                     cfc_spec: CFCSpec) -> float:
    """Stub for the probability density function (p.d.f.)."""
    pass


# noinspection PyUnusedLocal
def _energy_stub(sys_conf: np.ndarray,
                 cfc_spec: CFCSpec) -> float:
    """Stub for the energy function."""
    pass


# noinspection PyUnusedLocal
def _obd_pos_offset_stub(cfc_spec: CFCSpec):
    """Stub for the obd_pos_offset function."""
    pass


# noinspection PyUnusedLocal
def _one_body_density_stub(step_idx: int,
                           pos_offset: float,
                           sys_conf: np.ndarray,
                           cfc_spec: CFCSpec,
                           iter_obd_array: np.ndarray) -> float:
    """Stub for the one_body_density function."""
    pass


# noinspection PyUnusedLocal
def _ssf_momenta_stub(cfc_spec: CFCSpec) -> np.ndarray:
    """Stub for the energy function."""
    pass


# noinspection PyUnusedLocal
def _init_ssf_est_data_stub(num_steps_batch: int,
                            cfc_spec: CFCSpec) -> SSFExecData:
    """Stub for the init_ssf_est_data function."""
    pass


# noinspection PyUnusedLocal
def _fourier_density_stub(step_idx: int,
                          momenta: np.ndarray,
                          sys_conf: np.ndarray,
                          cfc_spec: CFCSpec,
                          iter_ssf_array: np.ndarray):
    """Stub for the fourier_density function."""
    pass


# noinspection PyUnusedLocal
def _ith_sys_conf_tpf_stub(i_: int, ini_sys_conf: np.ndarray,
                           prop_sys_conf: np.ndarray,
                           tpf_params: TPFParams) -> float:
    """Stub for the (i-th particle) transition probability function."""
    pass


# noinspection PyUnusedLocal
def _sys_conf_tpf_stub(ini_sys_conf: np.ndarray,
                       prop_sys_conf: np.ndarray,
                       cfc_spec: CFCSpec):
    """Stub for the transition probability function."""
    pass


@jit(nopython=True)
def rand_displace(tpf_params: TPFParams):
    """Generates a random number from a uniform distribution.

    The number lies in the half-open interval
    ``[-0.5 * move_spread, 0.5 * move_spread)``, with
    ``move_spread = spec.move_spread``.

    :param tpf_params:
    :return:
    """
    # Avoid `Untyped global name error` when executing the code in a
    # multiprocessing pool.
    rand = random.rand
    move_spread = tpf_params.move_spread
    return (rand() - 0.5) * move_spread


class CoreFuncs(metaclass=ABCMeta):
    """The core functions to realize a VMC sampling.

    These functions perform the sampling of the probability density of a QMC
    model using the Metropolis-Hastings algorithm. A uniform distribution is
    used to generate random numbers.
    """

    @property
    @abstractmethod
    def wf_abs_log(self):
        """The probability density function (p.d.f.) to sample."""
        return _wf_abs_log_stub

    @cached_property
    def rand_displace(self):
        """Generates a random number from a normal distribution."""
        return rand_displace

    @property
    @abstractmethod
    def energy(self):
        return _energy_stub

    @property
    @abstractmethod
    def one_body_density(self):
        return _one_body_density_stub

    @property
    @abstractmethod
    def init_obd_est_data(self):
        return _obd_pos_offset_stub

    @property
    @abstractmethod
    def fourier_density(self):
        return _fourier_density_stub

    @property
    @abstractmethod
    def init_ssf_est_data(self):
        return _init_ssf_est_data_stub

    @property
    @abstractmethod
    def ith_sys_conf_tpf(self):
        """The transition probability function applied to the ith particle."""
        return _ith_sys_conf_tpf_stub

    @property
    @abstractmethod
    def sys_conf_tpf(self):
        """The transition probability function."""
        return _sys_conf_tpf_stub

    @cached_property
    def generator(self):
        """VMC sampling generator.

        A generator object for the sampling configurations that follow
        the p.d.f.

        :return:
        """
        wf_abs_log = self.wf_abs_log
        sys_conf_tpf = self.sys_conf_tpf

        @jit(nopython=True, nogil=True)
        def _generator(ini_state: State,
                       rng_seed: int,
                       cfc_spec: CFCSpec):
            """VMC sampling generator.

            Generator-based sampling of the probability density function.
            The Metropolis-Hastings algorithm is used to generate the Markov
            chain. Each time a new configuration is tested, the generator
            yields it, as well as the status of the test: if the move was
            accepted, the status is ``STAT_ACCEPTED``, otherwise is
            ``STAT_REJECTED``.

            :param ini_state: The initial configuration of the particles.
            :param rng_seed: The seed used to generate the random numbers.
            :param cfc_spec: The common spec of the core functions.
            """
            # Avoid `Untyped global name error` when executing the code in a
            # multiprocessing pool.
            # Short names :)
            rand = random.rand
            log = math.log

            # Feed the numba random number generator with the given seed.
            # TODO: Handling of None seeds...
            random.seed(rng_seed)

            # Initial configuration.
            ini_sys_conf = ini_state.sys_conf
            wf_abs_log_actual = ini_state.wf_abs_log

            # Buffers
            main_conf = np.zeros_like(ini_sys_conf)
            aux_conf = np.zeros_like(ini_sys_conf)

            # Initial configuration
            actual_conf, next_conf = main_conf, aux_conf
            actual_conf[:] = ini_sys_conf[:]

            # Yield initial value.
            # TODO: Remove the sum from the expression STAT_REJECTED + 0 when
            #  numba project gets this bug resolved (version 0.42 apparently).
            #  See https://github.com/numba/numba/issues/3565
            # yield State(actual_conf, wf_abs_log_actual, STAT_REJECTED + 0)
            yield ini_state

            # Iterate indefinitely.
            while True:

                # Just keep advancing...
                sys_conf_tpf(actual_conf, next_conf, cfc_spec)
                wf_abs_log_next = wf_abs_log(next_conf, cfc_spec)
                move_stat = STAT_REJECTED

                # Metropolis condition
                if wf_abs_log_next > 0.5 * log(rand()) + wf_abs_log_actual:
                    # Accept the movement
                    # main_conf[:] = aux_conf[:]
                    actual_conf, next_conf = next_conf, actual_conf
                    wf_abs_log_actual = wf_abs_log_next
                    move_stat = STAT_ACCEPTED

                # NOTICE: Using a tuple creates a performance hit?
                yield State(actual_conf, wf_abs_log_actual, move_stat)

        return _generator

    @cached_property
    def batches(self):
        """Returns the VMC sampling as an array object.

        JIT-compiled function to generate a Markov chain with the
        sampling of the probability density function.

        :return: The JIT compiled function that execute the Monte Carlo
            integration.
        """
        wf_abs_log_field = IterProp.WF_ABS_LOG.value
        energy_field = IterProp.ENERGY.value
        move_stat_field = IterProp.MOVE_STAT.value

        generator = self.generator
        energy_func = self.energy

        init_ssf_est_data = self.init_ssf_est_data
        fourier_density = self.fourier_density

        @jit(nopython=True, nogil=True)
        def _batches(num_steps_batch: int,
                     ini_state: State,
                     rng_seed: int,
                     cfc_spec: CFCSpec):
            """Returns the VMC sampling batches of states configurations.

            :param num_steps_batch: The number of steps per batch.
            :param ini_state: The initial configuration of the particles.
            :param rng_seed: The seed used to generate the random numbers.
            :param cfc_spec:
            :return: An array with the Markov chain configurations, the values
                of the p.d.f. and the acceptance rate.
            """
            ssf_params = cfc_spec.ssf_params
            ns_batch = num_steps_batch

            ipb_shape = ns_batch,

            # Array to store the main properties.
            iter_props_array = np.empty(ipb_shape, dtype=iter_props_dtype)
            wf_abs_log_set = iter_props_array[wf_abs_log_field]
            energy_set = iter_props_array[energy_field]
            move_stat_set = iter_props_array[move_stat_field]

            # Static structure factor arrays.
            ssf_est_data = init_ssf_est_data(num_steps_batch, cfc_spec)
            ssf_momenta = ssf_est_data.momenta
            iter_ssf_array = ssf_est_data.iter_ssf_array

            sampling_iter = generator(ini_state, rng_seed, cfc_spec)

            # Initial state.
            state = ini_state

            # Yield batches indefinitely.
            while True:

                # We use an initial index instead enumerate.
                step_idx = 0

                # Reset accepted counter.
                accepted = 0.

                for state in sampling_iter:

                    # This loop with start in the last generated state.
                    # Metropolis-Hastings iterator.
                    # TODO: Test use of next() function.
                    sys_conf, wf_abs_log, move_stat = state
                    wf_abs_log_set[step_idx] = wf_abs_log
                    move_stat_set[step_idx] = bool(move_stat)
                    accepted += move_stat

                    # Calculate the energy. Necessary for the optimization
                    # process.
                    energy = energy_func(sys_conf, cfc_spec)
                    energy_set[step_idx] = energy

                    # Evaluate the Static Structure Factor.
                    if not ssf_params.assume_none:
                        fourier_density(step_idx, ssf_momenta,
                                        sys_conf, cfc_spec,
                                        iter_ssf_array)

                    # Yield the batch just before generating a new state 🤔.
                    if step_idx + 1 >= ns_batch:
                        # Get the acceptance rate.
                        break

                    # Counter goes up 🙂.
                    step_idx += 1

                accept_rate = accepted / ns_batch

                # NOTE: We yield references to the internal buffer.
                last_state = State(state.sys_conf,
                                   state.wf_abs_log,
                                   state.move_stat)

                yield SamplingBatch(iter_props_array,
                                    iter_ssf_array,
                                    accept_rate,
                                    last_state=last_state)

        return _batches

    @cached_property
    def as_chain(self):
        """Returns the VMC sampling as an array object.

        JIT-compiled function to generate a Markov chain with the
        sampling of the probability density function.

        :return: The JIT compiled function that execute the Monte Carlo
            integration.
        """
        confs_props_batches = self.confs_props_batches

        @jit(nopython=True, nogil=True)
        def _as_chain(num_steps: int,
                      ini_state: State,
                      rng_seed: int,
                      cfc_spec: CFCSpec) -> SamplingConfsPropsBatch:
            """Returns the VMC sampling as a single batch.

            :return:
            :param num_steps: The number of samples of the Markov chain.
            :param ini_state: The initial configuration of the particles.
            :param rng_seed: The seed used to generate the random numbers.
            :param cfc_spec:
            :return: An array with the Markov chain configurations, the values
                of the p.d.f. and the acceptance rate.
            """
            num_steps_batch = num_steps

            # Return the only batch as the result.
            return next(confs_props_batches(num_steps_batch,
                                            ini_state, rng_seed,
                                            cfc_spec))

        return _as_chain

    @cached_property
    def confs_props_batches(self):
        """Returns the VMC sampling as an array object.

        JIT-compiled function to generate a Markov chain with the
        sampling of the probability density function.

        :return: The JIT compiled function that execute the Monte Carlo
            integration.
        """
        wf_abs_log_field = StateProp.WF_ABS_LOG.value
        move_stat_field = StateProp.MOVE_STAT.value
        generator = self.generator

        @jit(nopython=True, nogil=True)
        def _confs_props_batches(num_steps_batch: int,
                                 ini_state: State,
                                 rng_seed: int,
                                 cfc_spec: CFCSpec):
            """Returns the VMC sampling batches of states configurations.

            :param num_steps_batch: The number of steps per batch.
            :param ini_state: The initial configuration of the particles.
            :param rng_seed: The seed used to generate the random numbers.
            :param cfc_spec:
            :return: An array with the Markov chain configurations, the values
                of the p.d.f. and the acceptance rate.
            """
            # Check for invalid parameters 🤔.
            if not num_steps_batch >= 1:
                raise ValueError('num_steps_batch must be nonzero and '
                                 'positive')

            scb_shape = (num_steps_batch,) + ini_state.sys_conf.shape
            states_confs = np.zeros(scb_shape, dtype=np.float64)
            states_props = np.empty(num_steps_batch, dtype=states_props_dtype)

            wf_abs_log_set = states_props[wf_abs_log_field]
            move_stat_set = states_props[move_stat_field]

            accepted = 0
            sampling_iter = generator(ini_state, rng_seed, cfc_spec)

            # Enumerating the sampling iterator.
            enum_iter: T_E_SIter = enumerate(sampling_iter)

            # Yield batches indefinitely.
            for cj_count, state in enum_iter:

                # Get the correct cj_ index from cj_count.
                cj_ = cj_count % num_steps_batch

                # This loop with start in the last generated state.
                # Metropolis-Hastings iterator.
                # TODO: Test use of next() function.
                sys_conf, wf_abs_log, move_stat = state
                states_confs[cj_, :] = sys_conf[:]
                wf_abs_log_set[cj_] = wf_abs_log
                move_stat_set[cj_] = bool(move_stat)
                accepted += move_stat

                # Keep a reference to the last state.
                tmp_state = state

                # Yield the batch just before generating a new state 🤔.
                if cj_ + 1 >= num_steps_batch:
                    # Get the acceptance rate.
                    accept_rate = accepted / num_steps_batch

                    # NOTE: We yield references to the internal buffer.
                    #  Delegate to the caller the choice to make any
                    #  copies of it.
                    last_state = State(tmp_state.sys_conf,
                                       tmp_state.wf_abs_log,
                                       tmp_state.move_stat)

                    batch_data = \
                        SamplingConfsPropsBatch(states_confs,
                                                states_props,
                                                accept_rate,
                                                last_state)
                    yield batch_data

                    # Reset accepted counter.
                    accepted = 0

        return _confs_props_batches
