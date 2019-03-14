"""
    my_research_libs.qmc_base.dmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements the main Diffusion Monte Carlo classes and routines.
"""

import enum
import typing as t
from abc import ABCMeta, abstractmethod
from math import log

import numba as nb
import numpy as np
from cached_property import cached_property
from numpy import random

CONF_INDEX = 0
ENERGY_INDEX = 1


class DDFParams:
    """The parameters of the diffusion-and-drift process."""
    time_step: float
    sigma_spread: float


class SSFParams:
    """Static structure factor parameters."""
    assume_none: bool


class SSFExecData(t.NamedTuple):
    """"""
    momenta: np.ndarray
    iter_ssf_array: np.ndarray
    pfw_aux_ssf_array: np.ndarray


class CFCSpec(t.NamedTuple):
    """Represent the common spec of the core functions."""
    ssf_params: t.Optional[SSFParams]


@enum.unique
class StateProp(str, enum.Enum):
    """The properties of a configuration."""
    ENERGY = 'ENERGY'
    WEIGHT = 'WEIGHT'
    MASK = 'MASK'


@enum.unique
class IterProp(str, enum.Enum):
    """"""
    ENERGY = 'ENERGY'
    WEIGHT = 'WEIGHT'
    NUM_WALKERS = 'NUM_WALKERS'
    REF_ENERGY = 'REF_ENERGY'
    ACCUM_ENERGY = 'ACCUM_ENERGY'


@enum.unique
class SSFPartSlot(enum.IntEnum):
    """Contributions to the static structure factor."""

    #: Squared module of the Fourier density component.
    FDK_SQR_ABS = 0

    #: Real part of the Fourier density component.
    FDK_REAL = 1

    #: Imaginary part of the Fourier density component.
    FDK_IMAG = 2


@enum.unique
class BranchingSpecField(str, enum.Enum):
    """The fields of a branching spec."""
    CLONING_FACTOR = 'CLONING_FACTOR'
    CLONING_REF = 'CLONING_REF'
    MASK = 'MASK'


class State(t.NamedTuple):
    """The """
    confs: np.ndarray
    props: np.ndarray
    energy: float
    weight: float
    num_walkers: int
    ref_energy: float
    accum_energy: float
    max_num_walkers: int
    branching_spec: t.Optional[np.ndarray] = None


class SamplingConfsPropsBatch(t.NamedTuple):
    """"""
    states_confs: np.ndarray
    states_props: np.ndarray
    iter_props: np.ndarray


T_SIter = t.Iterator[State]
T_E_SIter = t.Iterator[t.Tuple[int, State]]
T_SCPBatchesIter = t.Iterator[SamplingConfsPropsBatch]
T_E_SCPBatchesIter = t.Iterator[t.Tuple[int, SamplingConfsPropsBatch]]


class SamplingBatch(t.NamedTuple):
    """"""
    #: Properties data.
    iter_props: np.ndarray
    iter_ssf: np.ndarray = None
    last_state: t.Optional[State] = None


class EstAuxData(t.NamedTuple):
    """"""
    #: Properties data.
    structure_factor: np.ndarray = None


# Variables for type-hints.
T_SBatchesIter = t.Iterator[SamplingBatch]
T_E_SBatchesIter = t.Iterator[t.Tuple[int, SamplingBatch]]


# noinspection PyUnusedLocal
@nb.jit(nopython=True)
def dummy_pure_est_core_func(step_idx: int,
                             sys_idx: int,
                             clone_ref_idx: int,
                             state_confs: np.ndarray,
                             iter_sf_array: np.ndarray,
                             aux_states_sf_array: np.ndarray):
    """Dummy pure estimator core function."""
    return


class SSFEstSpecNT(t.NamedTuple):
    """"""

    #: Number of modes to evaluate the structure factor S(k).
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: int = None
    core_func: t.Callable = dummy_pure_est_core_func


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator."""

    #: Number of modes to evaluate the structure factor S(k).
    num_modes: int

    #: Number of steps for the forward sampling of the pure estimator.
    as_pure_est: bool = False

    #: Number of time steps of the forward walking to accumulate the pure
    # estimator of S(k).
    pfw_num_time_steps: int = 512

    # @property
    # @abstractmethod
    # def momenta(self, *args, **kwargs):
    #     """"""
    #     pass

    # @property
    # @abstractmethod
    # def core_func(self, *args, **kwargs):
    #     """"""
    #     pass


class Sampling(metaclass=ABCMeta):
    """Realizes a DMC sampling.

    Defines the parameters and related properties of a Diffusion Monte
    Carlo calculation, as well as the evaluation of expected values
    (estimators).
    """
    __slots__ = ()

    #: The "time-step" (squared, average move spread) of the sampling.
    time_step: float

    #: The maximum wight of the population of walkers.
    max_num_walkers: int

    #: The average total weight of the population of walkers.
    target_num_walkers: int

    #: Multiplier for the population control during the branching stage.
    num_walkers_control_factor: float

    #: The seed of the pseudo-RNG used to realize the sampling.
    rng_seed: int

    # *** *** Configuration of the estimators. *** ***

    #: Spec of the static structure factor estimator.
    ssf_est_spec: t.Optional[SSFEstSpec]

    #: Parallel execution where possible.
    jit_parallel: bool

    #: Use fastmath compiler directive.
    # NOTE: Not sure how useful it could be.
    jit_fastmath: bool

    @property
    @abstractmethod
    def ddf_params(self) -> DDFParams:
        """Represent the diffusion-and-drift process parameters."""
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

    @property
    @abstractmethod
    def state_confs_shape(self):
        pass

    @property
    def state_props_shape(self):
        """"""
        max_num_walkers = self.max_num_walkers
        return max_num_walkers,

    @abstractmethod
    def build_state(self, sys_conf_set: np.ndarray,
                    ref_energy: float) -> State:
        """Builds a state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.

        :param sys_conf_set: The initial configuration set of the
            sampling.
        :param ref_energy: The initial energy of reference.
        """
        pass

    @property
    @abstractmethod
    def core_funcs(self) -> 'CoreFuncs':
        """The sampling core functions."""
        pass

    def batches(self, ini_state: State,
                num_time_steps_batch: int) -> T_SBatchesIter:
        """Generator of batches of states.

        :param ini_state: The initial state of the sampling.
        :param num_time_steps_batch:
        :return:
        """
        time_step = self.time_step
        target_num_walkers = self.target_num_walkers
        return self.core_funcs.batches(time_step,
                                       ini_state,
                                       num_time_steps_batch,
                                       target_num_walkers,
                                       self.rng_seed,
                                       self.cfc_spec)

    def confs_props_batches(self, ini_state: State,
                            num_time_steps_batch: int) -> T_SCPBatchesIter:
        """Generator of batches of states.

        :param ini_state: The initial state of the sampling.
        :param num_time_steps_batch:
        :return:
        """
        time_step = self.time_step
        target_num_walkers = self.target_num_walkers
        rng_seed = self.rng_seed
        return self.core_funcs.confs_props_batches(time_step,
                                                   ini_state,
                                                   num_time_steps_batch,
                                                   target_num_walkers,
                                                   rng_seed)

    def states(self, ini_state: State) -> T_SIter:
        """Generator object that yields DMC states.

        :param ini_state: The initial state of the sampling.
        :return:
        """
        time_step = self.time_step
        target_num_walkers = self.target_num_walkers
        return self.core_funcs.states_generator(time_step,
                                                ini_state,
                                                target_num_walkers,
                                                self.rng_seed,
                                                self.cfc_spec)


iter_props_dtype = np.dtype([
    (IterProp.ENERGY.value, np.float64),
    (IterProp.WEIGHT.value, np.float64),
    (IterProp.NUM_WALKERS.value, np.uint32),
    (IterProp.REF_ENERGY.value, np.float64),
    (IterProp.ACCUM_ENERGY.value, np.float64)
])

branching_spec_dtype = np.dtype([
    (BranchingSpecField.CLONING_FACTOR.value, np.int32),
    (BranchingSpecField.CLONING_REF.value, np.int32)
])


# noinspection PyUnusedLocal
def _recast_stub(z: float, ddf_params: DDFParams):
    """Stub for the recast function."""
    pass


# noinspection PyUnusedLocal
def _fourier_density_stub(step_idx: int,
                          momenta: np.ndarray,
                          state_confs: np.ndarray,
                          num_walkers: int,
                          max_num_walkers: int,
                          branching_spec: np.ndarray,
                          cfc_spec: CFCSpec,
                          iter_ssf_array: np.ndarray,
                          aux_states_sf_array: np.ndarray):
    """Stub for the fourier density function (p.d.f.)."""
    pass


# noinspection PyUnusedLocal
def _init_ssf_est_data_stub(num_time_steps_batch: int,
                            max_num_walkers: int,
                            cfc_spec: CFCSpec) -> SSFExecData:
    """Stub for the energy function (p.d.f.)."""
    pass


# noinspection PyUnusedLocal
def _evolve_state_stub(prev_state_confs: np.ndarray,
                       prev_state_props: np.ndarray,
                       actual_state_confs: np.ndarray,
                       actual_state_props: np.ndarray,
                       aux_next_state_confs: np.ndarray,
                       aux_next_state_props: np.ndarray,
                       actual_num_walkers: int,
                       max_num_walkers: int,
                       time_step: float,
                       ref_energy: float,
                       branching_spec: np.ndarray,
                       cfc_spec: CFCSpec):
    """Stub for the evolve_state function."""
    pass


class CoreFuncs(metaclass=ABCMeta):
    """The core functions for a DMC sampling."""

    __slots__ = ()

    #: Parallel the execution where possible.
    jit_parallel: bool

    #: Use fastmath compiler directive.
    jit_fastmath: bool

    @property
    @abstractmethod
    def recast(self):
        return _recast_stub

    @property
    @abstractmethod
    def fourier_density(self):
        return _fourier_density_stub

    @property
    @abstractmethod
    def init_ssf_est_data(self):
        return _init_ssf_est_data_stub

    @cached_property
    def sync_branching_spec(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        props_weight_field = StateProp.WEIGHT.value
        clone_ref_field = BranchingSpecField.CLONING_REF.value

        @nb.jit(nopython=True, fastmath=fastmath)
        def _sync_branching_spec(prev_state_props: np.ndarray,
                                 prev_num_walkers: int,
                                 max_num_walkers: int,
                                 branching_spec: np.ndarray):
            """

            :param branching_spec:
            :return:
            """
            prev_state_weights = prev_state_props[props_weight_field]
            cloning_refs = branching_spec[clone_ref_field]

            final_num_walkers = 0
            for sys_idx in range(prev_num_walkers):
                # NOTE: num_walkers cannot be greater, right?
                if final_num_walkers >= max_num_walkers:
                    break
                # Cloning factor of the previous walker.
                sys_weight = prev_state_weights[sys_idx]
                clone_factor = int(sys_weight + random.rand())
                if not clone_factor:
                    continue
                prev_num_walkers = final_num_walkers
                final_num_walkers = \
                    min(max_num_walkers, final_num_walkers + clone_factor)
                # We will copy the system with index sys_idx as many
                # times as clone_factor (only limited by the maximum number
                # of walkers).
                cloning_refs[prev_num_walkers:final_num_walkers] = sys_idx

            return final_num_walkers

        return _sync_branching_spec

    @property
    @abstractmethod
    def evolve_state(self):
        """"""
        return _evolve_state_stub

    @cached_property
    def states_generator(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        props_energy_field = StateProp.ENERGY.value
        props_weight_field = StateProp.WEIGHT.value

        # JIT functions.
        evolve_state = self.evolve_state
        sync_branching_spec = self.sync_branching_spec

        @nb.jit(nopython=True, fastmath=fastmath)
        def _states_generator(time_step: float,
                              ini_state: State,
                              target_num_walkers: int,
                              rng_seed: int,
                              cfc_spec: CFCSpec):
            """Realizes the DMC sampling state-by-state.

            The sampling is done in batches, with each batch having a fixed
            number of time steps given by the ``num_time_steps_batch``
            argument.

            :param ini_state:
            :param time_step:
            :param target_num_walkers:
            :param rng_seed:
            :param cfc_spec: The common spec to pass to the core functions
                of the model.
            :return:
            """
            # The initial state fixes the arrays of the following states.
            ini_state_confs = ini_state.confs
            ini_state_props = ini_state.props
            max_num_walkers = ini_state.max_num_walkers

            # Configurations and properties of the current
            # state of the sampling (the one that will be yielded).
            state_confs = np.zeros_like(ini_state_confs)
            state_props = np.zeros_like(ini_state_props)

            # Table to control the branching process.
            state_branching_spec = \
                np.zeros(max_num_walkers, dtype=branching_spec_dtype)

            # Auxiliary arrays for the previous state.
            aux_prev_state_confs = np.zeros_like(ini_state_confs)
            aux_prev_state_props = np.zeros_like(ini_state_props)

            # Auxiliary arrays for the next state.
            aux_next_state_confs = np.zeros_like(ini_state_confs)
            aux_next_state_props = np.zeros_like(ini_state_props)

            # Current state properties.
            state_energies = state_props[props_energy_field]
            state_weights = state_props[props_weight_field]

            # Seed the numba RNG.
            # TODO: Handling of None seeds...
            random.seed(rng_seed)

            # The total energy and weight, used to update the
            # energy of reference for population control.
            total_energy = 0.
            total_weight = 0.
            state_ref_energy = ini_state.ref_energy

            # Initial configuration.
            aux_prev_state_confs[:] = ini_state.confs[:]
            aux_prev_state_props[:] = ini_state.props[:]
            prev_num_walkers = ini_state.num_walkers

            prev_state_confs, prev_state_props = \
                aux_prev_state_confs, aux_prev_state_props

            next_state_confs, next_state_props = \
                aux_next_state_confs, aux_next_state_props

            # The philosophy of the generator is simple: keep sampling
            # new states until the loop is broken from an outer scope.
            while True:

                # We now have the effective number of walkers after branching.
                state_num_walkers = \
                    sync_branching_spec(prev_state_props,
                                        prev_num_walkers,
                                        max_num_walkers,
                                        state_branching_spec)

                evolve_state(prev_state_confs,
                             prev_state_props,
                             state_confs,
                             state_props,
                             next_state_confs,
                             next_state_props,
                             state_num_walkers,
                             max_num_walkers,
                             time_step,
                             state_ref_energy,
                             state_branching_spec,
                             cfc_spec)

                # Update total energy and weight of the system.
                state_energy = \
                    state_energies[:state_num_walkers].sum()
                state_weight = \
                    state_weights[:state_num_walkers].sum()

                total_energy += state_energy
                total_weight += state_weight

                # Update reference energy to avoid the explosion of the
                # number of walkers.
                # TODO: Pass the control factor (0.5) as an argument.
                state_accum_energy = total_energy / total_weight
                state_ref_energy = state_accum_energy - 0.5 * log(
                        state_weight / target_num_walkers) / time_step

                yield State(confs=state_confs,
                            props=state_props,
                            energy=state_energy,
                            weight=state_weight,
                            num_walkers=state_num_walkers,
                            ref_energy=state_ref_energy,
                            accum_energy=state_accum_energy,
                            max_num_walkers=max_num_walkers,
                            branching_spec=state_branching_spec)

                # Exchange previous and next states arrays.
                # NOTE: I'm not sure if Numba is copying the data from one
                #  array to the other.
                prev_state_confs, next_state_confs = \
                    next_state_confs, prev_state_confs

                prev_state_props, next_state_props = \
                    next_state_props, prev_state_props

                prev_num_walkers = state_num_walkers

        return _states_generator

    @cached_property
    def batches(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath

        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value
        accum_energy_field = IterProp.ACCUM_ENERGY.value

        # JIT routines.
        states_generator = self.states_generator
        init_ssf_est_data = self.init_ssf_est_data
        fourier_density = self.fourier_density

        @nb.jit(nopython=True, nogil=True, fastmath=fastmath)
        def _batches(time_step: float,
                     ini_state: State,
                     num_time_steps_batch: int,
                     target_num_walkers: int,
                     rng_seed: int,
                     cfc_spec: CFCSpec):
            """

            :param time_step:
            :param ini_state:
            :param num_time_steps_batch:
            :param target_num_walkers:
            :param rng_seed:
            :param cfc_spec:
            :return:
            """
            ssf_params = cfc_spec.ssf_params
            max_num_walkers = ini_state.max_num_walkers

            # Alias :)...
            nts_batch = num_time_steps_batch

            # The shape of the batches.
            ipb_shape = nts_batch,

            # Array to store the configuration data of a batch of states.
            iter_props_array = np.zeros(ipb_shape, dtype=iter_props_dtype)
            iter_energies = iter_props_array[iter_energy_field]
            iter_weights = iter_props_array[iter_weight_field]
            iter_num_walkers = iter_props_array[iter_num_walkers_field]
            iter_ref_energies = iter_props_array[ref_energy_field]
            iter_accum_energies = iter_props_array[accum_energy_field]

            # Static structure factor arrays.
            ssf_est_data = \
                init_ssf_est_data(nts_batch, max_num_walkers, cfc_spec)
            ssf_momenta = ssf_est_data.momenta
            iter_ssf_array = ssf_est_data.iter_ssf_array
            pfw_aux_ssf_array = ssf_est_data.pfw_aux_ssf_array

            # Create a new sampling generator.
            generator = states_generator(time_step, ini_state,
                                         target_num_walkers,
                                         rng_seed, cfc_spec)

            # Future reference to the last DMC state of each batch.
            state = ini_state

            # Yield indefinitely 🤔.
            while True:

                # NOTE: Enumerate causes a memory leak in numba 0.40.
                #   See https://github.com/numba/numba/issues/3473.
                # enum_generator: T_E_SIter = enumerate(generator)

                # Reset the zero the accumulated S(k) of all the states.
                # This is very important, as the elements of this array are
                # modified in place during the sampling of the estimator.
                iter_ssf_array[:] = 0

                # Reset to zero the auxiliary states after the end of
                # each batch to accumulate new values during the forward
                # walking to sampling pure estimator. This has to be done
                # in order to gather new data in the next batch/block.
                pfw_aux_ssf_array[:] = 0.

                # We use an initial index instead enumerate.
                step_idx = 0

                for state in generator:

                    state_confs = state.confs
                    num_walkers = state.num_walkers
                    branching_spec = state.branching_spec

                    # Copy other data to keep track of the evolution.
                    iter_energies[step_idx] = state.energy
                    iter_weights[step_idx] = state.weight
                    iter_num_walkers[step_idx] = num_walkers
                    iter_ref_energies[step_idx] = state.ref_energy
                    iter_accum_energies[step_idx] = state.accum_energy

                    # Evaluate the Static Structure Factor.
                    if not ssf_params.assume_none:

                        fourier_density(step_idx,
                                        ssf_momenta,
                                        state_confs,
                                        num_walkers,
                                        max_num_walkers,
                                        branching_spec,
                                        cfc_spec,
                                        iter_ssf_array,
                                        pfw_aux_ssf_array)

                    # Stop/pause the iteration.
                    if step_idx + 1 >= nts_batch:
                        break

                    # Counter goes up 🙂.
                    step_idx += 1

                # NOTE: A copy. It is not so ugly 🤔.
                last_state = State(state.confs,
                                   state.props,
                                   state.energy,
                                   state.weight,
                                   state.num_walkers,
                                   state.ref_energy,
                                   state.accum_energy,
                                   state.max_num_walkers,
                                   state.branching_spec)

                batch_data = SamplingBatch(iter_props_array,
                                           iter_ssf_array,
                                           last_state)
                yield batch_data

        return _batches

    @cached_property
    def confs_props_batches(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value
        accum_energy_field = IterProp.ACCUM_ENERGY.value

        states_generator = self.states_generator

        @nb.jit(nopython=True, nogil=True, fastmath=fastmath)
        def _batches(time_step: float,
                     ini_state: State,
                     num_time_steps_batch: int,
                     target_num_walkers: int,
                     rng_seed: int = None):
            """The DMC sampling generator.

            :param time_step:
            :param num_time_steps_batch:
            :param ini_state:
            :param target_num_walkers:
            :param rng_seed:
            :return:
            """
            # Initial state properties.
            ini_state_confs = ini_state.confs
            ini_state_props = ini_state.props

            # Alias 🙂
            nts_batch = num_time_steps_batch
            isc_shape = ini_state_confs.shape
            isp_shape = ini_state_props.shape

            # The shape of the batches.
            scb_shape = (nts_batch,) + isc_shape
            spb_shape = (nts_batch,) + isp_shape
            ipb_shape = nts_batch,

            # Array dtypes.
            state_confs_dtype = ini_state_confs.dtype
            state_props_dtype = ini_state_props.dtype

            # Array for the states configurations data.
            states_confs_array = np.zeros(scb_shape, dtype=state_confs_dtype)

            # Array for the states properties data.
            states_props_array = np.zeros(spb_shape, dtype=state_props_dtype)

            # Array to store the configuration data of a batch of states.
            iter_props_array = np.zeros(ipb_shape, dtype=iter_props_dtype)

            iter_energies = iter_props_array[iter_energy_field]
            iter_weights = iter_props_array[iter_weight_field]
            iter_num_walkers = iter_props_array[iter_num_walkers_field]
            iter_ref_energies = iter_props_array[ref_energy_field]
            iter_accum_energies = iter_props_array[accum_energy_field]

            # Create a new sampling generator.
            generator = states_generator(time_step, ini_state,
                                         target_num_walkers, rng_seed)

            # Yield batches indefinitely.
            while True:

                enum_generator: T_E_SIter = enumerate(generator)

                for sj_, state in enum_generator:

                    # Copy the data to the batch.
                    states_confs_array[sj_] = state.confs[:]
                    states_props_array[sj_] = state.props[:]

                    # Copy other data to keep track of the evolution.
                    iter_energies[sj_] = state.energy
                    iter_weights[sj_] = state.weight
                    iter_num_walkers[sj_] = state.num_walkers
                    iter_ref_energies[sj_] = state.ref_energy
                    iter_accum_energies[sj_] = state.accum_energy

                    # Stop/pause the iteration.
                    if sj_ + 1 >= nts_batch:
                        break

                iter_data = SamplingConfsPropsBatch(states_confs_array,
                                                    states_props_array,
                                                    iter_props_array)
                yield iter_data

        return _batches
