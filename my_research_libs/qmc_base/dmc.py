"""
    my_research_libs.qmc_base.dmc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements the main Diffusion Monte Carlo classes and routines.
"""

import enum
import typing as t
from abc import ABCMeta, abstractmethod
from collections import Iterable
from math import log

import numba as nb
import numpy as np
from cached_property import cached_property
from numpy import random

CONF_INDEX = 0
ENERGY_INDEX = 1

__all__ = [
    'BranchingSpecField',
    'CoreFuncs',
    'EstAuxData',
    'EstSampling',
    'EstSamplingBatch',
    'EstSamplingCoreFuncs',
    'EvoStateResult',
    'EvoStatesBatchResult',
    'IterProp',
    'Sampling',
    'SamplingIterData',
    'State',
    'StateProp',
    'StructureFactorEst',
    'branching_spec_dtype',
    'iter_props_dtype',
    'dummy_pure_est_core_func'
]


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


class EvoStateResult(t.NamedTuple):
    """"""
    energy: float
    weight: float
    num_walkers: int


class EvoStatesBatchResult(t.NamedTuple):
    """"""
    last_confs: np.ndarray
    last_props: np.ndarray
    last_num_walkers: int
    last_ref_energy: float


class SamplingIterData(t.NamedTuple):
    """"""
    states_confs: np.ndarray
    states_props: np.ndarray
    iter_props: np.ndarray


T_BatchesGenerator = t.Generator[SamplingIterData, t.Any, None]
T_StateIter = t.Iterator[State]
T_E_StateIter = t.Iterator[t.Tuple[int, State]]


class Sampling(Iterable, metaclass=ABCMeta):
    """Realizes a DMC sampling using an iterable interface.

    Defines the parameters and related properties of a Diffusion Monte
    Carlo calculation.
    """
    __slots__ = ()

    #: The "time-step" (squared, average move spread) of the sampling.
    time_step: float

    #: The initial configuration set of the sampling.
    ini_sys_conf_set: np.ndarray

    #: The initial energy of reference.
    ini_ref_energy: t.Optional[float] = None

    #: The maximum wight of the population of walkers.
    max_num_walkers: int

    #: The average total weight of the population of walkers.
    target_num_walkers: int

    #: Multiplier for the population control during the branching stage.
    num_walkers_control_factor: float

    #: The seed of the pseudo-RNG used to realize the sampling.
    rng_seed: int

    #:
    state_props: t.ClassVar = StateProp

    #:
    iter_props: t.ClassVar = IterProp

    @property
    @abstractmethod
    def state_confs_shape(self):
        pass

    @property
    @abstractmethod
    def state_props_shape(self):
        pass

    @abstractmethod
    def init_get_ini_state(self) -> State:
        """The initial state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.
        """
        pass

    @property
    @abstractmethod
    def core_funcs(self) -> 'CoreFuncs':
        """The sampling core functions."""
        pass

    def batches(self, num_time_steps_batch: int) -> T_BatchesGenerator:
        """Generator object that yields batches of states."""
        ini_state = self.init_get_ini_state()
        time_step = self.time_step
        target_num_walkers = self.target_num_walkers
        rng_seed = self.rng_seed

        return self.core_funcs.batches(time_step,
                                       ini_state,
                                       num_time_steps_batch,
                                       target_num_walkers,
                                       rng_seed)

    def __iter__(self) -> t.Generator[State, t.Any, None]:
        """Iterable interface."""

        time_step = self.time_step
        rng_seed = self.rng_seed
        ini_state = self.init_get_ini_state()
        target_num_walkers = self.target_num_walkers

        return self.core_funcs.states_generator(time_step,
                                                ini_state,
                                                target_num_walkers,
                                                rng_seed)


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


class CoreFuncs(metaclass=ABCMeta):
    """The core functions for a DMC sampling."""

    __slots__ = ()

    @property
    @abstractmethod
    def evolve_system(self):
        """"""
        pass

    @cached_property
    def sync_branching_spec(self):
        """

        :return:
        """
        props_weight_field = StateProp.WEIGHT.value
        clone_ref_field = BranchingSpecField.CLONING_REF.value

        @nb.jit(nopython=True)
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

    @cached_property
    def evolve_state(self):
        """

        :return:
        """
        props_energy_field = StateProp.ENERGY.value
        props_weight_field = StateProp.WEIGHT.value
        props_mask_field = StateProp.MASK.value
        branch_ref_field = BranchingSpecField.CLONING_REF.value

        # JIT methods.
        evolve_system = self.evolve_system

        @nb.jit(nopython=True, parallel=True)
        def _evolve_state(prev_state_confs: np.ndarray,
                          prev_state_props: np.ndarray,
                          actual_state_confs: np.ndarray,
                          actual_state_props: np.ndarray,
                          aux_next_state_confs: np.ndarray,
                          aux_next_state_props: np.ndarray,
                          actual_num_walkers: int,
                          max_num_walkers: int,
                          time_step: float,
                          ref_energy: float,
                          branching_spec: np.ndarray):
            """Realizes the diffusion-branching process.

            This function realize a simple diffusion process over each
            one of the walkers, followed by the branching process.

            :param prev_state_confs:
            :param prev_state_props:
            :param actual_state_confs:
            :param actual_state_props:
            :param aux_next_state_confs:
            :param aux_next_state_props:
            :param max_num_walkers:
            :param time_step:
            :param ref_energy:
            :return:
            """
            # Arrays of properties.
            prev_state_energies = prev_state_props[props_energy_field]
            actual_state_energies = actual_state_props[props_energy_field]
            aux_next_state_energies = aux_next_state_props[props_energy_field]

            # prev_state_weights = prev_state_props[props_weight_field]
            actual_state_weights = actual_state_props[props_weight_field]
            aux_next_state_weights = aux_next_state_props[props_weight_field]

            actual_state_masks = actual_state_props[props_mask_field]
            cloning_refs = branching_spec[branch_ref_field]

            # Total energy and weight of the next configuration.
            state_energy = 0.
            state_weight = 0.

            # Initially, mask all the configurations.
            actual_state_masks[:] = True

            # Branching and diffusion process (parallel for).
            for sys_idx in nb.prange(max_num_walkers):

                # Beyond the actual number of walkers just pass to
                # the next iteration.
                if sys_idx >= actual_num_walkers:
                    continue

                # Lookup which configuration should be cloned.
                ref_idx = cloning_refs[sys_idx]
                sys_energy = prev_state_energies[ref_idx]
                # sys_weight = prev_state_weights[ref_idx]

                # Cloning process. Actual states are not modified.
                actual_state_confs[sys_idx] = prev_state_confs[ref_idx]
                actual_state_energies[sys_idx] = sys_energy

                # Basic algorithm of branching gives a unit weight to each
                # new walker. We set the value here. In addition, we unmask
                # the walker, i.e., we mark it as valid.
                actual_state_weights[sys_idx] = 1.
                actual_state_masks[sys_idx] = False

                # The contribution to the total energy and weight.
                state_energy += sys_energy
                state_weight += 1.

                # Evolve the system for the next iteration.
                # TODO: Can we return tuples inside a nb.prange?
                evolve_system(sys_idx, actual_state_confs,
                              actual_state_energies,
                              actual_state_weights,
                              time_step,
                              ref_energy,
                              aux_next_state_confs,
                              aux_next_state_energies,
                              aux_next_state_weights)

            return EvoStateResult(state_energy, state_weight,
                                  actual_num_walkers)

        return _evolve_state

    @cached_property
    def states_generator(self):
        """

        :return:
        """
        # JIT functions.
        evolve_state = self.evolve_state
        sync_branching_spec = self.sync_branching_spec

        @nb.jit(nopython=True)
        def _states_generator(time_step: float,
                              ini_state: State,
                              target_num_walkers: int,
                              rng_seed: int = None):
            """Realizes the DMC sampling state-by-state.

            The sampling is done in batches, with each batch having a fixed
            number of time steps given by the ``num_time_steps_batch``
            argument.

            :param ini_state:
            :param time_step:
            :param target_num_walkers:
            :return:
            """
            # The initial state fixes the arrays of the following states.
            ini_state_confs = ini_state.confs
            ini_state_props = ini_state.props
            max_num_walkers = ini_state.max_num_walkers

            # Configurations and properties of the current
            # state of the sampling (the one that will be yielded).
            actual_state_confs = np.zeros_like(ini_state_confs)
            actual_state_props = np.zeros_like(ini_state_props)

            # Auxiliary configuration.
            aux_next_state_confs = np.zeros_like(ini_state_confs)
            aux_next_state_props = np.zeros_like(ini_state_props)

            # Seed the numba RNG.
            if rng_seed:
                random.seed(rng_seed)

            # Table to control the branching process.
            branching_spec = \
                np.zeros(max_num_walkers, dtype=branching_spec_dtype)

            # The total energy and weight, used to update the
            # energy of reference for population control.
            total_energy = 0.
            total_weight = 0.

            # Initial configuration.
            prev_state = ini_state
            prev_state_confs = prev_state.confs
            prev_state_props = prev_state.props
            prev_num_walkers = prev_state.num_walkers
            ref_energy = prev_state.ref_energy

            # The philosophy of the generator is simple: keep sampling
            # new states until the loop is broken from an outer scope.
            while True:

                # We now have the effective number of walkers after branching.
                actual_num_walkers = sync_branching_spec(prev_state_props,
                                                         prev_num_walkers,
                                                         max_num_walkers,
                                                         branching_spec)

                evo_result = evolve_state(prev_state_confs,
                                          prev_state_props,
                                          actual_state_confs,
                                          actual_state_props,
                                          aux_next_state_confs,
                                          aux_next_state_props,
                                          actual_num_walkers,
                                          max_num_walkers,
                                          time_step,
                                          ref_energy,
                                          branching_spec)

                # Update total energy and weight of the system.
                state_energy = evo_result.energy
                state_weight = evo_result.weight
                total_energy += state_energy
                total_weight += state_weight

                # Update reference energy to avoid the explosion of the
                # number of walkers.
                # TODO: Pass the control factor (0.5) as an argument.
                accum_energy = total_energy / total_weight
                ref_energy = accum_energy - 0.5 * log(
                        state_weight / target_num_walkers) / time_step

                yield State(confs=actual_state_confs,
                            props=actual_state_props,
                            energy=state_energy,
                            weight=state_weight,
                            num_walkers=actual_num_walkers,
                            ref_energy=ref_energy,
                            accum_energy=accum_energy,
                            max_num_walkers=max_num_walkers,
                            branching_spec=branching_spec)

                prev_state_confs = aux_next_state_confs
                prev_state_props = aux_next_state_props
                prev_num_walkers = actual_num_walkers

        return _states_generator

    @property
    @abstractmethod
    def prepare_ini_state(self):
        """"""
        pass

    @cached_property
    def prepare_ini_iter_data(self):
        """Gets the initial state data as a one-element sampling batch."""

        props_energy_field = StateProp.ENERGY.value
        props_weight_field = StateProp.WEIGHT.value

        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        iter_ref_energy_field = IterProp.REF_ENERGY.value

        @nb.jit(nopython=True)
        def _prepare_ini_iter_data(ini_state: State,
                                   ini_ref_energy: float):
            """Gets the initial state data as a one-element sampling batch.

            :param ini_state: The initial state.
            :param ini_ref_energy: The initial energy of reference.
            :return: the initial state data as a one-element sampling batch.
            """
            # Initial state iter properties.
            ini_state_confs = ini_state.confs
            ini_state_props = ini_state.props
            ini_num_walkers = ini_state.num_walkers

            # The initial iter properties.
            ini_ipb_shape = (1,)

            # NOTE: We may use the data of the initial state directly, and
            #  return only views of the arrays. However, we use copies.
            ini_states_confs_array = ini_state_confs.copy()
            ini_states_props_array = ini_state_props.copy()
            ini_iter_props_array = \
                np.zeros(ini_ipb_shape, dtype=iter_props_dtype)

            is_energies = ini_state_props[props_energy_field]
            is_weights = ini_state_props[props_weight_field]

            iter_energies = ini_iter_props_array[iter_energy_field]
            iter_weights = ini_iter_props_array[iter_weight_field]
            iter_num_walkers = ini_iter_props_array[iter_num_walkers_field]
            iter_ref_energies = ini_iter_props_array[iter_ref_energy_field]

            inw = ini_num_walkers
            ini_energy = (is_energies[:inw] * is_weights[:inw]).sum()
            ini_weight = is_weights[:inw].sum()

            iter_energies[0] = ini_energy
            iter_weights[0] = ini_weight
            iter_num_walkers[0] = ini_num_walkers
            iter_ref_energies[0] = ini_ref_energy

            # The one-element batch.
            ini_scb_shape = ini_ipb_shape + ini_state_confs.shape
            ini_spb_shape = ini_ipb_shape + ini_state_props.shape

            ini_states_confs_array = \
                ini_states_confs_array.reshape(ini_scb_shape)
            ini_states_props_array = \
                ini_states_props_array.reshape(ini_spb_shape)

            return SamplingIterData(ini_states_confs_array,
                                    ini_states_props_array,
                                    ini_iter_props_array)

        return _prepare_ini_iter_data

    @cached_property
    def batches(self):
        """

        :return:
        """
        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value
        accum_energy_field = IterProp.ACCUM_ENERGY.value

        states_generator = self.states_generator

        @nb.jit(nopython=True, nogil=True)
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

                enum_generator: T_E_StateIter = enumerate(generator)

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

                iter_data = SamplingIterData(states_confs_array,
                                             states_props_array,
                                             iter_props_array)
                yield iter_data

        return _batches


class EstSamplingBatch(t.NamedTuple):
    """"""
    #: Properties data.
    iter_props: np.ndarray
    iter_structure_factor: np.ndarray = None


class EstAuxData(t.NamedTuple):
    """"""
    #: Properties data.
    structure_factor: np.ndarray = None


# Variables for type-hints.
T_ESBatchIter = t.Iterator[EstSamplingBatch]
T_E_ESBatchIter = t.Iterator[t.Tuple[int, EstSamplingBatch]]


class StructureFactorEst:
    """Structure factor estimator."""

    #: Number of modes to evaluate the structure factor S(k).
    num_modes: int

    #: Number of time steps to accumulate the pure estimator of S(k).
    init_num_time_steps: int = 512

    @property
    @abstractmethod
    def get_momenta(self, *args, **kwargs):
        """"""
        pass

    @property
    @abstractmethod
    def build_core_func(self, *args, **kwargs):
        """"""
        pass


class EstSampling(Sampling):
    """Evaluates expected values (estimators) using a DMC sampling."""

    # Inherited fields...
    time_step: float
    ini_sys_conf_set: np.ndarray
    ini_ref_energy: t.Optional[float] = None
    max_num_walkers: int = 544
    target_num_walkers: int = 512
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    # *** *** Configuration of the estimators. *** ***

    #: Structure factor estimator.
    structure_factor: t.Optional[StructureFactorEst] = None

    @property
    @abstractmethod
    def core_funcs(self) -> 'CoreFuncs':
        """"""
        pass


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


class EstSamplingCoreFuncs(CoreFuncs, metaclass=ABCMeta):
    """The DMC core functions for the Bloch-Phonon model."""

    #: Number of modes to evaluate the structure factor S(k).
    sf_num_modes: int

    #: Number of time steps to accumulate the pure estimator of S(k).
    sf_init_num_time_steps: int

    #: Core function (JIT) to calculate S(k).
    sf_core_func: t.Callable

    @property
    def should_eval_sf_est(self):
        """"""
        return self.sf_core_func is not dummy_pure_est_core_func

    @cached_property
    def batches(self):
        """

        :return:
        """
        sf_num_modes = self.sf_num_modes

        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value
        accum_energy_field = IterProp.ACCUM_ENERGY.value

        # JIT routines.
        eval_estimators = self.eval_estimators
        states_generator = self.states_generator

        @nb.jit(nopython=True, nogil=True)
        def _batches(time_step: float,
                     ini_state: State,
                     num_time_steps_batch: int,
                     target_num_walkers: int,
                     rng_seed: int = None):
            """

            :param time_step:
            :param ini_state:
            :param num_time_steps_batch:
            :param target_num_walkers:
            :param rng_seed:
            :return:
            """
            max_num_walkers = ini_state.max_num_walkers

            # Alias :)...
            nts_batch = num_time_steps_batch

            # The shape of the batches.
            ipb_shape = nts_batch,

            # The shape of the structure factor array.
            isf_shape = nts_batch, sf_num_modes  # S(k) batch.

            # The shape of the auxiliary arrays to store the structure
            # for a whole DMC state.
            aux_sfb_shape = 2, max_num_walkers, sf_num_modes

            # Array to store the configuration data of a batch of states.
            iter_props_array = np.zeros(ipb_shape, dtype=iter_props_dtype)
            iter_sf_array = np.zeros(isf_shape, dtype=np.float64)
            aux_sf_array = np.zeros(aux_sfb_shape, dtype=np.float64)

            # Extract fields.
            iter_energies = iter_props_array[iter_energy_field]
            iter_weights = iter_props_array[iter_weight_field]
            iter_num_walkers = iter_props_array[iter_num_walkers_field]
            iter_ref_energies = iter_props_array[ref_energy_field]
            iter_accum_energies = iter_props_array[accum_energy_field]

            # Create a new sampling generator.
            generator = states_generator(time_step, ini_state,
                                         target_num_walkers, rng_seed)

            # Yield indefinitely 🤔.
            while True:

                # NOTE: Enumerate here, inside the while.
                enum_generator: T_E_StateIter = enumerate(generator)

                # Reset to zero the auxiliary states after the end of
                # each batch.
                aux_sf_array[:] = 0.

                for step_idx, state in enum_generator:

                    state_confs = state.confs
                    num_walkers = state.num_walkers
                    branching_spec = state.branching_spec

                    # Copy other data to keep track of the evolution.
                    iter_energies[step_idx] = state.energy
                    iter_weights[step_idx] = state.weight
                    iter_num_walkers[step_idx] = num_walkers
                    iter_ref_energies[step_idx] = state.ref_energy
                    iter_accum_energies[step_idx] = state.accum_energy

                    # Evaluate the estimators.
                    eval_estimators(step_idx,
                                    state_confs,
                                    num_walkers,
                                    max_num_walkers,
                                    branching_spec,
                                    iter_props_array,
                                    iter_sf_array,
                                    aux_sf_array)

                    # Stop/pause the iteration.
                    if step_idx + 1 >= nts_batch:
                        break

                batch_data = EstSamplingBatch(iter_props_array,
                                              iter_sf_array)
                yield batch_data

        return _batches

    @cached_property
    def eval_estimators(self):
        """

        :return:
        """
        branch_ref_field = BranchingSpecField.CLONING_REF.value

        # Structure factor
        # Flag to evaluate structure factor.
        sf_init_nts = self.sf_init_num_time_steps
        sf_est_core_func = self.sf_core_func
        should_eval_sf_est = self.should_eval_sf_est

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, parallel=True)
        def _eval_estimators(step_idx: int,
                             state_confs: np.ndarray,
                             num_walkers: int,
                             max_num_walkers: int,
                             branching_spec: np.ndarray,
                             iter_props_array: np.ndarray,
                             iter_sf_array: np.ndarray,
                             aux_states_sf_array: np.ndarray):
            """

            :param step_idx:
            :param state_confs:
            :param num_walkers:
            :param max_num_walkers:
            :param branching_spec:
            :param iter_props_array:
            :param iter_sf_array:
            :param aux_states_sf_array:
            :return:
            """
            # Cloning table. Needed for evaluate pure estimators.
            cloning_refs = branching_spec[branch_ref_field]

            # Branching process (parallel for).
            for sys_idx in nb.prange(max_num_walkers):

                # Beyond the actual number of walkers just pass to
                # the next iteration.
                if sys_idx >= num_walkers:
                    continue

                # Lookup which configuration should be cloned.
                clone_ref_idx = cloning_refs[sys_idx]

                if should_eval_sf_est:
                    # Evaluate structure factor.
                    sf_est_core_func(step_idx,
                                     sys_idx,
                                     clone_ref_idx,
                                     state_confs,
                                     iter_sf_array,
                                     aux_states_sf_array)

            # Calculate structure factor pure estimator after the
            # initialization stage.
            if should_eval_sf_est:
                #
                if step_idx < sf_init_nts:
                    iter_sf_array[step_idx] /= step_idx + 1
                else:
                    iter_sf_array[step_idx] /= sf_init_nts

        return _eval_estimators
