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
    'EvoStateResult',
    'EvoStatesBatchResult',
    'IterProp',
    'Sampling',
    'SamplingIterData',
    'State',
    'StateProp',
    'branching_spec_dtype',
    'iter_props_dtype'
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
    num_walkers: int
    max_num_walkers: int


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


class Sampling(Iterable, metaclass=ABCMeta):
    """Realizes a VMC sampling using an iterable interface.

    Defines the parameters and related properties of a Variational Monte
    Carlo calculation.
    """
    __slots__ = ()

    #: The "time-step" (squared, average move spread) of the sampling.
    time_step: float

    #: The number of batches of the sampling.
    num_batches: int

    #: The number of steps of a sampling batch.
    num_time_steps_batch: int

    #: The initial configuration set of the sampling.
    ini_sys_conf_set: np.ndarray

    #: The initial energy of reference.
    ini_ref_energy: float

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

    #:
    branching_spec: t.ClassVar = BranchingSpecField

    @property
    def num_steps(self):
        """The total number of time steps of the sampling"""
        return self.num_time_steps_batch * self.num_batches

    @property
    @abstractmethod
    def state_confs_shape(self):
        pass

    @property
    @abstractmethod
    def state_props_shape(self):
        pass

    @property
    @abstractmethod
    def ini_state(self):
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


iter_props_dtype = np.dtype([
    (IterProp.ENERGY.value, np.float64),
    (IterProp.WEIGHT.value, np.float64),
    (IterProp.NUM_WALKERS.value, np.uint32),
    (IterProp.REF_ENERGY.value, np.float64)
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
        factor_field = BranchingSpecField.CLONING_FACTOR.value
        clone_ref_field = BranchingSpecField.CLONING_REF.value

        @nb.jit(nopython=True)
        def _sync_branching_spec(branching_spec: np.ndarray,
                                 actual_num_walkers: int,
                                 max_num_walkers: int):
            """

            :param branching_spec:
            :param max_num_walkers:
            :return:
            """
            clone_refs = branching_spec[clone_ref_field]
            cloning_factors = branching_spec[factor_field]

            new_num_walkers = 0
            for sys_idx in range(actual_num_walkers):
                # NOTE: num_walkers cannot be greater, right?
                if new_num_walkers >= max_num_walkers:
                    break
                clone_factor = int(cloning_factors[sys_idx])
                if not clone_factor:
                    continue
                prev_num_walkers = new_num_walkers
                new_num_walkers = min(max_num_walkers,
                                      new_num_walkers + clone_factor)
                clone_refs[prev_num_walkers:new_num_walkers] = sys_idx

            return new_num_walkers

        return _sync_branching_spec

    @cached_property
    def evolve_state(self):
        """

        :return:
        """
        props_energy_field = StateProp.ENERGY.value
        props_weight_field = StateProp.WEIGHT.value
        props_mask_field = StateProp.MASK.value

        branch_factor_field = BranchingSpecField.CLONING_FACTOR.value
        branch_ref_field = BranchingSpecField.CLONING_REF.value

        # JIT methods.
        evolve_system = self.evolve_system
        sync_branching_spec = self.sync_branching_spec

        @nb.jit(nopython=True, parallel=True)
        def _evolve_state(prev_state_confs: np.ndarray,
                          prev_state_props: np.ndarray,
                          actual_state_confs: np.ndarray,
                          actual_state_props: np.ndarray,
                          aux_next_state_confs: np.ndarray,
                          aux_next_state_props: np.ndarray,
                          prev_num_walkers: int,
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

            prev_state_weights = prev_state_props[props_weight_field]
            actual_state_weights = actual_state_props[props_weight_field]
            aux_next_state_weights = aux_next_state_props[props_weight_field]

            actual_state_masks = actual_state_props[props_mask_field]

            # Total energy and weight of the next configuration.
            state_energy = 0.
            state_weight = 0.

            # Initially, mask all the configurations.
            actual_state_masks[:] = True

            # We now have the effective number of walkers after branching.
            num_walkers = sync_branching_spec(branching_spec,
                                              prev_num_walkers,
                                              max_num_walkers)

            cloning_factors = branching_spec[branch_factor_field]
            cloning_refs = branching_spec[branch_ref_field]

            # Branching and diffusion process (parallel for).
            for sys_idx in nb.prange(num_walkers):
                # Lookup which configuration should be cloned.
                ref_idx = cloning_refs[sys_idx]
                sys_energy = prev_state_energies[ref_idx]
                sys_weight = prev_state_weights[ref_idx]

                # Cloning process. Actual states are not modified.
                actual_state_confs[sys_idx] = prev_state_confs[ref_idx]
                actual_state_energies[sys_idx] = sys_energy

                # Basic algorithm of branching gives a unit weight to each
                # new walker. We set the value here. In addition, we unmask
                # the walker, i.e., we mark it as valid.
                actual_state_weights[sys_idx] = sys_weight
                actual_state_masks[sys_idx] = False

                # The contribution to the total energy and weight.
                state_energy += sys_energy
                state_weight += 1.0

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

                # Next system energy and weight.
                sys_weight = aux_next_state_weights[sys_idx]

                # Cloning factor of the current walker of the next
                # generation.
                # NOTE: Keep an eye on cloning_factors array...
                clone_factor = int(sys_weight + random.rand())
                cloning_factors[sys_idx] = clone_factor

            return EvoStateResult(state_energy, state_weight, num_walkers)

        return _evolve_state

    @cached_property
    def evolve_states_batch(self):
        """

        :return:
        """
        props_weight_field = StateProp.WEIGHT.value

        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value

        branch_factor_field = BranchingSpecField.CLONING_FACTOR.value

        # JIT functions.
        evolve_state = self.evolve_state

        @nb.jit(nopython=True)
        def _evolve_states_batch(ini_state_confs: np.ndarray,
                                 ini_state_props: np.ndarray,
                                 ini_num_walkers: int,
                                 ini_ref_energy: float,
                                 time_step: float,
                                 num_time_steps_batch: int,
                                 target_num_walkers: int,
                                 max_num_walkers: int,
                                 states_confs_array: np.ndarray,
                                 states_props_array: np.ndarray,
                                 iter_props_array: np.ndarray):
            """Realizes the DMC sampling in batches.

            The sampling is done in batches, with each batch having a fixed
            number of time steps given by the ``num_time_steps_batch``
            argument.

            :param ini_state_confs:
            :param ini_state_props:
            :param states_confs_array:
            :param states_props_array:
            :param iter_props_array:
            :param num_time_steps_batch:
            :param max_num_walkers:
            :param time_step:
            :param ini_ref_energy:
            :param target_num_walkers:
            :return:
            """
            iter_energies = iter_props_array[iter_energy_field]
            iter_weights = iter_props_array[iter_weight_field]
            iter_num_walkers = iter_props_array[iter_num_walkers_field]
            iter_ref_energies = iter_props_array[ref_energy_field]

            # Auxiliary configuration.
            aux_next_state_confs = np.zeros_like(ini_state_confs)
            aux_next_state_props = np.zeros_like(ini_state_props)

            # Actual states configurations.
            actual_state_confs = np.zeros_like(ini_state_confs)
            actual_state_props = np.zeros_like(ini_state_props)

            # Table to control the branching process.
            branching_spec = \
                np.zeros(max_num_walkers, dtype=branching_spec_dtype)

            # Initial configuration.
            prev_state_confs = ini_state_confs
            prev_state_props = ini_state_props
            ini_state_weights = ini_state_props[props_weight_field]

            # Calculate the cloning factors for the initial state.
            cloning_factors = branching_spec[branch_factor_field]
            for sys_idx in range(ini_num_walkers):
                sys_weight = ini_state_weights[sys_idx]
                clone_factor = int(sys_weight + random.rand())
                cloning_factors[sys_idx] = clone_factor

            # Energy of reference for population control.
            batch_energy = 0.
            batch_weight = 0.
            ref_energy = ini_ref_energy
            prev_num_walkers = ini_num_walkers

            # Limits of the loop.
            sj_ini = 0
            sj_end = num_time_steps_batch

            for sj_ in range(sj_ini, sj_end):
                # The next configuration of this block.
                # actual_state_confs = states_confs_array[sj_]
                # actual_state_props = states_props_array[sj_]

                evo_result = evolve_state(prev_state_confs,
                                          prev_state_props,
                                          actual_state_confs,
                                          actual_state_props,
                                          aux_next_state_confs,
                                          aux_next_state_props,
                                          prev_num_walkers,
                                          max_num_walkers,
                                          time_step,
                                          ref_energy,
                                          branching_spec)

                # Update total energy and weight of the system.
                sj_energy = evo_result.energy
                sj_weight = evo_result.weight
                sj_num_walkers = evo_result.num_walkers
                batch_energy += sj_energy
                batch_weight += sj_weight

                # Update reference energy to avoid the explosion of the
                # number of walkers...
                # TODO: Pass the control factor (0.5) as an argument.
                ref_energy = batch_energy / batch_weight
                ref_energy -= 0.5 * log(
                        sj_weight / target_num_walkers) / time_step

                # Update energy and weights.
                iter_energies[sj_] = sj_energy
                iter_weights[sj_] = sj_weight
                iter_ref_energies[sj_] = ref_energy

                # Update the number of walkers of this step.
                iter_num_walkers[sj_] = sj_num_walkers

                prev_state_confs = aux_next_state_confs
                prev_state_props = aux_next_state_props
                prev_num_walkers = sj_num_walkers

            return EvoStatesBatchResult(prev_state_confs,
                                        prev_state_props,
                                        prev_num_walkers,
                                        ref_energy)

        return _evolve_states_batch

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
    def generator(self):
        """

        :return:
        """
        evolve_states_batch = self.evolve_states_batch

        # prepare_ini_iter_data = self.prepare_ini_iter_data

        @nb.jit(nopython=True, nogil=True)
        def _generator(time_step: float,
                       num_batches: int,
                       num_time_steps_batch: int,
                       ini_state: State,
                       ini_ref_energy: float,
                       target_num_walkers: int,
                       rng_seed: int):
            """The DMC sampling generator.

            :param time_step:
            :param num_batches:
            :param num_time_steps_batch:
            :param ini_state:
            :param ini_ref_energy:
            :param target_num_walkers:
            :param rng_seed:
            :return:
            """
            # Initial state properties.
            ini_state_confs = ini_state.confs
            ini_state_props = ini_state.props
            ini_num_walkers = ini_state.num_walkers

            # Alias 🙂
            nts_batch = num_time_steps_batch
            isc_shape = ini_state_confs.shape
            isp_shape = ini_state_props.shape
            max_num_walkers = ini_state.max_num_walkers

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

            # Seed the numba RNG.
            random.seed(rng_seed)

            # We yield the initial state iter properties.
            # yield prepare_ini_iter_data(ini_state, ini_ref_energy)

            # Limits of the sampling.
            nbj_ini, nbj_end = 0, num_batches
            for nbj in range(nbj_ini, nbj_end):
                #
                evo_result = evolve_states_batch(ini_state_confs,
                                                 ini_state_props,
                                                 ini_num_walkers,
                                                 ini_ref_energy,
                                                 time_step,
                                                 num_time_steps_batch,
                                                 target_num_walkers,
                                                 max_num_walkers,
                                                 states_confs_array,
                                                 states_props_array,
                                                 iter_props_array)

                iter_data = SamplingIterData(states_confs_array,
                                             states_props_array,
                                             iter_props_array)
                yield iter_data

                ini_state_confs = evo_result.last_confs.copy()
                ini_state_props = evo_result.last_props.copy()
                ini_num_walkers = evo_result.last_num_walkers
                ini_ref_energy = evo_result.last_ref_energy

        return _generator
