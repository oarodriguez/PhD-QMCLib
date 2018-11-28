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
    'CoreFuncs',
    'Sampling',
    'State'
]


@enum.unique
class StateProp(enum.Enum):
    """The properties of a configuration."""
    ENERGY = 'ENERGY'
    WEIGHT = 'WEIGHT'
    MASK = 'MASK'


@enum.unique
class IterProp(enum.Enum):
    """"""
    ENERGY = 'ENERGY'
    WEIGHT = 'WEIGHT'
    NUM_WALKERS = 'NUM_WALKERS'
    REF_ENERGY = 'REF_ENERGY'


@enum.unique
class BranchingSpecField(enum.Enum):
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

    #: The sampling core functions.
    core_funcs: 'CoreFuncs'

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

    #
    model_spec: Sampling

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

        branch_factor_field = BranchingSpecField.CLONING_FACTOR.value
        branch_ref_field = BranchingSpecField.CLONING_REF.value

        # JIT methods.
        evolve_system = self.evolve_system
        sync_branching_spec = self.sync_branching_spec

        @nb.jit(nopython=True, parallel=True)
        def _evolve_state(actual_state_conf: np.ndarray,
                          actual_state_props: np.ndarray,
                          next_state_conf: np.ndarray,
                          next_state_props: np.ndarray,
                          aux_state_conf: np.ndarray,
                          aux_state_props: np.ndarray,
                          actual_num_walkers: int,
                          max_num_walkers: int,
                          time_step: float,
                          ref_energy: float,
                          branching_spec: np.ndarray):
            """Realizes the diffusion-branching process.

            This function realize a simple diffusion process over each
            one of the walkers, followed by the branching process.

            :param actual_state_conf:
            :param actual_state_props:
            :param next_state_conf:
            :param next_state_props:
            :param aux_state_conf:
            :param aux_state_props:
            :param max_num_walkers:
            :param time_step:
            :param ref_energy:
            :return:
            """

            # Arrays of properties.
            actual_state_energies = actual_state_props[props_energy_field]
            next_state_energies = next_state_props[props_energy_field]
            aux_state_energies = aux_state_props[props_energy_field]

            actual_state_weights = actual_state_props[props_weight_field]
            next_state_weights = next_state_props[props_weight_field]
            aux_state_weights = aux_state_props[props_weight_field]

            cloning_factors = branching_spec[branch_factor_field]
            cloning_refs = branching_spec[branch_ref_field]

            # Total energy and weight of the next configuration.
            state_energy = 0.
            state_weight = 0.

            # Diffusion process (parallel).
            for sys_idx in nb.prange(actual_num_walkers):

                # TODO: Can we return tuples inside a nb.prange?
                evolve_system(sys_idx, actual_state_conf,
                              actual_state_energies, actual_state_weights,
                              aux_state_conf, aux_state_energies,
                              aux_state_weights, time_step, ref_energy,
                              next_state_conf, next_state_energies,
                              next_state_weights)

                # Current system energy and weight.
                sys_energy = next_state_energies[sys_idx]
                sys_weight = next_state_weights[sys_idx]

                # Cloning factor.
                clone_factor = int(sys_weight + random.rand())
                cloning_factors[sys_idx] = clone_factor

                # Basic algorithm of branch and rebirth gives a unit
                state_energy += sys_energy * sys_weight
                state_weight += sys_weight

            # We now have the effective number of walkers after branching.
            num_walkers = sync_branching_spec(branching_spec,
                                              actual_num_walkers,
                                              max_num_walkers)

            # Branching process (parallel for).
            for sys_idx in nb.prange(num_walkers):
                # Lookup which configuration should be cloned.
                ref_idx = cloning_refs[sys_idx]

                # Cloning process.
                next_state_conf[sys_idx] = aux_state_conf[ref_idx]
                next_state_energies[sys_idx] = aux_state_energies[ref_idx]
                next_state_weights[sys_idx] = aux_state_weights[ref_idx]

            return EvoStateResult(state_energy, state_weight, num_walkers)

        return _evolve_state

    @cached_property
    def evolve_states_batch(self):
        """

        :return:
        """
        iter_energy_field = IterProp.ENERGY.value
        iter_weight_field = IterProp.WEIGHT.value
        iter_num_walkers_field = IterProp.NUM_WALKERS.value
        ref_energy_field = IterProp.REF_ENERGY.value

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
            """

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
            aux_state_confs = np.zeros_like(ini_state_confs)
            aux_state_props = np.zeros_like(ini_state_props)

            # Table to control the branching process.
            branching_spec = \
                np.zeros(max_num_walkers, dtype=branching_spec_dtype)

            # Initial configuration.
            actual_state_confs = ini_state_confs
            actual_state_props = ini_state_props

            # NOTE: Is this necessary and/or useful?
            # states_props_mask = states_props_array[props_mask_field]
            # states_props_mask[:] = True

            # Energy of reference for population control.
            batch_energy = 0.
            batch_weight = 0.
            ref_energy = ini_ref_energy
            actual_num_walkers = ini_num_walkers

            # Limits of the loop.
            sj_ini = 0
            sj_end = num_time_steps_batch

            for sj_ in range(sj_ini, sj_end):
                # The next configuration of this block.
                next_state_confs = states_confs_array[sj_]
                next_state_props = states_props_array[sj_]

                evo_result = evolve_state(actual_state_confs,
                                          actual_state_props,
                                          next_state_confs,
                                          next_state_props,
                                          aux_state_confs,
                                          aux_state_props,
                                          actual_num_walkers,
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

                actual_state_confs = next_state_confs
                actual_state_props = next_state_props
                actual_num_walkers = sj_num_walkers

            return EvoStatesBatchResult(actual_state_confs,
                                        actual_state_props,
                                        actual_num_walkers,
                                        ref_energy)

        return _evolve_states_batch

    @property
    @abstractmethod
    def prepare_ini_state(self):
        """"""
        pass

    @cached_property
    def generator(self):
        """

        :return:
        """
        evolve_states_batch = self.evolve_states_batch

        @nb.jit(nopython=True)
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
            scp_shape = (nts_batch,) + isp_shape
            ipb_shape = nts_batch,

            # Array dtypes.
            state_confs_dtype = ini_state_confs.dtype
            state_props_dtype = ini_state_props.dtype

            # Array for the states configurations data.
            states_confs_array = np.zeros(scb_shape, dtype=state_confs_dtype)

            # Array for the states properties data.
            states_props_array = np.zeros(scp_shape, dtype=state_props_dtype)

            # Array to store the configuration data of a batch of states.
            iter_props_array = np.zeros(ipb_shape, dtype=iter_props_dtype)

            # Seed the numba RNG.
            random.seed(rng_seed)

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
