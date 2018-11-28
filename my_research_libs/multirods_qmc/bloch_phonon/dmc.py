import typing as t
from math import exp, sqrt

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from numpy import random

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.utils import recast_to_supercell
from . import model


class State(qmc_base.dmc.State, t.NamedTuple):
    """"""
    confs: np.ndarray
    props: np.ndarray
    num_walkers: int
    max_num_walkers: int


StateProp = qmc_base.dmc.StateProp

state_confs_dtype = np.float64

state_props_dtype = np.dtype([
    (StateProp.ENERGY.value, np.float64),
    (StateProp.WEIGHT.value, np.float64),
])


@attr.s(auto_attribs=True, frozen=True)
class Sampling(qmc_base.dmc.Sampling):
    """A class to realize a DMC sampling."""

    #: The model instance.
    model_spec: model.Spec

    time_step: float
    num_batches: int
    num_time_steps_batch: int
    ini_sys_conf_set: np.ndarray
    ini_ref_energy: t.Optional[float] = None
    max_num_walkers: int = 1000
    target_num_walkers: int = 500
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    core_funcs: 'CoreFuncs' = attr.ib(init=False, cmp=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = utils.get_random_rng_seed()
            super().__setattr__('rng_seed', rng_seed)

        super().__setattr__('core_funcs', CoreFuncs(self.model_spec))

    @property
    def state_confs_shape(self):
        """"""
        max_num_walkers = self.max_num_walkers
        sys_conf_shape = self.model_spec.sys_conf_shape
        return (max_num_walkers,) + sys_conf_shape

    @property
    def state_props_shape(self):
        """"""
        max_num_walkers = self.max_num_walkers
        return max_num_walkers,

    @cached_property
    def ini_state(self):
        """The initial state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.
        """
        confs_shape = self.state_confs_shape
        props_shape = self.state_props_shape
        max_num_walkers = self.max_num_walkers
        ini_sys_conf_set = self.ini_sys_conf_set
        num_walkers = len(ini_sys_conf_set)

        # Initial state arrays.
        state_confs = np.zeros(confs_shape, dtype=state_confs_dtype)
        state_props = np.zeros(props_shape, dtype=state_props_dtype)

        # Calculate the initial state arrays properties.
        self.core_funcs.prepare_ini_state(ini_sys_conf_set, state_confs,
                                          state_props)

        return State(state_confs, state_props, num_walkers, max_num_walkers)

    def __iter__(self):
        """Iterable interface."""

        # Initial
        ini_state = self.ini_state
        ini_ref_energy = self.ini_ref_energy

        # Calculate the initial energy of reference as the average of the
        # energy of the initial state.
        if ini_ref_energy is None:
            #
            inw = ini_state.num_walkers
            state_energies = ini_state.props[StateProp.ENERGY.value][:inw]
            state_weights = ini_state.props[StateProp.WEIGHT.value][:inw]
            ini_ref_energy = np.average(state_energies, weights=state_weights)

        time_step = self.time_step
        num_batches = self.num_batches
        num_time_steps_batch = self.num_time_steps_batch
        target_num_walkers = self.target_num_walkers
        generator = self.core_funcs.generator(time_step,
                                              num_batches,
                                              num_time_steps_batch,
                                              ini_state,
                                              ini_ref_energy,
                                              target_num_walkers,
                                              rng_seed=self.rng_seed)
        return generator


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(qmc_base.dmc.CoreFuncs):
    """The DMC core functions for the Bloch-Phonon model."""

    model_spec: model.Spec

    @cached_property
    def recast(self):
        """Apply the periodic boundary conditions on a configuration."""
        z_min, z_max = self.model_spec.boundaries

        @nb.jit(nopython=True)
        def _recast(z: float):
            """Apply the periodic boundary conditions on a configuration.

            :param z:
            :return:
            """
            return recast_to_supercell(z, z_min, z_max)

        return _recast

    @cached_property
    def ith_diffusion(self):
        """

        :return:
        """
        pos_slot = int(self.model_spec.sys_conf_slots.pos)
        drift_slot = int(self.model_spec.sys_conf_slots.drift)
        recast = self.recast

        @nb.jit(nopython=True)
        def _ith_diffuse(i_: int, time_step: float, sys_conf: np.ndarray):
            """

            :param i_:
            :param sys_conf:
            :param time_step:
            :return:
            """
            # Alias 🙂
            normal = random.normal

            # Standard deviation as a function of time step.
            z_i = sys_conf[pos_slot, i_]
            drift_i = sys_conf[drift_slot, i_]

            # Diffuse current configuration.
            sigma = sqrt(2 * time_step)
            rnd_spread = normal(0, sigma)
            z_i_next = z_i + 2 * drift_i * time_step + rnd_spread
            z_i_next_recast = recast(z_i_next)

            return z_i_next_recast

        return _ith_diffuse

    @cached_property
    def evolve_system(self):
        """

        :return:
        """
        model_spec = self.model_spec
        cfc_spec = model_spec.cfc_spec_nt
        pos_slot = int(model_spec.sys_conf_slots.pos)
        drift_slot = int(model_spec.sys_conf_slots.drift)

        # JIT functions.
        ith_diffusion = self.ith_diffusion
        ith_energy_and_drift = model.core_funcs.ith_energy_and_drift

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True)
        def _evolve_system(sys_idx: int,
                           state_conf: np.ndarray,
                           state_energy: np.ndarray,
                           state_weight: np.ndarray,
                           aux_state_conf: np.ndarray,
                           aux_state_energy: np.ndarray,
                           aux_state_weight: np.ndarray,
                           time_step: float,
                           ref_energy: float,
                           next_state_conf: np.ndarray,
                           next_state_energy: np.ndarray,
                           next_state_weight: np.ndarray):
            """Executes the diffusion process.

            :param sys_idx: The index of the system.
            :param state_conf:
            :param state_energy:
            :param state_weight:
            :param aux_state_conf:
            :param aux_state_energy:
            :param aux_state_weight:
            :param time_step:
            :param ref_energy:
            :param next_state_conf:
            :param next_state_energy:
            :param next_state_weight:
            :return:
            """
            # Standard deviation as a function of time step.
            sigma = sqrt(2 * time_step)

            sys_conf = state_conf[sys_idx]
            aux_conf = aux_state_conf[sys_idx]
            next_conf = next_state_conf[sys_idx]

            nop = cfc_spec.model_spec.boson_number
            for i_ in range(nop):
                # Diffuse current configuration.
                z_i_next = ith_diffusion(i_, time_step, sys_conf)

                # Now we can update the position of auxiliary configuration,
                # and even of the next configuration.
                aux_conf[pos_slot, i_] = z_i_next
                next_conf[pos_slot, i_] = z_i_next

            energy = state_energy[sys_idx]
            energy_next = 0.
            for i_ in range(nop):
                ith_energy_drift = ith_energy_and_drift(i_, aux_conf, cfc_spec)
                ith_energy_next, ith_drift_next = ith_energy_drift
                energy_next += ith_energy_next
                next_conf[drift_slot, i_] = ith_drift_next

            mean_energy = (energy_next + energy) / 2
            weight_next = exp(-time_step * (mean_energy - ref_energy))

            # Copy drift slot data back to aux_conf
            aux_conf[:] = next_conf[:]

            # Update the energy of the next configuration.
            aux_state_energy[sys_idx] = energy_next
            next_state_energy[sys_idx] = energy_next

            # Update the weight of the next configuration.
            aux_state_weight[sys_idx] = weight_next
            next_state_weight[sys_idx] = weight_next

        return _evolve_system

    @cached_property
    def prepare_ini_ith_system(self):
        """

        :return:
        """
        model_spec = self.model_spec
        nop = model_spec.boson_number
        cfc_spec = model_spec.cfc_spec_nt
        pos_slot = int(model_spec.sys_conf_slots.pos)
        drift_slot = int(model_spec.sys_conf_slots.drift)

        # JIT functions.
        ith_energy_and_drift = model.core_funcs.ith_energy_and_drift

        @nb.jit(nopython=True)
        def _prepare_ini_ith_system(sys_idx: int,
                                    state_confs: np.ndarray,
                                    state_energies: np.ndarray,
                                    state_weights: np.ndarray,
                                    ini_sys_conf_set: np.ndarray):
            """Prepare a system of the initial state of the sampling.

            :param sys_idx:
            :param state_confs:
            :param state_energies:
            :param state_weights:
            :param ini_sys_conf_set:
            :return:
            """
            sys_conf = state_confs[sys_idx]
            ini_sys_conf = ini_sys_conf_set[sys_idx]
            energy_sum = 0.

            for i_ in range(nop):
                # Particle-by-particle loop.
                energy_drift = ith_energy_and_drift(i_, ini_sys_conf, cfc_spec)
                ith_energy, ith_drift = energy_drift

                sys_conf[pos_slot, i_] = ini_sys_conf[pos_slot, i_]
                sys_conf[drift_slot, i_] = ith_drift
                energy_sum += ith_energy

            # Store the energy and initialize all weights to unity
            state_energies[sys_idx] = energy_sum
            state_weights[sys_idx] = 1.

        return _prepare_ini_ith_system

    @cached_property
    def prepare_ini_state(self):
        """

        :return:
        """
        # Fields
        state_props_fields = qmc_base.dmc.StateProp
        energy_field = state_props_fields.ENERGY.value
        weight_field = state_props_fields.WEIGHT.value

        # JIT functions.
        prepare_ini_ith_system = self.prepare_ini_ith_system

        @nb.jit(nopython=True, parallel=True)
        def _prepare_ini_state(ini_sys_conf_set: np.ndarray,
                               state_confs: np.ndarray,
                               state_props: np.ndarray):
            """Prepare the initial state of the sampling.

            :param ini_sys_conf_set:
            :param state_confs:
            :param state_props:
            :return:
            """
            ini_num_walkers = len(ini_sys_conf_set)
            state_energy = state_props[energy_field]
            state_weight = state_props[weight_field]

            for sys_idx in nb.prange(ini_num_walkers):
                # Prepare each one of the configurations of the state.
                prepare_ini_ith_system(sys_idx, state_confs, state_energy,
                                       state_weight, ini_sys_conf_set)

        return _prepare_ini_state
