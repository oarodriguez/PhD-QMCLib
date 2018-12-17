import typing as t
from math import exp, sqrt

import attr
import numba as nb
import numpy as np
import numpy.ma as ma
from cached_property import cached_property
from numpy import random

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.dmc import SamplingIterData
from my_research_libs.qmc_base.utils import recast_to_supercell
from . import model

__all__ = [
    'BatchFuncResult',
    'CoreFuncs',
    'IterProp',
    'Sampling',
    'SamplingIterData',
    'State',
    'StateProp'
]


class State(qmc_base.dmc.State, t.NamedTuple):
    """"""
    confs: np.ndarray
    props: np.ndarray
    num_walkers: int
    max_num_walkers: int


class BatchFuncResult(t.NamedTuple):
    """The result of a function evaluated over a sampling batch."""
    func: np.ndarray
    iter_props: np.ndarray


StateProp = qmc_base.dmc.StateProp
IterProp = qmc_base.dmc.IterProp

state_confs_dtype = np.float64

state_props_dtype = np.dtype([
    (StateProp.ENERGY.value, np.float64),
    (StateProp.WEIGHT.value, np.float64),
    (StateProp.MASK.value, np.bool)
])

T_ExtArrays = t.Tuple[np.ndarray, ...]
T_RelDist = t.Union[t.SupportsFloat, np.ndarray]
T_Momentum = t.Union[t.SupportsFloat, np.ndarray]


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

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = utils.get_random_rng_seed()
            super().__setattr__('rng_seed', rng_seed)

        # Only take as much sys_conf items as max_num_walkers.
        # NOTE: Take the configurations counting from the last one.
        ini_sys_conf_set = self.ini_sys_conf_set[-self.max_num_walkers:]
        super().__setattr__('ini_sys_conf_set', ini_sys_conf_set)

        core_funcs = CoreFuncs.from_model_spec(self.model_spec)
        super().__setattr__('core_funcs', core_funcs)

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

    def broadcast_with_iter_batch(self, ext_arrays: T_ExtArrays,
                                  iter_batch: SamplingIterData) -> t.Tuple:
        """

        :param iter_batch:
        :param ext_arrays:
        :return:
        """
        # Broadcast the external arrays. We will use this object to
        # construct an intermediate shape used to take advantage of
        # broadcasting.
        ext_broadcast = np.broadcast(*ext_arrays)
        ext_broadcast_shape = tuple(1 for _ in ext_broadcast.shape)

        states_confs_array = iter_batch.states_confs
        states_props_array = iter_batch.states_props
        iter_props_array = iter_batch.iter_props

        spb_shape = states_props_array.shape
        ipb_shape = iter_props_array.shape
        sys_conf_shape = self.model_spec.sys_conf_shape

        # Create new shapes to take advantage of broadcasting.
        spb_bdc_shape = spb_shape + ext_broadcast_shape
        scb_bdc_shape = spb_bdc_shape + sys_conf_shape
        ipb_bdc_shape = ipb_shape + ext_broadcast_shape

        states_confs_array = states_confs_array.reshape(scb_bdc_shape)
        states_props_array = states_props_array.reshape(spb_bdc_shape)
        iter_props_array = iter_props_array.reshape(ipb_bdc_shape)

        # This array broadcasting is used to adjust the iteration
        # properties with the external arrays.
        iter_props_array, *_ = \
            np.broadcast_arrays(iter_props_array, *ext_arrays)

        # This array broadcasting is needed to adjust the mask of
        # the batch data with the external arrays.
        states_props_array, *_ext_arrays_ = \
            np.broadcast_arrays(states_props_array, *ext_arrays)

        return _ext_arrays_, SamplingIterData(states_confs_array,
                                              states_props_array,
                                              iter_props_array)

    @staticmethod
    def energy_batch(iter_data: SamplingIterData):
        """

        :param iter_data:
        :return:
        """
        state_props_fields = qmc_base.dmc.StateProp
        energy_field = state_props_fields.ENERGY.value
        weight_field = state_props_fields.WEIGHT.value
        mask_field = state_props_fields.MASK.value

        states_props_array = iter_data.states_props

        # Take the weighs and the masks.
        states_energies_array = states_props_array[energy_field]
        states_weights_array = states_props_array[weight_field]
        states_masks_array = states_props_array[mask_field]

        states_energies_array: ma.MaskedArray = \
            ma.MaskedArray(states_energies_array, mask=states_masks_array)
        states_weights_array: ma.MaskedArray = \
            ma.masked_array(states_weights_array, mask=states_masks_array)

        energy_array = states_energies_array * states_weights_array
        # NOTE: How should we do this summation?
        #   1. np.add doesn't handle masked arrays correctly.
        #   2. ndarray.sum seems to handle masked arrays correctly.
        #   3. ma.add exists. Is this a better option?
        #   .
        #   The same considerations apply for other batch functions.
        total_energy_array = energy_array.sum(axis=1)
        return BatchFuncResult(total_energy_array, iter_data.iter_props)

    def one_body_density_batch(self, rel_dist: T_RelDist,
                               iter_data: SamplingIterData,
                               result: np.ndarray = None):
        """Calculates the one-body density for a sampling batch.

        :param rel_dist:
        :param iter_data:
        :param result:
        :return:
        """
        core_funcs = self.core_funcs
        state_props_fields = qmc_base.dmc.StateProp
        weight_field = state_props_fields.WEIGHT.value
        mask_field = state_props_fields.MASK.value

        rel_dist = np.asarray(rel_dist)
        (rel_dist,), iter_data = \
            self.broadcast_with_iter_batch((rel_dist,), iter_data)

        states_confs_array = iter_data.states_confs
        states_props_array = iter_data.states_props

        # Take the weighs and the masks.
        states_weights_array: np.ndarray = states_props_array[weight_field]
        states_masks_array: np.ndarray = states_props_array[mask_field]

        # noinspection PyTypeChecker
        obd_array = core_funcs.one_body_density(rel_dist,
                                                states_confs_array,
                                                states_weights_array,
                                                states_masks_array,
                                                result)

        obd_masked_array: ma.MaskedArray = \
            ma.MaskedArray(obd_array, mask=states_masks_array)

        # Sum over the axis that indexes the walkers.
        total_obd_array = obd_masked_array.sum(axis=1)
        return BatchFuncResult(total_obd_array, iter_data.iter_props)

    def structure_factor_batch(self, momentum: T_Momentum,
                               batch_data: SamplingIterData,
                               result: np.ndarray = None) -> BatchFuncResult:
        """Evaluates the static structure factor for a sampling batch.

        :param momentum:
        :param batch_data:
        :param result:
        :return:
        """
        core_funcs = self.core_funcs
        state_props_fields = qmc_base.dmc.StateProp
        weight_field = state_props_fields.WEIGHT.value
        mask_field = state_props_fields.MASK.value

        momentum = np.asarray(momentum)
        (momentum,), batch_data = \
            self.broadcast_with_iter_batch((momentum,), batch_data)

        states_confs_array = batch_data.states_confs
        states_props_array = batch_data.states_props

        # Take the weighs and the masks.
        states_weights_array: np.ndarray = states_props_array[weight_field]
        states_masks_array: np.ndarray = states_props_array[mask_field]

        # noinspection PyTypeChecker
        sf_array = core_funcs.structure_factor(momentum,
                                               states_confs_array,
                                               states_weights_array,
                                               states_masks_array,
                                               result)

        # Mask the resulting array
        sf_masked_array: ma.MaskedArray = \
            ma.MaskedArray(sf_array, mask=states_masks_array)

        # Sum over the axis that indexes the walkers.
        total_sf_array = sf_masked_array.sum(axis=1)
        return BatchFuncResult(total_sf_array, batch_data.iter_props)

    #
    def __iter__(self) -> \
            t.Generator[qmc_base.dmc.SamplingIterData, t.Any, None]:
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

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """The sampling core functions."""
        return CoreFuncs.from_model_spec(self.model_spec)


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(qmc_base.dmc.CoreFuncs):
    """The DMC core functions for the Bloch-Phonon model."""

    #: The boundaries of the QMC supercell.
    boundaries: t.Tuple[float, float]

    #: The slots of a system configuration array.
    sys_conf_slots: model.Spec.sys_conf_slots

    #: The common (fixed) spec to pass to the core functions of the model.
    cfc_spec_nt: model.CFCSpecNT

    @classmethod
    def from_model_spec(cls, model_spec: model.Spec):
        """Initializes the core functions from a model spec.

        :param model_spec: The model spec.
        :return: An instance of the core functions.
        """
        return cls(model_spec.boundaries,
                   model_spec.sys_conf_slots,
                   model_spec.cfc_spec_nt)

    @cached_property
    def recast(self):
        """Apply the periodic boundary conditions on a configuration."""
        z_min, z_max = self.boundaries

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
        pos_slot = int(self.sys_conf_slots.pos)
        drift_slot = int(self.sys_conf_slots.drift)
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
        cfc_spec = self.cfc_spec_nt
        pos_slot = int(self.sys_conf_slots.pos)
        drift_slot = int(self.sys_conf_slots.drift)

        # JIT functions.
        ith_diffusion = self.ith_diffusion
        ith_energy_and_drift = model.core_funcs.ith_energy_and_drift

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True)
        def _evolve_system(sys_idx: int,
                           actual_state_confs: np.ndarray,
                           actual_state_energies: np.ndarray,
                           actual_state_weights: np.ndarray,
                           time_step: float,
                           ref_energy: float,
                           next_state_confs: np.ndarray,
                           next_state_energies: np.ndarray,
                           next_state_weights: np.ndarray):
            """Executes the diffusion process.

            :param sys_idx: The index of the system.
            :param actual_state_confs:
            :param actual_state_energies:
            :param actual_state_weights:
            :param time_step:
            :param ref_energy:
            :param next_state_confs:
            :param next_state_energies:
            :param next_state_weights:
            :return:
            """
            # Standard deviation as a function of time step.
            # sigma = sqrt(2 * time_step)

            sys_conf = actual_state_confs[sys_idx]
            next_conf = next_state_confs[sys_idx]

            nop = cfc_spec.model_spec.boson_number
            for i_ in range(nop):
                # Diffuse current configuration. We can update the position
                # of the next configuration.
                z_i_next = ith_diffusion(i_, time_step, sys_conf)
                next_conf[pos_slot, i_] = z_i_next

            energy = actual_state_energies[sys_idx]
            energy_next = 0.
            for i_ in range(nop):
                ith_energy_drift = ith_energy_and_drift(i_, next_conf,
                                                        cfc_spec)
                ith_energy_next, ith_drift_next = ith_energy_drift
                next_conf[drift_slot, i_] = ith_drift_next
                energy_next += ith_energy_next

            mean_energy = (energy_next + energy) / 2
            weight_next = exp(-time_step * (mean_energy - ref_energy))

            # Update the energy and weight of the next configuration.
            next_state_energies[sys_idx] = energy_next
            next_state_weights[sys_idx] = weight_next

        return _evolve_system

    @cached_property
    def prepare_ini_ith_system(self):
        """Prepare a system of the initial state of the sampling."""

        cfc_spec = self.cfc_spec_nt
        nop = cfc_spec.model_spec.boson_number
        pos_slot = int(self.sys_conf_slots.pos)
        drift_slot = int(self.sys_conf_slots.drift)

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
        """Prepare the initial state of the sampling. """

        # Fields
        state_props_fields = qmc_base.dmc.StateProp
        energy_field = state_props_fields.ENERGY.value
        weight_field = state_props_fields.WEIGHT.value
        mask_field = state_props_fields.MASK.value

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
            state_mask = state_props[mask_field]

            # Initialize the mask.
            state_mask[:] = True

            for sys_idx in nb.prange(ini_num_walkers):
                # Prepare each one of the configurations of the state.
                prepare_ini_ith_system(sys_idx, state_confs, state_energy,
                                       state_weight, ini_sys_conf_set)

                # Unmask this walker.
                state_mask[sys_idx] = False

        return _prepare_ini_state

    @cached_property
    def one_body_density(self):
        """"""

        types = ['void(f8,f8[:,:],f8,b1,f8[:])']
        signature = '(),(ns,nop),(),() -> ()'
        cfc_spec = self.cfc_spec_nt

        one_body_density = model.core_funcs.one_body_density

        @nb.guvectorize(types, signature, nopython=True, target='parallel')
        def _one_body_density(rel_dist: float,
                              sys_conf: np.ndarray,
                              sys_weight: float,
                              sys_mask: bool,
                              result: np.ndarray):
            """

            :param sys_conf:
            :param sys_weight:
            :param sys_mask:
            :param result:
            :return:
            """
            if not sys_mask:
                sys_obd = one_body_density(rel_dist, sys_conf, cfc_spec)
                result[0] = sys_weight * sys_obd
            else:
                result[0] = 0.

        return _one_body_density

    @cached_property
    def structure_factor(self):
        """The weighed structure factor."""

        types = ['void(f8,f8[:,:],f8,b1,f8[:])']
        signature = '(),(ns,nop),(),() -> ()'
        cfc_spec = self.cfc_spec_nt

        structure_factor = model.core_funcs.structure_factor

        # noinspection PyTypeChecker
        @nb.guvectorize(types, signature, nopython=True, target='parallel')
        def _structure_factor(momentum: float,
                              sys_conf: np.ndarray,
                              sys_weight: float,
                              sys_mask: bool,
                              result: np.ndarray) -> np.ndarray:
            """

            :param sys_conf:
            :param sys_weight:
            :param sys_mask:
            :param result:
            :return:
            """
            # NOTE: We need if... else... to avoid bugs.
            if not sys_mask:
                sys_sf = structure_factor(momentum, sys_conf, cfc_spec)
                result[0] = sys_weight * sys_sf
            else:
                result[0] = 0.

        return _structure_factor
