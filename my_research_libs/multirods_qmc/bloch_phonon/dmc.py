import typing as t
from math import exp, pi, sqrt

import attr
import numba as nb
import numpy as np
import numpy.ma as ma
from cached_property import cached_property
from numpy import random

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.dmc import SamplingBatch, branching_spec_dtype
from my_research_libs.qmc_base.jastrow import SysConfSlot
from my_research_libs.qmc_base.utils import recast_to_supercell
from . import model

__all__ = [
    'BatchFuncResult',
    'CoreFuncs',
    'CoreFuncsBase',
    'EstSampling',
    'EstSamplingCoreFuncs',
    'IterProp',
    'Sampling',
    'SamplingBase',
    'State',
    'StateError',
    'StateProp',
    'SSFEstSpec'
]

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


class State(qmc_base.dmc.State, t.NamedTuple):
    """"""
    confs: np.ndarray
    props: np.ndarray
    energy: float
    weight: float
    num_walkers: int
    ref_energy: float
    accum_energy: float
    max_num_walkers: int
    branching_spec: t.Optional[np.ndarray] = None


class BatchFuncResult(t.NamedTuple):
    """The result of a function evaluated over a sampling batch."""
    func: np.ndarray
    iter_props: np.ndarray


class StateError(ValueError):
    """Flags errors related to the handling of a DMC state."""
    pass


class SamplingBase(qmc_base.dmc.Sampling):
    """A class to realize a DMC sampling."""
    __slots__ = ()

    #: The model instance.
    model_spec: model.Spec

    time_step: float
    max_num_walkers: int
    target_num_walkers: int
    num_walkers_control_factor: t.Optional[float]
    rng_seed: t.Optional[int] = None

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

    def build_state(self, sys_conf_set: np.ndarray,
                    ref_energy: float = None) -> State:
        """Builds a state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.

        :param sys_conf_set:
        :param ref_energy:
        :return:
        """
        confs_shape = self.state_confs_shape
        props_shape = self.state_props_shape
        max_num_walkers = self.max_num_walkers

        if len(confs_shape) == len(sys_conf_set.shape):
            # Equal number of dimensions, but...
            if confs_shape[1:] != self.model_spec.sys_conf_shape:
                raise StateError("sys_conf_set is not a valid set of "
                                 "configurations of the model spec")

        # Only take as much sys_conf items as target_num_walkers.
        sys_conf_set = np.asarray(sys_conf_set)[-self.target_num_walkers:]
        num_walkers = len(sys_conf_set)

        # Initial state arrays.
        state_confs = np.zeros(confs_shape, dtype=state_confs_dtype)
        state_props = np.zeros(props_shape, dtype=state_props_dtype)

        # Calculate the initial state arrays properties.
        self.core_funcs.prepare_ini_state(sys_conf_set, state_confs,
                                          state_props)

        state_energies = state_props[StateProp.ENERGY][:num_walkers]
        state_weights = state_props[StateProp.WEIGHT][:num_walkers]

        state_energy = (state_energies * state_weights).sum()
        state_weight = state_weights.sum()
        energy = state_energy / state_weight

        if ref_energy is None:
            # Calculate the initial energy of reference as the
            # average of the energy of the initial state.
            ref_energy = energy

        # Table to control the branching process.
        branching_spec = \
            np.zeros(max_num_walkers, dtype=branching_spec_dtype)

        # NOTE: The branching spec for the initial state is None.
        return qmc_base.dmc.State(confs=state_confs,
                                  props=state_props,
                                  energy=state_energy,
                                  weight=state_weight,
                                  num_walkers=num_walkers,
                                  ref_energy=ref_energy,
                                  accum_energy=energy,
                                  max_num_walkers=max_num_walkers,
                                  branching_spec=branching_spec)

    def broadcast_with_iter_batch(self, ext_arrays: T_ExtArrays,
                                  iter_batch: SamplingBatch) -> t.Tuple:
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

        return _ext_arrays_, SamplingBatch(states_confs_array,
                                           states_props_array,
                                           iter_props_array)

    @staticmethod
    def energy_batch(iter_data: SamplingBatch):
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
                               iter_data: SamplingBatch,
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
                               batch_data: SamplingBatch,
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

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """The sampling core functions."""
        return CoreFuncs.from_model_spec(self.model_spec)


class CoreFuncsBase(qmc_base.dmc.CoreFuncs):
    """The DMC core functions for the Bloch-Phonon model."""
    __slots__ = ()

    #: The boundaries of the QMC supercell.
    boundaries: t.Tuple[float, float]

    #: The common (fixed) spec to pass to the core functions of the model.
    cfc_spec_nt: model.CFCSpecNT

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
        pos_slot = int(SysConfSlot.pos)
        drift_slot = int(SysConfSlot.drift)
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
        pos_slot = int(SysConfSlot.pos)
        drift_slot = int(SysConfSlot.drift)

        # JIT functions.
        ith_diffusion = self.ith_diffusion
        ith_energy_and_drift = model.core_funcs.ith_energy_and_drift

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True)
        def _evolve_system(sys_idx: int,
                           cloning_ref_idx: int,
                           prev_state_confs: np.ndarray,
                           prev_state_energies: np.ndarray,
                           prev_state_weights: np.ndarray,
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
            prev_conf = prev_state_confs[cloning_ref_idx]
            sys_conf = actual_state_confs[sys_idx]
            next_conf = next_state_confs[sys_idx]

            nop = cfc_spec.model_spec.boson_number
            for i_ in range(nop):
                # Diffuse current configuration. We can update the position
                # of the next configuration.
                z_i_next = ith_diffusion(i_, time_step, prev_conf)
                sys_conf[pos_slot, i_] = z_i_next
                next_conf[pos_slot, i_] = z_i_next

            energy = actual_state_energies[sys_idx]
            energy_next = 0.
            for i_ in range(nop):
                ith_energy_drift = ith_energy_and_drift(i_, sys_conf,
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
        pos_slot = int(SysConfSlot.pos)
        drift_slot = int(SysConfSlot.drift)

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


@attr.s(auto_attribs=True, frozen=True)
class Sampling(SamplingBase):
    """A class to realize a DMC sampling."""

    #: The model instance.
    model_spec: model.Spec

    time_step: float
    max_num_walkers: int = 512
    target_num_walkers: int = 480
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            super().__setattr__('rng_seed', rng_seed)

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """The sampling core functions."""
        return CoreFuncs.from_model_spec(self.model_spec)


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(CoreFuncsBase):
    """The DMC core functions for the Bloch-Phonon model."""

    boundaries: t.Tuple[float, float]
    cfc_spec_nt: model.CFCSpecNT

    @classmethod
    def from_model_spec(cls, model_spec: model.Spec):
        """Initializes the core functions from a model spec.

        :param model_spec: The model spec.
        :return: An instance of the core functions.
        """
        return cls(model_spec.boundaries,
                   model_spec.cfc_spec_nt)


class SSFEstSpecNT(qmc_base.dmc.SSFEstSpecNT, t.NamedTuple):
    """"""
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: int = None
    core_func: t.Callable = qmc_base.dmc.dummy_pure_est_core_func


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(qmc_base.dmc.SSFEstSpec):
    """Structure factor estimator."""

    model_spec: model.Spec
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""

        if self.pfw_num_time_steps is None:
            # A very large integer 🤔.
            pfs_nts = 99999999
            object.__setattr__(self, 'pfw_num_time_steps', pfs_nts)

    @cached_property
    def momenta(self):
        """

        :return:
        """
        num_modes = self.num_modes
        supercell_size = self.model_spec.supercell_size
        return np.arange(1, num_modes + 1) * 2 * pi / supercell_size

    @cached_property
    def core_func(self):
        """

        :return:
        """
        model_spec = self.model_spec
        num_modes = self.num_modes
        as_pure_est = self.as_pure_est
        pfs_nts = self.pfw_num_time_steps
        momenta = self.momenta

        cfc_spec_nt = model_spec.cfc_spec_nt
        ssf_func = model.core_funcs.structure_factor

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True)
        def _core_func(step_idx: int,
                       sys_idx: int,
                       clone_ref_idx: int,
                       state_confs: np.ndarray,
                       iter_ssf_array: np.ndarray,
                       aux_states_ssf_array: np.ndarray):
            """

            :param step_idx:
            :param sys_idx:
            :param clone_ref_idx:
            :param state_confs:
            :param iter_ssf_array:
            :param aux_states_ssf_array:
            :return:
            """
            prev_step_idx = step_idx % 2 - 1
            actual_step_idx = step_idx % 2

            prev_state_ssf = aux_states_ssf_array[prev_step_idx]
            actual_state_ssf = aux_states_ssf_array[actual_step_idx]

            prev_sys_sk = prev_state_ssf[clone_ref_idx]
            sys_conf = state_confs[sys_idx]

            if not as_pure_est:
                # Mixed estimator.

                for kz_idx in range(num_modes):
                    momentum = momenta[kz_idx]
                    sys_sk_idx = \
                        ssf_func(momentum, sys_conf, cfc_spec_nt)

                    # Just update the actual state.
                    actual_state_ssf[sys_idx, kz_idx] = sys_sk_idx

                # Finish.
                return

            # Pure estimator.
            if step_idx >= pfs_nts:
                # Just "transport" the structure factor of the previous
                # configuration to the new one.
                for kz_idx in range(num_modes):
                    actual_state_ssf[sys_idx, kz_idx] = prev_sys_sk[kz_idx]

            else:
                # Evaluate the structure factor for the actual
                # system configuration.
                for kz_idx in range(num_modes):
                    momentum = momenta[kz_idx]
                    sys_sk_idx = \
                        ssf_func(momentum, sys_conf, cfc_spec_nt)

                    # Update with the previous state ("transport").
                    actual_state_ssf[sys_idx, kz_idx] = \
                        sys_sk_idx + prev_sys_sk[kz_idx]

        return _core_func


@attr.s(auto_attribs=True, frozen=True)
class EstSampling(SamplingBase, qmc_base.dmc.EstSampling):
    """Class to evaluate estimators using a DMC sampling."""

    model_spec: model.Spec
    time_step: float
    max_num_walkers: int = 512
    target_num_walkers: int = 480
    num_walkers_control_factor: t.Optional[float] = 0.5
    rng_seed: t.Optional[int] = None

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            object.__setattr__(self, 'rng_seed', rng_seed)

    @cached_property
    def core_funcs(self) -> 'EstSamplingCoreFuncs':
        """"""
        model_spec = self.model_spec
        boundaries = model_spec.boundaries
        cfc_spec_nt = model_spec.cfc_spec_nt

        ssf_spec = self.ssf_spec
        if ssf_spec is not None:
            ssf_num_modes = ssf_spec.num_modes
            ssf_as_pure_est = ssf_spec.as_pure_est
            ssf_pfw_nts = ssf_spec.pfw_num_time_steps
            ssf_core_func = ssf_spec.core_func

        else:
            ssf_num_modes = 1
            ssf_as_pure_est = False
            ssf_pfw_nts = 512
            ssf_core_func = None

        sf_spec_nt = SSFEstSpecNT(ssf_num_modes,
                                  ssf_as_pure_est,
                                  ssf_pfw_nts,
                                  ssf_core_func)

        return EstSamplingCoreFuncs(boundaries,
                                    cfc_spec_nt,
                                    sf_spec_nt)


@attr.s(auto_attribs=True, frozen=True)
class EstSamplingCoreFuncs(CoreFuncsBase, qmc_base.dmc.EstSamplingCoreFuncs):
    """Core functions to evaluate estimators using a DMC sampling."""

    boundaries: t.Tuple[float, float]
    cfc_spec_nt: model.CFCSpecNT
    ssf_spec_nt: SSFEstSpecNT = None

    def __attrs_post_init__(self):
        """"""
        pass
