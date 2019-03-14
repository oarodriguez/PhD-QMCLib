import typing as t
from math import pi, sqrt

import attr
import numba as nb
import numpy as np
import numpy.ma as ma
from cached_property import cached_property

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.dmc import (
    SSFExecData, SSFPartSlot, SamplingConfsPropsBatch,
    branching_spec_dtype
)
from my_research_libs.qmc_base.jastrow import dmc as jsw_dmc
from my_research_libs.qmc_base.utils import recast_to_supercell
from my_research_libs.util.attr import Record
from . import model

__all__ = [
    'BatchFuncResult',
    'CoreFuncs',
    'IterProp',
    'Sampling',
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


@attr.s(auto_attribs=True)
class DDFParams(jsw_dmc.DDFParams, Record):
    """The parameters of the diffusion-and-drift process."""
    boson_number: int
    time_step: float
    sigma_spread: float
    lower_bound: float
    upper_bound: float


@attr.s(auto_attribs=True)
class SSFParams(jsw_dmc.SSFParams, Record):
    """Static structure factor parameters."""
    num_modes: int
    as_pure_est: bool
    pfw_num_time_steps: int
    assume_none: bool


class CFCSpec(jsw_dmc.CFCSpec, t.NamedTuple):
    """The spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    ddf_params: DDFParams
    ssf_params: t.Optional[jsw_dmc.SSFParams] = None


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(qmc_base.dmc.SSFEstSpec):
    """Structure factor estimator."""

    # model_spec: model.Spec
    num_modes: int
    as_pure_est: bool = True
    pfw_num_time_steps: t.Optional[int] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""

        if self.pfw_num_time_steps is None:
            # A very large integer 🤔.
            pfs_nts = 99999999
            object.__setattr__(self, 'pfw_num_time_steps', pfs_nts)


@attr.s(auto_attribs=True, frozen=True)
class Sampling(jsw_dmc.Sampling):
    """A class to realize a DMC sampling."""

    #: The model instance.
    model_spec: model.Spec

    time_step: float
    max_num_walkers: int
    target_num_walkers: int
    num_walkers_control_factor: t.Optional[float] = None
    rng_seed: t.Optional[int] = None

    # *** Estimators configuration ***
    ssf_est_spec: t.Optional[SSFEstSpec] = None
    jit_parallel: bool = True
    jit_fastmath: bool = False

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            super().__setattr__('rng_seed', rng_seed)

    @property
    def ddf_params(self) -> DDFParams:
        """Represent the diffusion-and-drift process parameters."""
        model_spec = self.model_spec
        boson_number = model_spec.boson_number
        time_step = self.time_step
        sigma_spread = sqrt(2 * time_step)
        z_min, z_max = model_spec.boundaries
        ddf_params = DDFParams(boson_number,
                               time_step=time_step,
                               sigma_spread=sigma_spread,
                               lower_bound=z_min,
                               upper_bound=z_max)
        return ddf_params.as_record()

    @property
    def ssf_params(self) -> SSFParams:
        """Static structure factor parameters.

        :return:
        """
        if self.ssf_est_spec is None:
            num_modes = 1
            as_pure_est = False
            pfw_nts = 1
            assume_none = True
        else:
            num_modes = self.ssf_est_spec.num_modes
            as_pure_est = self.ssf_est_spec.as_pure_est
            pfw_nts = self.ssf_est_spec.pfw_num_time_steps
            assume_none = False

        ssf_params = \
            SSFParams(num_modes, as_pure_est, pfw_nts, assume_none)
        return ssf_params.as_record()

    @property
    def cfc_spec(self) -> CFCSpec:
        """"""
        model_spec = self.model_spec
        return CFCSpec(model_spec.params,
                       model_spec.obf_params,
                       model_spec.tbf_params,
                       self.ddf_params,
                       self.ssf_params)

    @property
    def ssf_momenta(self):
        """Get the momenta to evaluate the static structure factor.

        :return:
        """
        if self.ssf_est_spec is None:
            raise TypeError('the static structure factor spec has no been '
                            'specified')
        else:
            model_spec = self.model_spec
            ssf_est_spec = self.ssf_est_spec
            num_modes = ssf_est_spec.num_modes
            supercell_size = model_spec.supercell_size
            return np.arange(num_modes) * 2 * pi / supercell_size

    def build_state(self, sys_conf_set: np.ndarray,
                    ref_energy: float = None) -> State:
        """Builds a state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.

        :param sys_conf_set:
        :param ref_energy:
        :return:
        """
        cfc_spec = self.cfc_spec
        confs_shape = self.state_confs_shape
        props_shape = self.state_props_shape
        max_num_walkers = self.max_num_walkers
        model_params = cfc_spec.model_params
        obf_params = cfc_spec.obf_params
        tbf_params = cfc_spec.tbf_params

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
                                          state_props, model_params,
                                          obf_params, tbf_params)

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
                                  iter_batch: SamplingConfsPropsBatch) -> \
            t.Tuple:
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

        return _ext_arrays_, SamplingConfsPropsBatch(states_confs_array,
                                                     states_props_array,
                                                     iter_props_array)

    @staticmethod
    def energy_batch(iter_data: SamplingConfsPropsBatch):
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
                               iter_data: SamplingConfsPropsBatch,
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

    def fourier_density_batch(self, momentum: T_Momentum,
                              batch_data: SamplingConfsPropsBatch,
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
        fdk_array = core_funcs.fourier_density(momentum,
                                               states_confs_array,
                                               states_weights_array,
                                               states_masks_array,
                                               result)

        # Mask the resulting array
        fdk_masked_array: ma.MaskedArray = \
            ma.MaskedArray(fdk_array, mask=states_masks_array)

        # Sum over the axis that indexes the walkers.
        total_fdk_array = fdk_masked_array.sum(axis=1)
        return BatchFuncResult(total_fdk_array, batch_data.iter_props)

    @cached_property
    def core_funcs(self) -> 'CoreFuncs':
        """"""
        core_funcs_id = self.jit_parallel, self.jit_fastmath
        return core_funcs_table[core_funcs_id]


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(jsw_dmc.CoreFuncs):
    """The DMC core functions for the Bloch-Phonon model."""

    #: Parallel the execution where possible.
    jit_parallel: bool = True

    #: Use fastmath compiler directive.
    jit_fastmath: bool = True

    @cached_property
    def model_core_funcs(self) -> model.CoreFuncs:
        """"""
        return model.core_funcs

    @cached_property
    def recast(self):
        """Apply the periodic boundary conditions on a configuration."""
        fastmath = self.jit_fastmath

        @nb.jit(nopython=True, fastmath=fastmath)
        def _recast(z: float, ddf_params: DDFParams):
            """Apply the periodic boundary conditions on a configuration.

            :param z:
            :param ddf_params:
            :return:
            """
            z_min = ddf_params.lower_bound
            z_max = ddf_params.upper_bound
            return recast_to_supercell(z, z_min, z_max)

        return _recast

    @cached_property
    def init_ssf_est_data(self):
        """

        :return:
        """
        # noinspection PyTypeChecker
        ssf_num_parts = len(SSFPartSlot)

        @nb.jit(nopython=True)
        def _init_ssf_est_data_stub(num_time_steps_batch: int,
                                    max_num_walkers: int,
                                    cfc_spec: CFCSpec) -> SSFExecData:
            """

            :param num_time_steps_batch:
            :param max_num_walkers:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            ssf_params = cfc_spec.ssf_params
            # Alias 😃...
            nts_batch = num_time_steps_batch
            supercell_size = model_params.supercell_size
            if ssf_params.assume_none:
                # A fake array will be created. It won't be used.
                nts_batch, num_modes = 1, 1
            else:
                num_modes = ssf_params.num_modes

            # The shape of the structure factor array.
            i_ssf_shape = nts_batch, num_modes, ssf_num_parts

            # The shape of the auxiliary arrays to store the structure
            # factor of a single state during the forward walking process.
            pfw_aux_ssf_b_shape = \
                2, max_num_walkers, num_modes, ssf_num_parts

            ssf_momenta = np.arange(num_modes) * 2 * pi / supercell_size
            iter_ssf_array = np.empty(i_ssf_shape, dtype=np.float64)
            pfw_aux_ssf_array = \
                np.zeros(pfw_aux_ssf_b_shape, dtype=np.float64)
            return SSFExecData(ssf_momenta,
                               iter_ssf_array,
                               pfw_aux_ssf_array)

        return _init_ssf_est_data_stub

    # @cached_property
    # def one_body_density_guv(self):
    #     """"""
    #
    #     types = ['void(f8,f8[:,:],f8,b1,f8[:])']
    #     signature = '(),(ns,nop),(),() -> ()'
    #     cfc_spec = self.cfc_spec_nt
    #
    #     one_body_density = model.core_funcs.one_body_density
    #
    #     @nb.guvectorize(types, signature, nopython=True, target='parallel')
    #     def _one_body_density(rel_dist: float,
    #                           sys_conf: np.ndarray,
    #                           sys_weight: float,
    #                           sys_mask: bool,
    #                           result: np.ndarray):
    #         """
    #
    #         :param sys_conf:
    #         :param sys_weight:
    #         :param sys_mask:
    #         :param result:
    #         :return:
    #         """
    #         if not sys_mask:
    #             sys_obd = one_body_density(rel_dist, sys_conf, cfc_spec)
    #             result[0] = sys_weight * sys_obd
    #         else:
    #             result[0] = 0.
    #
    #     return _one_body_density

    # @cached_property
    # def fourier_density_guv(self):
    #     """The weighed structure factor."""
    #
    #     types = ['void(f8,f8[:,:],f8,b1,c16[:])']
    #     signature = '(),(ns,nop),(),() -> ()'
    #     cfc_spec = self.cfc_spec_nt
    #
    #     fourier_density = model.core_funcs.fourier_density
    #
    #     # noinspection PyTypeChecker
    #     @nb.guvectorize(types, signature, nopython=True, target='parallel')
    #     def _fourier_density(momentum: float,
    #                          sys_conf: np.ndarray,
    #                          sys_weight: float,
    #                          sys_mask: bool,
    #                          result: np.ndarray) -> np.ndarray:
    #         """
    #
    #         :param sys_conf:
    #         :param sys_weight:
    #         :param sys_mask:
    #         :param result:
    #         :return:
    #         """
    #         # NOTE: We need if... else... to avoid bugs.
    #         if not sys_mask:
    #             sys_fdk = fourier_density(momentum, sys_conf, cfc_spec)
    #             result[0] = sys_weight * sys_fdk
    #         else:
    #             result[0] = 0.
    #
    #     return _fourier_density


# Variants of the core functions.
par_fast_core_funcs = CoreFuncs(jit_parallel=True, jit_fastmath=True)
par_nofast_core_funcs = CoreFuncs(jit_parallel=True, jit_fastmath=False)
nopar_fast_core_funcs = CoreFuncs(jit_parallel=False, jit_fastmath=False)
nopar_nofast_core_funcs = CoreFuncs(jit_parallel=False, jit_fastmath=False)

# Table to organize the core functions.
core_funcs_table = {
    (True, True): par_fast_core_funcs,
    (True, False): par_nofast_core_funcs,
    (False, True): nopar_fast_core_funcs,
    (False, False): nopar_nofast_core_funcs
}
