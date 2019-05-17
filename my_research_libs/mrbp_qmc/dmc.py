import typing as t
from math import pi, sqrt

import attr
import numba as nb
import numpy as np
from cached_property import cached_property

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.dmc import (
    DensityExecData, SSFExecData, SSFPartSlot, branching_spec_dtype
)
from my_research_libs.qmc_base.jastrow import (
    dmc as jsw_dmc, model as jsw_model
)
from my_research_libs.qmc_base.utils import recast_to_supercell
from my_research_libs.util.attr import (
    Record, bool_converter, bool_validator, int_converter, int_validator
)
from . import model

__all__ = [
    'CoreFuncs',
    'Sampling',
    'State',
    'StateError',
    'SSFEstSpec'
]

StateProp = qmc_base.dmc.StateProp

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
class DensityParams(jsw_dmc.DensityParams, Record):
    """Static structure factor parameters."""
    num_bins: int
    as_pure_est: bool
    pfw_num_time_steps: int
    assume_none: bool


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
    density_params: t.Optional[DensityParams]
    ssf_params: t.Optional[jsw_dmc.SSFParams] = None


@attr.s(auto_attribs=True, frozen=True)
class DensityEstSpec(qmc_base.dmc.DensityEstSpec):
    """Structure factor estimator."""

    # model_spec: model.Spec
    num_bins: int = attr.ib(converter=int_converter, validator=int_validator)
    as_pure_est: bool = attr.ib(default=True,
                                converter=bool_converter,
                                validator=bool_validator)
    pfw_num_time_steps: int = attr.ib(default=99999999,
                                      converter=int_converter,
                                      validator=int_validator)

    def __attrs_post_init__(self):
        """Post-initialization stage."""

        if self.pfw_num_time_steps is None:
            # A very large integer 🤔.
            pfs_nts = 99999999
            object.__setattr__(self, 'pfw_num_time_steps', pfs_nts)


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
    density_est_spec: t.Optional[DensityEstSpec] = None
    ssf_est_spec: t.Optional[SSFEstSpec] = None
    jit_parallel: bool = True
    jit_fastmath: bool = False

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            object.__setattr__(self, 'rng_seed', rng_seed)

        nwc_factor = self.num_walkers_control_factor
        if nwc_factor is None:
            object.__setattr__(self, 'num_walkers_control_factor', 1.25e-1)

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
    def density_params(self) -> DensityParams:
        """"""
        density_est_spec = self.density_est_spec
        if density_est_spec is None:
            num_bins = 1
            as_pure_est = False
            pfw_nts = 1
            assume_none = True
        else:
            num_bins = density_est_spec.num_bins
            as_pure_est = density_est_spec.as_pure_est
            pfw_nts = density_est_spec.pfw_num_time_steps
            assume_none = False

        density_params = \
            DensityParams(num_bins, as_pure_est, pfw_nts, assume_none)
        return density_params.as_record()

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
                       self.density_params,
                       self.ssf_params)

    @property
    def density_bins_edges(self) -> np.ndarray:
        """

        :return:
        """
        density_est_spec = self.density_est_spec
        if density_est_spec is None:
            raise TypeError('the density spec has no been specified')
        model_spec = self.model_spec
        supercell_size = model_spec.supercell_size
        num_bins = density_est_spec.num_bins
        return np.linspace(0, supercell_size, num_bins + 1)

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
        max_num_walkers = self.max_num_walkers

        if len(confs_shape) == len(sys_conf_set.shape):
            # Equal number of dimensions, but...
            if confs_shape[1:] != self.model_spec.sys_conf_shape:
                raise StateError("sys_conf_set is not a valid set of "
                                 "configurations of the model spec")

        # Only take as much sys_conf items as target_num_walkers.
        sys_conf_set = np.asarray(sys_conf_set)[-self.target_num_walkers:]
        num_walkers = len(sys_conf_set)

        state_data = \
            self.core_funcs.init_state_data(max_num_walkers, cfc_spec)
        state_props = state_data.props

        # Calculate the initial state arrays properties.
        self.core_funcs.prepare_state_data(sys_conf_set,
                                           state_data,
                                           cfc_spec)

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
        ini_state = \
            self.core_funcs.build_state(state_data,
                                        state_energy,
                                        state_weight,
                                        num_walkers,
                                        ref_energy,
                                        energy,
                                        max_num_walkers,
                                        branching_spec)
        return ini_state

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
    def density_core(self):
        """

        :return:
        """

        fastmath = self.jit_fastmath

        # Slots to save data to evaluate the density.
        density_slot = int(jsw_model.DensityPartSlot.MAIN)
        pos_slot = int(jsw_model.SysConfSlot.pos)

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, fastmath=fastmath)
        def _density_core(step_idx: int,
                          sys_idx: int,
                          clone_ref_idx: int,
                          state_confs: np.ndarray,
                          model_params: model.Params,
                          obf_params: model.OBFParams,
                          tbf_params: model.TBFParams,
                          density_params: DensityParams,
                          iter_density_array: np.ndarray,
                          aux_states_density_array: np.ndarray):
            """

            :param step_idx:
            :param sys_idx:
            :param clone_ref_idx:
            :param state_confs:
            :param iter_density_array:
            :param aux_states_density_array:
            :return:
            """
            prev_step_idx = step_idx % 2 - 1
            actual_step_idx = step_idx % 2

            prev_state_density = aux_states_density_array[prev_step_idx]
            actual_state_density = aux_states_density_array[actual_step_idx]

            # System to be "moved-forward" and current system.
            prev_sys_density = prev_state_density[clone_ref_idx]
            actual_sys_density = actual_state_density[sys_idx]
            sys_conf = state_confs[sys_idx]

            # Density parameters.
            num_bins = density_params.num_bins
            pfw_nts = density_params.pfw_num_time_steps

            # The bin size.
            boson_number = model_params.boson_number
            bin_size = model_params.supercell_size / num_bins

            if not density_params.as_pure_est:
                # Mixed estimator.
                for idx in range(boson_number):
                    # Just update the actual state.
                    z_i = sys_conf[pos_slot, idx]
                    bin_idx = int(z_i // bin_size)
                    actual_sys_density[bin_idx, density_slot] += 1

                # Finish.
                return

            # Pure estimator.
            if step_idx >= pfw_nts:
                pass

            else:
                # Evaluate the density for the actual # system configuration.
                for idx in range(boson_number):
                    z_i = sys_conf[pos_slot, idx]
                    bin_idx = int(z_i // bin_size)
                    actual_sys_density[bin_idx, density_slot] += 1

        return _density_core

    @cached_property
    def init_density_est_data(self):
        """

        :return:
        """
        # noinspection PyTypeChecker
        density_num_parts = len(jsw_model.DensityPartSlot)

        @nb.jit(nopython=True)
        def _init_density_est_data(num_time_steps_block: int,
                                   max_num_walkers: int,
                                   cfc_spec: CFCSpec) -> DensityExecData:
            """Initialize the buffers to store the density data.

            :param num_time_steps_block:
            :param max_num_walkers:
            :param cfc_spec:
            :return:
            """
            density_params = cfc_spec.density_params
            # Alias 😃...
            nts_block = num_time_steps_block
            if density_params.assume_none:
                # A fake array will be created. It won't be used.
                nts_block, num_bins = 1, 1
            else:
                num_bins = density_params.num_bins

            # The shape of the density array.
            i_density_shape = nts_block, num_bins, density_num_parts

            # The shape of the auxiliary arrays to store the density of a
            # single state during the forward walking process.
            pfw_aux_density_b_shape = \
                2, max_num_walkers, num_bins, density_num_parts

            iter_density_array = np.empty(i_density_shape, dtype=np.float64)
            pfw_aux_density_array = \
                np.zeros(pfw_aux_density_b_shape, dtype=np.float64)
            return DensityExecData(iter_density_array,
                                   pfw_aux_density_array)

        return _init_density_est_data

    @cached_property
    def init_ssf_est_data(self):
        """

        :return:
        """
        # noinspection PyTypeChecker
        ssf_num_parts = len(SSFPartSlot)

        @nb.jit(nopython=True)
        def _init_ssf_est_data(num_time_steps_block: int,
                               max_num_walkers: int,
                               cfc_spec: CFCSpec) -> SSFExecData:
            """

            :param num_time_steps_block:
            :param max_num_walkers:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            ssf_params = cfc_spec.ssf_params
            # Alias 😃...
            nts_block = num_time_steps_block
            supercell_size = model_params.supercell_size
            if ssf_params.assume_none:
                # A fake array will be created. It won't be used.
                nts_block, num_modes = 1, 1
            else:
                num_modes = ssf_params.num_modes

            # The shape of the structure factor array.
            i_ssf_shape = nts_block, num_modes, ssf_num_parts

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

        return _init_ssf_est_data


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
