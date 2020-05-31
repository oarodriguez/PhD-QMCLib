import typing as t
from abc import ABCMeta, abstractmethod

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from math import exp
from numpy import random

from phd_qmclib import qmc_base
from . import model
from .. import dmc

StateProp = dmc.StateProp
IterProp = dmc.IterProp

state_confs_dtype = np.float64
state_props_dtype = np.dtype([
    (StateProp.ENERGY.value, np.float64),
    (StateProp.WEIGHT.value, np.float64),
    (StateProp.MASK.value, np.bool)
])


# NOTE: Is this redundant...
class StateProps(dmc.StateProps, t.NamedTuple):
    """"""
    energy: np.ndarray
    weight: np.ndarray
    mask: np.ndarray


class DDFParams(dmc.DDFParams):
    """The parameters of the diffusion-and-drift process."""
    boson_number: int
    time_step: float
    sigma_spread: float


class DensityParams(dmc.DensityParams):
    """Static structure factor parameters."""
    num_bins: int
    as_pure_est: bool
    pfw_num_time_steps: int
    assume_none: bool


class SSFParams(dmc.SSFParams):
    """Static structure factor parameters."""
    num_modes: int
    as_pure_est: bool
    pfw_num_time_steps: int
    assume_none: bool


class CFCSpec(dmc.CFCSpec, t.NamedTuple):
    """Represent the common spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    ddf_params: DDFParams
    density_params: t.Optional[DensityParams]
    ssf_params: t.Optional[SSFParams]


class CFCSpecAlt(t.NamedTuple):
    """Represent the common spec of the core functions."""
    model_params: np.ndarray
    obf_params: np.ndarray
    tbf_params: np.ndarray
    ddf_params: np.ndarray
    density_params: t.Optional[np.ndarray]
    ssf_params: t.Optional[np.ndarray]


class Sampling(dmc.Sampling, metaclass=ABCMeta):
    """Spec for the VMC sampling of a Bijl-Jastrow model."""

    #: The spec of a concrete Jastrow model.
    model_spec: model.Spec

    @property
    def state_confs_shape(self):
        """"""
        max_num_walkers = self.max_num_walkers
        sys_conf_shape = self.model_spec.sys_conf_shape
        return (max_num_walkers,) + sys_conf_shape


# noinspection PyUnusedLocal
def _ddf_params_transform_stub(ddf_params: DDFParams) -> np.ndarray:
    pass


# noinspection PyUnusedLocal
def _density_params_transform_stub(density_params: DensityParams) -> \
        np.ndarray:
    pass


# noinspection PyUnusedLocal
def _ssf_params_transform_stub(ssf_params: SSFParams) -> np.ndarray:
    pass


# noinspection PyUnusedLocal
def _ddf_params_reconstruct_stub(ddf_params: np.ndarray) -> DDFParams:
    pass


# noinspection PyUnusedLocal
def _density_params_reconstruct_stub(density_params: np.ndarray) -> \
        DensityParams:
    pass


# noinspection PyUnusedLocal
def _ssf_params_reconstruct_stub(ssf_params: np.ndarray) -> SSFParams:
    pass


# noinspection PyUnusedLocal
def _density_core_stub(step_idx: int,
                       sys_idx: int,
                       clone_ref_idx: int,
                       state_confs: np.ndarray,
                       model_params: model.Params,
                       obf_params: model.OBFParams,
                       tbf_params: model.TBFParams,
                       density_params: DensityParams,
                       iter_density_array: np.ndarray,
                       aux_states_density_array: np.ndarray):
    """Stub for density_core function."""
    pass


@attr.s(auto_attribs=True, frozen=True)
class CoreFuncs(qmc_base.dmc.CoreFuncs):
    """The DMC core functions for the Bloch-Phonon model."""

    #: Parallel the execution where possible.
    jit_parallel: bool = True

    #: Use fastmath compiler directive.
    jit_fastmath: bool = True

    @property
    @abstractmethod
    def model_core_funcs(self) -> model.CoreFuncs:
        pass

    @property
    @abstractmethod
    def ddf_params_transform(self):
        """"""
        return _ddf_params_transform_stub

    @property
    @abstractmethod
    def density_params_transform(self):
        """"""
        return _density_params_transform_stub

    @property
    @abstractmethod
    def ssf_params_transform(self):
        """"""
        return _ssf_params_transform_stub

    @property
    @abstractmethod
    def ddf_params_reconstruct(self):
        """"""
        return _ddf_params_reconstruct_stub

    @property
    @abstractmethod
    def density_params_reconstruct(self):
        """"""
        return _density_params_reconstruct_stub

    @property
    @abstractmethod
    def ssf_params_reconstruct(self):
        """"""
        return _ssf_params_reconstruct_stub

    @property
    @abstractmethod
    def density_core(self):
        return _density_core_stub

    @cached_property
    def density_inner(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        parallel = self.jit_parallel
        model_core_funcs = self.model_core_funcs

        model_params_reconstruct = model_core_funcs.model_params_reconstruct
        obf_params_reconstruct = model_core_funcs.obf_params_reconstruct
        tbf_params_reconstruct = model_core_funcs.tbf_params_reconstruct
        density_params_reconstruct = self.density_params_reconstruct
        density_core = self.density_core

        @nb.jit(nopython=True, parallel=parallel, fastmath=fastmath)
        def _density_inner(step_idx: int,
                           state_confs: np.ndarray,
                           num_walkers: int,
                           max_num_walkers: int,
                           cloning_refs: np.ndarray,
                           model_params: np.ndarray,
                           obf_params: np.ndarray,
                           tbf_params: np.ndarray,
                           density_params: np.ndarray,
                           iter_density_array: np.ndarray,
                           aux_states_density_array: np.ndarray):
            """

            :param step_idx:
            :param state_confs:
            :param num_walkers:
            :param max_num_walkers:
            :param iter_density_array:
            :param aux_states_density_array:
            :return:
            """
            # Cloning table. Needed for evaluate pure estimators.
            prev_step_idx = step_idx % 2 - 1  #
            actual_step_idx = step_idx % 2

            prev_state_density = aux_states_density_array[prev_step_idx]  #
            actual_state_density = aux_states_density_array[actual_step_idx]
            actual_iter_density = iter_density_array[step_idx]

            # Hack to use this arrays inside the parallel for.
            density_params_nt = density_params_reconstruct(density_params)
            num_bins = density_params_nt.num_bins
            as_pure_est = density_params_nt.as_pure_est
            pfw_nts = density_params_nt.pfw_num_time_steps

            if as_pure_est:
                # Copy previous auxiliary state data to current auxiliary
                # state array. This is the "transport" of properties to
                # calculate the pure estimator.
                for bin_idx in nb.prange(num_bins):
                    actual_state_density[:, bin_idx] = \
                        prev_state_density[:, bin_idx]

            # Estimator evaluation (parallel for).
            for sys_idx in nb.prange(max_num_walkers):

                # Hack to use this arrays inside the parallel for.
                model_params_nt = model_params_reconstruct(model_params)
                obf_params_nt = obf_params_reconstruct(obf_params)
                tbf_params_nt = tbf_params_reconstruct(tbf_params)
                density_params_nt = density_params_reconstruct(density_params)

                # Beyond the actual number of walkers just pass to
                # the next iteration.
                if sys_idx >= num_walkers:
                    continue

                # Lookup which configuration should be cloned.
                clone_ref_idx = cloning_refs[sys_idx]

                # Evaluate structure factor.
                density_core(step_idx,
                             sys_idx,
                             clone_ref_idx,
                             state_confs,
                             model_params_nt,
                             obf_params_nt,
                             tbf_params_nt,
                             density_params_nt,
                             iter_density_array,
                             aux_states_density_array)

            if as_pure_est:
                if step_idx < pfw_nts:
                    est_divisor = step_idx + 1
                else:
                    est_divisor = pfw_nts
            else:
                est_divisor = 1

            # Accumulate the totals of the estimators.
            for bin_idx in nb.prange(num_bins):
                #
                actual_iter_density[bin_idx] = \
                    actual_state_density[:num_walkers, bin_idx].sum(axis=0)

                # Calculate structure factor pure estimator after the
                # forward sampling stage.
                if as_pure_est:
                    actual_iter_density[bin_idx] /= est_divisor

        return _density_inner

    @cached_property
    def density(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        model_core_funcs = self.model_core_funcs

        model_params_transform = model_core_funcs.model_params_transform
        obf_params_transform = model_core_funcs.obf_params_transform
        tbf_params_transform = model_core_funcs.tbf_params_transform
        density_params_transform = self.density_params_transform
        density_inner = self.density_inner

        @nb.jit(nopython=True, fastmath=fastmath)
        def _density(step_idx: int,
                     state: dmc.State,
                     cfc_spec: CFCSpec,
                     density_exec_data: dmc.DensityExecData):
            """

            :param step_idx:
            :param state:
            :param cfc_spec:
            :param density_exec_data:
            :return:
            """
            model_params = model_params_transform(cfc_spec.model_params)
            obf_params = obf_params_transform(cfc_spec.obf_params)
            tbf_params = tbf_params_transform(cfc_spec.tbf_params)
            density_params = density_params_transform(cfc_spec.density_params)

            # State data attributes.
            state_confs = state.confs
            num_walkers = state.num_walkers
            max_num_walkers = state.max_num_walkers
            branching_spec = state.branching_spec
            cloning_refs = branching_spec.cloning_ref

            # Density data attributes.
            iter_density_array = density_exec_data.iter_density_array
            aux_states_density_array = density_exec_data.pfw_aux_density_array

            density_inner(step_idx,
                          state_confs,
                          num_walkers,
                          max_num_walkers,
                          cloning_refs,
                          model_params,
                          obf_params,
                          tbf_params,
                          density_params,
                          iter_density_array,
                          aux_states_density_array)

        return _density

    @cached_property
    def fourier_density_core(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath

        # Slots to save data to  evaluate S(k).
        sqr_abs_slot = int(dmc.SSFPartSlot.FDK_SQR_ABS)
        real_slot = int(dmc.SSFPartSlot.FDK_REAL)
        imag_slot = int(dmc.SSFPartSlot.FDK_IMAG)

        fourier_density = self.model_core_funcs.fourier_density

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, fastmath=fastmath)
        def _fourier_density_core(step_idx: int,
                                  sys_idx: int,
                                  clone_ref_idx: int,
                                  momenta: np.ndarray,
                                  state_confs: np.ndarray,
                                  model_params: model.Params,
                                  obf_params: model.OBFParams,
                                  tbf_params: model.TBFParams,
                                  ssf_params: SSFParams,
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

            # System to be "moved-forward" and current system.
            prev_sys_ssf = prev_state_ssf[clone_ref_idx]
            actual_sys_ssf = actual_state_ssf[sys_idx]
            sys_conf = state_confs[sys_idx]

            # Static structure factor parameters.
            num_modes = ssf_params.num_modes
            pfw_nts = ssf_params.pfw_num_time_steps

            if not ssf_params.as_pure_est:
                # Mixed estimator.

                for kz_idx in range(num_modes):
                    momentum = momenta[kz_idx]
                    sys_fdk_idx = \
                        fourier_density(momentum, sys_conf, model_params,
                                        obf_params, tbf_params)

                    # Just update the actual state.
                    fdk_sqr_abs = sys_fdk_idx * sys_fdk_idx.conjugate()

                    actual_sys_ssf[kz_idx, sqr_abs_slot] = fdk_sqr_abs.real
                    actual_sys_ssf[kz_idx, real_slot] = sys_fdk_idx.real
                    actual_sys_ssf[kz_idx, imag_slot] = sys_fdk_idx.imag

                # Finish.
                return

            # Pure estimator.
            if step_idx >= pfw_nts:
                # Just "transport" the structure factor parts of the previous
                # configuration to the new one.
                for kz_idx in range(num_modes):
                    actual_state_ssf[sys_idx, kz_idx] = prev_sys_ssf[kz_idx]

            else:
                # Evaluate the structure factor for the actual
                # system configuration.
                for kz_idx in range(num_modes):
                    momentum = momenta[kz_idx]
                    sys_fdk_idx = \
                        fourier_density(momentum, sys_conf, model_params,
                                        obf_params, tbf_params)

                    fdk_sqr_abs = sys_fdk_idx * sys_fdk_idx.conjugate()

                    # Update with the previous state ("transport").
                    actual_sys_ssf[kz_idx, sqr_abs_slot] = \
                        fdk_sqr_abs.real + prev_sys_ssf[kz_idx, sqr_abs_slot]

                    actual_sys_ssf[kz_idx, real_slot] = \
                        sys_fdk_idx.real + prev_sys_ssf[kz_idx, real_slot]

                    actual_sys_ssf[kz_idx, imag_slot] = \
                        sys_fdk_idx.imag + prev_sys_ssf[kz_idx, imag_slot]

        return _fourier_density_core

    @cached_property
    def fourier_density_inner(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        parallel = self.jit_parallel
        model_core_funcs = self.model_core_funcs

        model_params_reconstruct = model_core_funcs.model_params_reconstruct
        obf_params_reconstruct = model_core_funcs.obf_params_reconstruct
        tbf_params_reconstruct = model_core_funcs.tbf_params_reconstruct
        ssf_params_reconstruct = self.ssf_params_reconstruct
        # Structure factor
        fourier_density_core = self.fourier_density_core

        @nb.jit(nopython=True, parallel=parallel, fastmath=fastmath)
        def _fourier_density_inner(step_idx: int,
                                   momenta: np.ndarray,
                                   state_confs: np.ndarray,
                                   num_walkers: int,
                                   max_num_walkers: int,
                                   cloning_refs: np.ndarray,
                                   model_params: np.ndarray,
                                   obf_params: np.ndarray,
                                   tbf_params: np.ndarray,
                                   ssf_params: np.ndarray,
                                   iter_ssf_array: np.ndarray,
                                   aux_states_sf_array: np.ndarray):
            """

            :param step_idx:
            :param state_confs:
            :param num_walkers:
            :param max_num_walkers:
            :param cloning_refs:
            :param iter_ssf_array:
            :param aux_states_sf_array:
            :return:
            """
            # Cloning table. Needed for evaluate pure estimators.
            actual_step_idx = step_idx % 2

            actual_state_ssf = aux_states_sf_array[actual_step_idx]
            actual_iter_ssf = iter_ssf_array[step_idx]

            # Branching process (parallel for).
            for sys_idx in nb.prange(max_num_walkers):

                # Beyond the actual number of walkers just pass to
                # the next iteration.
                if sys_idx >= num_walkers:
                    continue

                # Hack to use this arrays inside the parallel for.
                model_params_nt = model_params_reconstruct(model_params)
                obf_params_nt = obf_params_reconstruct(obf_params)
                tbf_params_nt = tbf_params_reconstruct(tbf_params)
                ssf_params_nt = ssf_params_reconstruct(ssf_params)

                # Lookup which configuration should be cloned.
                clone_ref_idx = cloning_refs[sys_idx]

                # Evaluate structure factor.
                fourier_density_core(step_idx,
                                     sys_idx,
                                     clone_ref_idx,
                                     momenta,
                                     state_confs,
                                     model_params_nt,
                                     obf_params_nt,
                                     tbf_params_nt,
                                     ssf_params_nt,
                                     iter_ssf_array,
                                     aux_states_sf_array)

            # Hack to use this arrays inside the parallel for.
            ssf_params_nt = ssf_params_reconstruct(ssf_params)

            num_modes = ssf_params_nt.num_modes
            as_pure_est = ssf_params_nt.as_pure_est
            pfw_nts = ssf_params_nt.pfw_num_time_steps

            if as_pure_est:
                if step_idx < pfw_nts:
                    est_divisor = step_idx + 1
                else:
                    est_divisor = pfw_nts
            else:
                est_divisor = 1

            # Accumulate the totals of the estimators.
            # NOTE: Fix up a memory leak using range instead numba.prange.
            # TODO: Compare speed of range vs numba.prange.
            # for sys_idx in range(num_walkers):
            #     # Accumulate S(k).
            #     actual_iter_ssf += actual_state_ssf[sys_idx]

            for kz_idx in nb.prange(num_modes):
                #
                actual_iter_ssf[kz_idx] = \
                    actual_state_ssf[:num_walkers, kz_idx].sum(axis=0)

                # Calculate structure factor pure estimator after the
                # forward sampling stage.
                if as_pure_est:
                    actual_iter_ssf[kz_idx] /= est_divisor

        return _fourier_density_inner

    @cached_property
    def fourier_density(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        model_core_funcs = self.model_core_funcs

        model_params_transform = model_core_funcs.model_params_transform
        obf_params_transform = model_core_funcs.obf_params_transform
        tbf_params_transform = model_core_funcs.tbf_params_transform
        ssf_params_transform = self.ssf_params_transform
        fourier_density_inner = self.fourier_density_inner

        @nb.jit(nopython=True, fastmath=fastmath)
        def _fourier_density(step_idx: int,
                             state: dmc.State,
                             cfc_spec: CFCSpec,
                             ssf_exec_data: dmc.SSFExecData):
            """

            :param step_idx:
            :param state:
            :param cfc_spec:
            :param ssf_exec_data:
            :return:
            """
            model_params = model_params_transform(cfc_spec.model_params)
            obf_params = obf_params_transform(cfc_spec.obf_params)
            tbf_params = tbf_params_transform(cfc_spec.tbf_params)
            ssf_params = ssf_params_transform(cfc_spec.ssf_params)

            state_confs = state.confs
            num_walkers = state.num_walkers
            max_num_walkers = state.max_num_walkers
            branching_spec = state.branching_spec
            cloning_refs = branching_spec.cloning_ref

            momenta = ssf_exec_data.momenta
            iter_ssf_array = ssf_exec_data.iter_ssf_array
            aux_states_ssf_array = ssf_exec_data.pfw_aux_ssf_array

            fourier_density_inner(step_idx,
                                  momenta,
                                  state_confs,
                                  num_walkers,
                                  max_num_walkers,
                                  cloning_refs,
                                  model_params,
                                  obf_params,
                                  tbf_params,
                                  ssf_params,
                                  iter_ssf_array,
                                  aux_states_ssf_array)

        return _fourier_density

    @cached_property
    def ith_diffusion(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        pos_slot = int(model.SysConfSlot.pos)
        drift_slot = int(model.SysConfSlot.drift)
        recast = self.recast

        @nb.jit(nopython=True, fastmath=fastmath)
        def _ith_diffuse(i_: int,
                         time_step: float,
                         sys_conf: np.ndarray,
                         ddf_params: DDFParams):
            """

            :param i_:
            :param sys_conf:
            :param time_step:
            :param ddf_params:
            :return:
            """
            # Alias 🙂
            normal = random.normal

            # Standard deviation as a function of time step.
            z_i = sys_conf[pos_slot, i_]
            drift_i = sys_conf[drift_slot, i_]

            # Diffuse current configuration.
            # sigma = sqrt(2 * time_step)
            sigma = ddf_params.sigma_spread
            rnd_spread = normal(0, sigma)
            z_i_next = z_i + 2 * drift_i * time_step + rnd_spread
            z_i_next_recast = recast(z_i_next, ddf_params)

            return z_i_next_recast

        return _ith_diffuse

    @cached_property
    def init_state_data(self):
        """Initialize the data arrays for the DMC states generator."""

        num_slots = len(model.SysConfSlot.__members__)

        @nb.njit
        def _init_state_data(base_shape: t.Tuple[int, ...],
                             cfc_spec: CFCSpec):
            """

            :param cfc_spec:
            :return:
            """
            nop = cfc_spec.model_params.boson_number
            confs_shape = base_shape + (num_slots, nop)
            props_shape = base_shape + ()

            state_confs = np.zeros(confs_shape, dtype=state_confs_dtype)
            energy = np.zeros(props_shape, dtype=np.float64)
            weight = np.zeros(props_shape, dtype=np.float64)
            mask = np.zeros(props_shape, dtype=np.bool_)
            state_props = StateProps(energy, weight, mask)
            return dmc.StateData(state_confs, state_props)

        return _init_state_data

    @cached_property
    def build_state(self):
        """"""

        @nb.njit
        def _build_state(state_data: dmc.StateData,
                         state_energy: float,
                         state_weight: float,
                         state_num_walkers: int,
                         state_ref_energy: float,
                         state_accum_energy: float,
                         max_num_walkers: int,
                         state_branching_spec: dmc.BranchingSpec = None):
            """

            :param state_data:
            :param state_energy:
            :param state_weight:
            :param state_num_walkers:
            :param state_ref_energy:
            :param state_accum_energy:
            :param max_num_walkers:
            :param state_branching_spec:
            :return:
            """

            state_confs = state_data.confs
            state_props = state_data.props
            return dmc.State(confs=state_confs,
                             props=state_props,
                             energy=state_energy,
                             weight=state_weight,
                             num_walkers=state_num_walkers,
                             ref_energy=state_ref_energy,
                             accum_energy=state_accum_energy,
                             max_num_walkers=max_num_walkers,
                             branching_spec=state_branching_spec)

        return _build_state

    @cached_property
    def evolve_system(self):
        """

        :return:
        """
        fastmath = self.jit_fastmath
        pos_slot = int(model.SysConfSlot.pos)
        drift_slot = int(model.SysConfSlot.drift)

        # JIT functions.
        ith_diffusion = self.ith_diffusion
        ith_energy_and_drift = self.model_core_funcs.ith_energy_and_drift

        # noinspection PyUnusedLocal
        @nb.jit(nopython=True, fastmath=fastmath)
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
                           next_state_weights: np.ndarray,
                           model_params: model.Params,
                           obf_params: model.OBFParams,
                           tbf_params: model.TBFParams,
                           ddf_params: DDFParams):
            """Executes the diffusion process.

            :param sys_idx: The index of the system.
            :param cloning_ref_idx:
            :param prev_state_confs:
            :param prev_state_energies:
            :param prev_state_weights:
            :param actual_state_confs:
            :param actual_state_energies:
            :param actual_state_weights:
            :param time_step:
            :param ref_energy:
            :param next_state_confs:
            :param next_state_energies:
            :param next_state_weights:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :param ddf_params:
            :return:
            """
            # Standard deviation as a function of time step.
            prev_conf = prev_state_confs[cloning_ref_idx]
            sys_conf = actual_state_confs[sys_idx]
            next_conf = next_state_confs[sys_idx]

            nop = model_params.boson_number
            for i_ in range(nop):
                # Diffuse current configuration. We can update the position
                # of the next configuration.
                z_i_next = \
                    ith_diffusion(i_, time_step, prev_conf, ddf_params)
                sys_conf[pos_slot, i_] = z_i_next
                next_conf[pos_slot, i_] = z_i_next

            energy = actual_state_energies[sys_idx]
            energy_next = 0.
            for i_ in range(nop):
                ith_energy_drift = \
                    ith_energy_and_drift(i_, sys_conf, model_params,
                                         obf_params, tbf_params)
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
    def evolve_state_inner(self):
        """

        :return:
        """
        parallel = self.jit_parallel
        fastmath = self.jit_fastmath
        model_core_funcs = self.model_core_funcs

        # JIT methods.
        model_params_reconstruct = model_core_funcs.model_params_reconstruct
        obf_params_reconstruct = model_core_funcs.obf_params_reconstruct
        tbf_params_reconstruct = model_core_funcs.tbf_params_reconstruct
        ddf_params_reconstruct = self.ddf_params_reconstruct
        evolve_system = self.evolve_system

        @nb.jit(nopython=True, parallel=parallel, fastmath=fastmath)
        def _evolve_state_inner(prev_state_confs: np.ndarray,
                                prev_state_energies: np.ndarray,
                                prev_state_weights: np.ndarray,
                                prev_state_masks: np.ndarray,
                                actual_state_confs: np.ndarray,
                                actual_state_energies: np.ndarray,
                                actual_state_weights: np.ndarray,
                                actual_state_masks: np.ndarray,
                                aux_next_state_confs: np.ndarray,
                                aux_next_state_energies: np.ndarray,
                                aux_next_state_weights: np.ndarray,
                                aux_next_state_masks: np.ndarray,
                                actual_num_walkers: int,
                                max_num_walkers: int,
                                time_step: float,
                                ref_energy: float,
                                cloning_refs: np.ndarray,
                                model_params: np.ndarray,
                                obf_params: np.ndarray,
                                tbf_params: np.ndarray,
                                ddf_params: np.ndarray):
            """Realize the diffusion-branching process.

            This function realize a simple diffusion process over each
            one of the walkers, followed by the branching process.

            :param prev_state_confs:
            :param actual_state_confs:
            :param aux_next_state_confs:
            :param actual_num_walkers:
            :param max_num_walkers:
            :param time_step:
            :param ref_energy:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            # Total energy and weight of the next configuration.
            # NOTE: This initialization causes a memory leak with
            #  parallel=True
            # state_energy = 0.
            # state_weight = 0.

            # Branching and diffusion process (parallel for).
            for sys_idx in nb.prange(max_num_walkers):

                # Hack to use this arrays inside the parallel for.
                model_params_nt = model_params_reconstruct(model_params)
                obf_params_nt = obf_params_reconstruct(obf_params)
                tbf_params_nt = tbf_params_reconstruct(tbf_params)
                ddf_params_nt = ddf_params_reconstruct(ddf_params)

                # Beyond the actual number of walkers just pass to
                # the next iteration.
                if sys_idx >= actual_num_walkers:

                    # Mask the configuration.
                    actual_state_masks[sys_idx] = True

                else:

                    # Lookup which configuration should be cloned.
                    cloning_ref_idx = cloning_refs[sys_idx]
                    sys_energy = prev_state_energies[cloning_ref_idx]
                    # sys_weight = prev_state_weights[ref_idx]

                    # Evolve the system for the next iteration.
                    # TODO: Can we return tuples inside a nb.prange?
                    evolve_system(sys_idx, cloning_ref_idx,
                                  prev_state_confs,
                                  prev_state_energies,
                                  prev_state_weights,
                                  actual_state_confs,
                                  actual_state_energies,
                                  actual_state_weights,
                                  time_step,
                                  ref_energy,
                                  aux_next_state_confs,
                                  aux_next_state_energies,
                                  aux_next_state_weights,
                                  model_params_nt,
                                  obf_params_nt,
                                  tbf_params_nt,
                                  ddf_params_nt)

                    # Cloning process. Actual states are not modified.
                    actual_state_confs[sys_idx] \
                        = prev_state_confs[cloning_ref_idx]
                    actual_state_energies[sys_idx] = sys_energy

                    # Basic algorithm of branching gives a unit weight to each
                    # new walker. We set the value here. In addition, we unmask
                    # the walker, i.e., we mark it as valid.
                    actual_state_weights[sys_idx] = 1.
                    actual_state_masks[sys_idx] = False

                    # The contribution to the total energy and weight.
                    # NOTE: See memory leak note above.
                    # state_energy += sys_energy
                    # state_weight += 1.

                # NOTE: It is faster not returning anything.

        return _evolve_state_inner

    @cached_property
    def evolve_state(self):
        """

        :return:
        """
        model_core_funcs = self.model_core_funcs
        evolve_state_inner = self.evolve_state_inner
        model_params_transform = model_core_funcs.model_params_transform
        obf_params_transform = model_core_funcs.obf_params_transform
        tbf_params_transform = model_core_funcs.tbf_params_transform
        ddf_params_transform = self.ddf_params_transform

        @nb.jit(nopython=True)
        def _evolve_state(prev_state_data: dmc.StateData,
                          actual_state_data: dmc.StateData,
                          next_state_data: dmc.StateData,
                          actual_num_walkers: int,
                          max_num_walkers: int,
                          time_step: float,
                          ref_energy: float,
                          branching_spec: dmc.BranchingSpec,
                          cfc_spec: CFCSpec):
            """Realizes the diffusion-branching process.

            This function realize a simple diffusion process over each
            one of the walkers, followed by the branching process.

            :param prev_state_data:
            :param actual_state_data:
            :param next_state_data:
            :param actual_num_walkers:
            :param max_num_walkers:
            :param time_step:
            :param ref_energy:
            :param branching_spec:
            :param cfc_spec:
            :return:
            """
            model_params = model_params_transform(cfc_spec.model_params)
            obf_params = obf_params_transform(cfc_spec.obf_params)
            tbf_params = tbf_params_transform(cfc_spec.tbf_params)
            ddf_params = ddf_params_transform(cfc_spec.ddf_params)

            actual_state_confs = actual_state_data.confs
            actual_state_props = actual_state_data.props
            prev_state_confs = prev_state_data.confs
            prev_state_props = prev_state_data.props
            aux_next_state_confs = next_state_data.confs
            aux_next_state_props = next_state_data.props
            cloning_refs = branching_spec.cloning_ref

            evolve_state_inner(prev_state_confs,
                               prev_state_props.energy,
                               prev_state_props.weight,
                               prev_state_props.mask,
                               actual_state_confs,
                               actual_state_props.energy,
                               actual_state_props.weight,
                               actual_state_props.mask,
                               aux_next_state_confs,
                               aux_next_state_props.energy,
                               aux_next_state_props.weight,
                               aux_next_state_props.mask,
                               actual_num_walkers,
                               max_num_walkers,
                               time_step,
                               ref_energy,
                               cloning_refs,
                               model_params,
                               obf_params,
                               tbf_params,
                               ddf_params)

        return _evolve_state

    @cached_property
    def prepare_ini_ith_system(self):
        """Prepare a system of the initial state of the sampling.

        :return:
        """
        fastmath = self.jit_fastmath
        pos_slot = int(model.SysConfSlot.pos)
        drift_slot = int(model.SysConfSlot.drift)

        # JIT functions.
        ith_energy_and_drift = self.model_core_funcs.ith_energy_and_drift

        @nb.jit(nopython=True, fastmath=fastmath)
        def _prepare_ini_ith_system(sys_idx: int,
                                    state_confs: np.ndarray,
                                    state_energies: np.ndarray,
                                    state_weights: np.ndarray,
                                    ini_sys_conf_set: np.ndarray,
                                    model_params: model.Params,
                                    obf_params: model.OBFParams,
                                    tbf_params: model.TBFParams):
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

            nop = model_params.boson_number
            for i_ in range(nop):
                # Particle-by-particle loop.
                energy_drift = \
                    ith_energy_and_drift(i_, ini_sys_conf, model_params,
                                         obf_params, tbf_params)
                ith_energy, ith_drift = energy_drift

                sys_conf[pos_slot, i_] = ini_sys_conf[pos_slot, i_]
                sys_conf[drift_slot, i_] = ith_drift
                energy_sum += ith_energy

            # Store the energy and initialize all weights to unity
            state_energies[sys_idx] = energy_sum
            state_weights[sys_idx] = 1.

        return _prepare_ini_ith_system

    @cached_property
    def prepare_state_data_inner(self):
        """Prepare the initial state of the sampling. """

        parallel = self.jit_parallel
        fastmath = self.jit_fastmath
        model_core_funcs = self.model_core_funcs

        # JIT functions.
        model_params_reconstruct = model_core_funcs.model_params_reconstruct
        obf_params_reconstruct = model_core_funcs.obf_params_reconstruct
        tbf_params_reconstruct = model_core_funcs.tbf_params_reconstruct
        prepare_ini_ith_system = self.prepare_ini_ith_system

        @nb.jit(nopython=True, parallel=parallel, fastmath=fastmath)
        def _prepare_state_data_inner(ini_sys_conf_set: np.ndarray,
                                      state_confs: np.ndarray,
                                      state_energy: np.ndarray,
                                      state_weight: np.ndarray,
                                      state_mask: np.ndarray,
                                      model_params: np.ndarray,
                                      obf_params: np.ndarray,
                                      tbf_params: np.ndarray):
            """Prepare the initial state of the sampling.

            :param ini_sys_conf_set:
            :param state_confs:
            :param state_energy:
            :param state_weight:
            :param state_mask:
            :param model_params:
            :param obf_params:
            :param tbf_params:
            :return:
            """
            ini_num_walkers = len(ini_sys_conf_set)

            # Initialize the mask.
            state_mask[:] = True

            for sys_idx in nb.prange(ini_num_walkers):

                # Hack to use this arrays inside the parallel for.
                model_params_nt = model_params_reconstruct(model_params)
                obf_params_nt = obf_params_reconstruct(obf_params)
                tbf_params_nt = tbf_params_reconstruct(tbf_params)

                # Prepare each one of the configurations of the state.
                prepare_ini_ith_system(sys_idx, state_confs, state_energy,
                                       state_weight, ini_sys_conf_set,
                                       model_params_nt, obf_params_nt,
                                       tbf_params_nt)

                # Unmask this walker.
                state_mask[sys_idx] = False

        return _prepare_state_data_inner

    @cached_property
    def prepare_state_data(self):
        """Prepare the initial state of the sampling. """

        fastmath = self.jit_fastmath
        model_core_funcs = self.model_core_funcs
        model_params_transform = model_core_funcs.model_params_transform
        obf_params_transform = model_core_funcs.obf_params_transform
        tbf_params_transform = model_core_funcs.tbf_params_transform
        prepare_state_data_inner = self.prepare_state_data_inner

        @nb.jit(nopython=True, fastmath=fastmath)
        def _prepare_state_data(ini_sys_conf_set: np.ndarray,
                                state_data: dmc.StateData,
                                cfc_spec: CFCSpec):
            """Prepare the initial state of the sampling.

            :param ini_sys_conf_set:
            :return:
            """
            state_confs = state_data.confs
            state_props = state_data.props
            model_params = model_params_transform(cfc_spec.model_params)
            obf_params = obf_params_transform(cfc_spec.obf_params)
            tbf_params = tbf_params_transform(cfc_spec.tbf_params)
            state_energy = state_props.energy
            state_weight = state_props.weight
            state_mask = state_props.mask

            prepare_state_data_inner(ini_sys_conf_set, state_confs,
                                     state_energy, state_weight,
                                     state_mask, model_params,
                                     obf_params, tbf_params)

        return _prepare_state_data
