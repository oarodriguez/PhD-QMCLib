import typing as t
from abc import ABCMeta, abstractmethod

import numpy as np
from cached_property import cached_property
from numba import jit, njit

from . import model
from .. import vmc

__all__ = [
    'CoreFuncs',
    'Sampling',
    'SSFParams',
    'TPFParams'
]

STAT_ACCEPTED = vmc.STAT_ACCEPTED
STAT_REJECTED = vmc.STAT_REJECTED


class TPFParams(vmc.TPFParams):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float


class SSFParams(vmc.SSFParams):
    """Static structure factor parameters."""
    num_modes: int


class CFCSpec(vmc.CFCSpec, t.NamedTuple):
    """Represent the common spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    tpf_params: TPFParams
    ssf_params: t.Optional[SSFParams]


class Sampling(vmc.Sampling, metaclass=ABCMeta):
    """Spec for the VMC sampling of a Bijl-Jastrow model."""

    #: The spec of a concrete Jastrow model.
    model_spec: model.Spec

    move_spread: float
    rng_seed: t.Optional[int]


class CoreFuncs(vmc.CoreFuncs, metaclass=ABCMeta):
    """The core functions to realize a VMC sampling.

    This class implements the functions to sample the probability
    density of a Bijl-Jastrow model using the Metropolis-Hastings algorithm.
    The implementation generates random movements from a normal distribution
    function.
    """
    # The slots available in a single particle configuration.
    sys_conf_slots: t.ClassVar = model.SysConfSlot

    @property
    @abstractmethod
    def model_core_funcs(self) -> model.CoreFuncs:
        pass

    @property
    def wf_abs_log(self):
        """

        :return:
        """
        wf_abs_log = self.model_core_funcs.wf_abs_log

        @jit(nopython=True)
        def _wf_abs_log(state_data: vmc.StateData,
                        cfc_spec: CFCSpec):
            """

            :param state_data:
            :param cfc_spec:
            :return:
            """
            actual_conf = state_data.sys_conf
            return wf_abs_log(actual_conf, cfc_spec.model_params,
                              cfc_spec.obf_params, cfc_spec.tbf_params)

        return _wf_abs_log

    @cached_property
    def init_state_data(self):
        """

        :return:
        """
        num_slots = len(model.SysConfSlot.__members__)

        @njit
        def _init_state_data(base_shape: t.Tuple[int, ...],
                             cfc_spec: CFCSpec):
            """

            :param cfc_spec:
            :return:
            """
            nop = cfc_spec.model_params.boson_number
            confs_shape = base_shape + (num_slots, nop)
            state_sys_conf = np.zeros(confs_shape, dtype=np.float64)
            return vmc.StateData(state_sys_conf)

        return _init_state_data

    @cached_property
    def build_state(self):
        """"""

        @njit
        def _build_state(state_data: vmc.StateData,
                         wf_abs_log: float,
                         move_stat: int):
            """

            :param state_data:
            :param wf_abs_log:
            :param move_stat:
            :return:
            """
            return vmc.State(state_data.sys_conf,
                             wf_abs_log, move_stat)

        return _build_state

    @cached_property
    def init_prepare_state(self):
        """

        :return:
        """
        wf_abs_log_func = self.model_core_funcs.wf_abs_log
        init_state_data = self.init_state_data
        build_state = self.build_state

        @njit
        def _init_prepare_state(sys_conf: np.ndarray,
                                cfc_spec: CFCSpec):
            """

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            model_params = cfc_spec.model_params
            obf_params = cfc_spec.obf_params
            tbf_params = cfc_spec.tbf_params
            state_data = init_state_data(cfc_spec)

            wf_abs_log = wf_abs_log_func(sys_conf, model_params,
                                         obf_params, tbf_params)
            state_data.sys_conf[:] = sys_conf[:]
            return build_state(state_data, wf_abs_log, STAT_ACCEPTED)

        return _init_prepare_state

    @cached_property
    def ith_sys_conf_tpf(self):
        """

        :return:
        """
        pos_slot = int(self.sys_conf_slots.pos)
        rand_displace = self.rand_displace

        @jit(nopython=True)
        def _ith_sys_conf_ppf(i_: int,
                              ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              tpf_params: TPFParams):
            """Move the i-nth particle of the current configuration of the
            system. The moves are displacements of the original position plus
            a term sampled from a uniform distribution.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_params:
            :return:
            """
            # Unpack data.
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(tpf_params)
            prop_sys_conf[pos_slot, i_] = z_i + rnd_spread

        return _ith_sys_conf_ppf

    @cached_property
    def sys_conf_tpf(self):
        """

        :return:
        """
        ith_sys_conf_tpf = self.ith_sys_conf_tpf

        @jit(nopython=True)
        def _sys_conf_ppf(actual_state_data: vmc.StateData,
                          next_state_data: vmc.StateData,
                          cfc_spec: CFCSpec):
            """Move the current configuration of the system.

            :param actual_state_data:
            :param next_state_data:
            :param cfc_spec:
            :return:
            """
            ini_sys_conf = actual_state_data.sys_conf
            prop_sys_conf = next_state_data.sys_conf
            tpf_params = cfc_spec.tpf_params
            nop = tpf_params.boson_number  # Number of particles
            for i_ in range(nop):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_params)

        return _sys_conf_ppf

    @cached_property
    def energy(self):
        """

        :return:
        """
        energy = self.model_core_funcs.energy

        @jit(nopython=True)
        def _energy(step_idx: int,
                    state: vmc.State,
                    cfc_spec: CFCSpec,
                    iter_props: vmc.PropsData):
            """

            :param step_idx:
            :param state:
            :param cfc_spec:
            :param iter_props:
            :return:
            """
            sys_conf = state.sys_conf
            move_stat = state.move_stat
            energy_set = iter_props.energy

            if move_stat == STAT_REJECTED:
                # Just get the previous value of the energy.
                # TODO: Verify this works correctly when step_idx is zero.
                state_energy = energy_set[step_idx - 1]

            else:
                state_energy = energy(sys_conf, cfc_spec.model_params,
                                      cfc_spec.obf_params, cfc_spec.tbf_params)

            energy_set[step_idx] = state_energy

        return _energy

    @cached_property
    def one_body_density(self):
        """

        :return:
        """
        one_body_density = self.model_core_funcs.one_body_density

        @jit(nopython=True)
        def _one_body_density(step_idx: int,
                              pos_offset: np.ndarray,
                              sys_conf: np.ndarray,
                              cfc_spec: CFCSpec,
                              iter_obd_array: np.ndarray):
            """

            :param step_idx:
            :param pos_offset:
            :param sys_conf:
            :param cfc_spec:
            :param iter_obd_array:
            :return:
            """
            actual_iter_obd = iter_obd_array[step_idx]

            num_pos = pos_offset.shape[0]
            # NOTE: This may be replaced by a numba prange.
            for pos_idx in range(num_pos):
                pos = pos_offset[pos_idx]
                obd_idx = one_body_density(pos, sys_conf,
                                           cfc_spec.model_params,
                                           cfc_spec.obf_params,
                                           cfc_spec.tbf_params)
                actual_iter_obd[pos_idx] = obd_idx

        return _one_body_density

    @cached_property
    def fourier_density(self):
        """

        :return:
        """
        # Slots to save data to  evaluate S(k).
        sqr_abs_slot = int(vmc.SSFPartSlot.FDK_SQR_ABS)
        real_slot = int(vmc.SSFPartSlot.FDK_REAL)
        imag_slot = int(vmc.SSFPartSlot.FDK_IMAG)

        fourier_density = self.model_core_funcs.fourier_density

        @jit(nopython=True)
        def _fourier_density(step_idx: int,
                             state: vmc.State,
                             cfc_spec: CFCSpec,
                             ssf_exec_data: vmc.SSFExecData):
            """

            :param step_idx:
            :param cfc_spec:
            :return:
            """
            sys_conf = state.sys_conf
            move_stat = state.move_stat
            momenta = ssf_exec_data.momenta
            iter_ssf_array = ssf_exec_data.iter_ssf_array

            if move_stat == STAT_REJECTED:
                # Just copy the previous value of S(k).
                # TODO: Verify this works correctly when step_idx is zero.
                iter_ssf_array[step_idx] = iter_ssf_array[step_idx - 1]

            else:
                num_modes = momenta.shape[0]
                actual_iter_ssf = iter_ssf_array[step_idx]
                for kz_idx in range(num_modes):
                    momentum = momenta[kz_idx]
                    sys_fdk_idx = fourier_density(momentum, sys_conf,
                                                  cfc_spec.model_params,
                                                  cfc_spec.obf_params,
                                                  cfc_spec.tbf_params)
                    fdk_sqr_abs = sys_fdk_idx * sys_fdk_idx.conjugate()
                    actual_iter_ssf[kz_idx, sqr_abs_slot] = fdk_sqr_abs.real
                    actual_iter_ssf[kz_idx, real_slot] = sys_fdk_idx.real
                    actual_iter_ssf[kz_idx, imag_slot] = sys_fdk_idx.imag

        return _fourier_density
