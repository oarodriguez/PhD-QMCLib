import typing as t
from abc import ABCMeta, abstractmethod

import numpy as np
from cached_property import cached_property
from numba import jit

from . import model
from .. import vmc

__all__ = [
    'CoreFuncs',
    'Sampling',
    'TPFParams',
]


class TPFParams(vmc.TPFParams, metaclass=ABCMeta):
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
        def _wf_abs_log(actual_conf: np.ndarray,
                        cfc_spec: CFCSpec):
            """

            :param actual_conf:
            :param cfc_spec:
            :return:
            """
            return wf_abs_log(actual_conf, cfc_spec.model_params,
                              cfc_spec.obf_params, cfc_spec.tbf_params)

        return _wf_abs_log

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
        def _sys_conf_ppf(ini_sys_conf: np.ndarray,
                          prop_sys_conf: np.ndarray,
                          cfc_spec: CFCSpec):
            """Move the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param cfc_spec:
            :return:
            """
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
        def _energy(sys_conf: np.ndarray,
                    cfc_spec: CFCSpec):
            """

            :param sys_conf:
            :param cfc_spec:
            :return:
            """
            return energy(sys_conf, cfc_spec.model_params,
                          cfc_spec.obf_params, cfc_spec.tbf_params)

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
                             momenta: np.ndarray,
                             sys_conf: np.ndarray,
                             cfc_spec: CFCSpec,
                             iter_ssf_array: np.ndarray):
            """

            :param step_idx:
            :param sys_conf:
            :param cfc_spec:
            :param iter_ssf_array:
            :return:
            """
            actual_iter_ssf = iter_ssf_array[step_idx]

            num_modes = momenta.shape[0]
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
