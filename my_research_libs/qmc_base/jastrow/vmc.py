import typing as t
from abc import ABCMeta

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


class CFCSpec(vmc.CFCSpec, t.NamedTuple):
    """Represent the common spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    tpf_params: TPFParams


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
