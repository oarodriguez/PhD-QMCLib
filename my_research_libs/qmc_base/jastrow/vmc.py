from abc import ABCMeta
from math import sqrt
from typing import NamedTuple, Optional, Union

import numpy as np
from cached_property import cached_property
from numba import jit

from . import model
from .. import vmc

__all__ = [
    'CoreFuncs',
    'Sampling',
    'NTPFSpecNT',
    'NormalSampling',
    'UTPFSpecNT',
]


class NTPFSpecNT(vmc.NTPFSpecNT, NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a (normal) gaussian distribution function.
    """
    boson_number: int
    sigma: float


class UTPFSpecNT(vmc.UTPFSpecNT, NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float


class Sampling(vmc.Sampling, metaclass=ABCMeta):
    """Spec for the VMC sampling of a Bijl-Jastrow model."""

    #: The spec of a concrete Jastrow model.
    model_spec: model.Spec

    move_spread: float
    ini_sys_conf: np.ndarray
    rng_seed: Optional[int]

    @property
    def wf_spec_nt(self):
        """The trial wave function spec."""
        return self.model_spec.cfc_spec_nt

    @property
    def tpf_spec_nt(self):
        """"""
        boson_number = self.model_spec.boson_number
        return UTPFSpecNT(boson_number, self.move_spread)


class NormalSampling(vmc.NormalSampling, metaclass=ABCMeta):
    """Spec for the VMC sampling of a Bijl-Jastrow model."""

    #: The spec of a concrete Jastrow model.
    model_spec: model.Spec

    time_step: float
    ini_sys_conf: np.ndarray
    rng_seed: int

    @property
    def wf_spec_nt(self):
        """The trial wave function spec."""
        return self.model_spec.cfc_spec_nt

    @property
    def tpf_spec_nt(self):
        """"""
        sigma = sqrt(self.time_step)
        number = self.model_spec.boson_number
        return NTPFSpecNT(number, sigma)


class CoreFuncs(vmc.CoreFuncs, metaclass=ABCMeta):
    """The core functions to realize a VMC sampling.

    This class implements the functions to sample the probability
    density of a Bijl-Jastrow model using the Metropolis-Hastings algorithm.
    The implementation generates random movements from a normal distribution
    function.
    """
    # The slots available in a single particle configuration.
    sys_conf_slots = model.SysConfSlot

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
                              tpf_spec: Union[NTPFSpecNT, UTPFSpecNT]):
            """Move the i-nth particle of the current configuration of the
            system. The moves are displacements of the original position plus
            a term sampled from a uniform distribution.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_spec:
            :return:
            """
            # Unpack data.
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(tpf_spec)
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
                          tpf_spec: Union[NTPFSpecNT, UTPFSpecNT]):
            """Move the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_spec:
            :return:
            """
            nop = tpf_spec.boson_number  # Number of particles
            for i_ in range(nop):
                ith_sys_conf_tpf(i_, ini_sys_conf, prop_sys_conf, tpf_spec)

        return _sys_conf_ppf
