from abc import ABCMeta
from typing import NamedTuple, Union

import numpy as np
from cached_property import cached_property
from numba import jit

from . import model
from .. import vmc

__all__ = [
    'TPFSpecNT',
    'SamplingFuncs',
    'UTPFSpecNT'
]


class TPFSpecNT(vmc.TPFSpecNT, NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a (normal) gaussian distribution function.
    """
    boson_number: int
    time_step: float


class UTPFSpecNT(vmc.UTPFSpecNT, NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float


class SamplingFuncs(vmc.SamplingFuncs, metaclass=ABCMeta):
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

        @jit(nopython=True, cache=True)
        def _ith_sys_conf_ppf(i_: int,
                              ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              tpf_spec: Union[TPFSpecNT, UTPFSpecNT]):
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

        @jit(nopython=True, cache=True)
        def _sys_conf_ppf(ini_sys_conf: np.ndarray,
                          prop_sys_conf: np.ndarray,
                          tpf_spec: Union[TPFSpecNT, UTPFSpecNT]):
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
