from abc import ABCMeta
from math import sqrt
from typing import Mapping, Tuple

import numpy as np
from numba import jit

from phdthesis_lib.utils import cached_property
from . import core
from .. import vmc
from ..utils import recast_to_supercell

__all__ = [
    'PBCSampling',
    'PBCUniformSampling',
    'Sampling',
    'UniformSampling'
]


class Sampling(vmc.Sampling, metaclass=ABCMeta):
    """The class that implements the functions to sample the probability
    density of a Bijl-Jastrow model using the Metropolis-Hastings algorithm.
    The implementation generates random movements from a normal distribution
    function.
    """
    # The slots available in a single particle configuration.
    sys_conf_slots = core.SysConfSlot

    def __init__(self, model: core.Model,
                 params: Mapping[str, float]):
        """

        :param model:
        :param params:
        """
        super().__init__(params)
        self._model = model

    @property
    def model(self):
        """"""
        return self._model

    @cached_property
    def wf_abs_log(self):
        """"""
        return self.model.core_funcs.wf_abs_log

    @property
    def ppf_args(self):
        """Set of parameters for the proposal probability function."""
        time_step = self.params[self.params_cls.names.TIME_STEP]
        # The average amplitude of the displacements.
        move_spread = sqrt(time_step)
        return move_spread,

    @cached_property
    def ith_sys_conf_ppf(self):
        """

        :return:
        """
        pos_slot = int(self.sys_conf_slots.POS_SLOT)
        rand_displace = self.rand_displace

        @jit(nopython=True, cache=True)
        def _ith_sys_conf_ppf(i_: int,
                              ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              ppf_params: Tuple[float, ...]):
            """Move the i-nth particle of the current configuration of the
            system. The moves are displacements of the original position plus
            a term sampled from a uniform distribution.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param ppf_params:
            :return:
            """
            # Unpack data.
            move_spread = ppf_params,
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(move_spread)
            prop_sys_conf[pos_slot, i_] = z_i + rnd_spread

        return _ith_sys_conf_ppf

    @cached_property
    def sys_conf_ppf(self):
        """

        :return:
        """
        boson_index_dim = int(core.SYS_CONF_PARTICLE_INDEX_DIM)
        ith_sys_conf_ppf = self.ith_sys_conf_ppf

        @jit(nopython=True, cache=True)
        def _sys_conf_ppf(ini_sys_conf: np.ndarray,
                          prop_sys_conf: np.ndarray,
                          ppf_params: Tuple[float, ...]):
            """Move the current configuration of the system.

            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param ppf_params:
            :return:
            """
            scs = ini_sys_conf.shape[boson_index_dim]  # Number of particles
            for i_ in range(scs):
                ith_sys_conf_ppf(i_, ini_sys_conf, prop_sys_conf, ppf_params)

        return _sys_conf_ppf


class PBCSampling(Sampling, vmc.PBCSampling, metaclass=ABCMeta):
    """Sampling (with periodic boundary conditions) of the probability
    density of a Bijl-Jastrow model using the Metropolis-Hastings algorithm.
    """

    #
    @property
    def ppf_args(self):
        """Set of parameters for the proposal probability function."""
        time_step = self.params[self.params_cls.names.TIME_STEP]
        z_min, z_max = self.model.boundaries
        move_spread = sqrt(time_step)
        return move_spread, z_min, z_max

    # For a pdf with periodic boundary conditions.
    @cached_property
    def recast(self):
        """"""
        # TODO: Move this function to this module.
        return recast_to_supercell

    @cached_property
    def ith_sys_conf_ppf(self):
        """

        :return:
        """
        pos_slot = int(self.sys_conf_slots.POS_SLOT)
        rand_displace = self.rand_displace
        recast = self.recast  # TODO: Use a better name, maybe?

        @jit(nopython=True, cache=True)
        def _ith_sys_conf_ppf(i_: int,
                              ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              ppf_params: Tuple[float, ...]):
            """Move the i-nth particle of the current configuration of the
            system under PBC.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param ppf_params:.
            :return:
            """
            # Unpack data
            move_spread, z_min, z_max = ppf_params
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(move_spread)
            z_i_upd = recast(z_i + rnd_spread, z_min, z_max)
            prop_sys_conf[pos_slot, i_] = z_i_upd

        return _ith_sys_conf_ppf


class UniformSampling(Sampling, vmc.UniformSampling,
                      metaclass=ABCMeta):
    """Sampling of the probability density of a Bijl-Jastrow model using
    the Metropolis-Hastings algorithm. It uses a uniform distribution
    function to generate random numbers.
    """

    @property
    def ppf_args(self):
        """Set of parameters for the proposal probability function."""
        move_spread = self.params[self.params_cls.names.MOVE_SPREAD]
        return move_spread,


class PBCUniformSampling(PBCSampling, UniformSampling,
                         vmc.PBCUniformSampling,
                         metaclass=ABCMeta):
    """Sampling (with periodic boundary conditions) of the probability
    density of a Bijl-Jastrow model using the Metropolis-Hastings algorithm.
    It uses a uniform distribution function to generate random numbers.
    """

    @property
    def ppf_args(self):
        """Set of parameters for the proposal probability function."""
        move_spread = self.params[self.params_cls.names.MOVE_SPREAD]
        z_min, z_max = self.model.boundaries
        return move_spread, z_min, z_max
