from math import sqrt
from typing import NamedTuple, Union

import attr
import numpy as np
from cached_property import cached_property
from numba import jit

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.utils import recast_to_supercell
from . import model

__all__ = [
    'CoreFuncs',
    'Sampling',
    'TPFSpecNT',
    'NormalCoreFuncs',
    'NormalSampling',
    'UTPFSpecNT',
    'core_funcs'
]


class TPFSpecNT(qmc_base.jastrow.vmc.NTPFSpecNT, NamedTuple):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a normal (gaussian) distribution function.
    """
    boson_number: int
    sigma: float
    lower_bound: float
    upper_bound: float


class UTPFSpecNT(qmc_base.jastrow.vmc.UTPFSpecNT, NamedTuple):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float
    lower_bound: float
    upper_bound: float


@attr.s(auto_attribs=True, frozen=True)
class Sampling(qmc_base.jastrow.vmc.Sampling):
    """The spec of the VMC sampling."""

    model_spec: model.Spec
    move_spread: float
    num_steps: int
    ini_sys_conf: np.ndarray
    rng_seed: int = None
    core_funcs: 'CoreFuncs' = attr.ib(init=False, cmp=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        # NOTE: Should we use a new CoreFuncs instance?
        super().__setattr__('core_funcs', core_funcs)

        if self.rng_seed is None:
            rng_seed = utils.get_random_rng_seed()
            super().__setattr__('rng_seed', rng_seed)

    @property
    def tpf_spec_nt(self):
        """"""
        move_spread = self.move_spread
        boson_number = self.model_spec.boson_number
        z_min, z_max = self.model_spec.boundaries
        return UTPFSpecNT(boson_number, move_spread=move_spread,
                          lower_bound=z_min, upper_bound=z_max)


class CoreFuncs(qmc_base.jastrow.vmc.CoreFuncs):
    """The core functions to realize a VMC calculation.

    The VMC sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a uniform distribution function.
    """

    @property
    def wf_abs_log(self):
        """"""
        return model.core_funcs.wf_abs_log

    @cached_property
    def recast(self):
        """Apply the periodic boundary conditions on a configuration."""

        @jit(nopython=True)
        def _recast(z: float, tpf_spec: Union[TPFSpecNT, UTPFSpecNT]):
            """Apply the periodic boundary conditions on a configuration.

            :param z:
            :param tpf_spec:
            :return:
            """
            z_min = tpf_spec.lower_bound
            z_max = tpf_spec.upper_bound
            return recast_to_supercell(z, z_min, z_max)

        return _recast

    @cached_property
    def ith_sys_conf_tpf(self):
        """

        :return:
        """
        pos_slot = int(self.sys_conf_slots.pos)
        rand_displace = self.rand_displace
        recast = self.recast  # TODO: Use a better name, maybe?

        @jit(nopython=True)
        def _ith_sys_conf_ppf(i_: int,
                              ini_sys_conf: np.ndarray,
                              prop_sys_conf: np.ndarray,
                              tpf_spec: Union[TPFSpecNT, UTPFSpecNT]):
            """Move the i-nth particle of the current configuration of the
            system under PBC.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_spec:.
            :return:
            """
            # Unpack data
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(tpf_spec)
            z_i_upd = recast(z_i + rnd_spread, tpf_spec)
            prop_sys_conf[pos_slot, i_] = z_i_upd

        return _ith_sys_conf_ppf


@attr.s(auto_attribs=True, frozen=True)
class NormalSampling(qmc_base.jastrow.vmc.NormalSampling):
    """The spec of the VMC sampling."""

    model_spec: model.Spec
    time_step: float
    num_steps: int
    ini_sys_conf: np.ndarray
    rng_seed: int = None
    core_funcs: 'CoreFuncs' = attr.ib(init=False, cmp=False, repr=False)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        # NOTE: Should we use a new CoreFuncs instance?
        super().__setattr__('core_funcs', normal_core_funcs)

        if self.rng_seed is None:
            rng_seed = utils.get_random_rng_seed()
            super().__setattr__('rng_seed', rng_seed)

    @property
    def tpf_spec_nt(self):
        """"""
        sigma = sqrt(self.time_step)
        boson_number = self.model_spec.boson_number
        z_min, z_max = self.model_spec.boundaries
        return TPFSpecNT(boson_number, sigma=sigma,
                         lower_bound=z_min, upper_bound=z_max)


class NormalCoreFuncs(CoreFuncs, qmc_base.vmc.NormalCoreFuncs):
    """The core functions to realize a VMC calculation.

    The VMC sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a normal (gaussian) distribution function.
    """
    pass


# Common reference to all the core functions.
core_funcs = CoreFuncs()
normal_core_funcs = NormalCoreFuncs()
