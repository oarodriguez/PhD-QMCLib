from math import sqrt
from typing import NamedTuple, Union

import attr
import numpy as np
from cached_property import cached_property
from numba import jit

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.utils import recast_to_supercell
from .model import Spec, core_funcs as model_core_funcs

__all__ = [
    'Sampling',
    'TPFSpecNT',
    'UniformVMCCoreFuncs',
    'UTPFSpecNT',
    'VMCSpec',
    'VMCCoreFuncs',
    'vmc_core_funcs',
]


class TPFSpecNT(qmc_base.jastrow.vmc.TPFSpecNT, NamedTuple):
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
class VMCSpec(qmc_base.jastrow.vmc.Spec):
    """The spec of the VMC sampling."""

    model_spec: Spec
    time_step: float
    chain_samples: int
    ini_sys_conf: np.ndarray
    burn_in_samples: int = 0
    rng_seed: int = None

    def __attrs_post_init__(self):
        """"""
        if self.rng_seed is None:
            rng_seed = utils.get_random_rng_seed()
            super().__setattr__('rng_seed', rng_seed)

    @property
    def tpf_spec_nt(self):
        """"""
        boson_number = self.model_spec.boson_number
        sigma = sqrt(self.time_step)
        z_min, z_max = self.model_spec.boundaries
        return TPFSpecNT(boson_number, sigma=sigma, lower_bound=z_min,
                         upper_bound=z_max)


class VMCCoreFuncs(qmc_base.jastrow.vmc.CoreFuncs):
    """Sampling of the probability density of the Bloch-Phonon model.

    The sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a normal (gaussian) distribution function.
    """

    @property
    def wf_abs_log(self):
        """"""
        return model_core_funcs.wf_abs_log

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

        @jit(nopython=True, cache=True)
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


#
vmc_core_funcs = VMCCoreFuncs()


class UniformVMCCoreFuncs(VMCCoreFuncs, qmc_base.vmc.UniformCoreFuncs):
    """Sampling of the probability density of the Bloch-Phonon model.

    The sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a uniform distribution function.
    """
    pass


@attr.s(auto_attribs=True, frozen=True)
class Sampling(qmc_base.vmc.Sampling):
    """Realizes a VMC sampling using an iterable interface."""

    spec: VMCSpec
    core_funcs: VMCCoreFuncs = attr.ib(init=False, cmp=False, repr=False)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # NOTE: Should we use a new VMCCoreFuncs instance?
        super().__setattr__('core_funcs', vmc_core_funcs)
