import typing as t

import attr
import numpy as np
from cached_property import cached_property
from numba import jit

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.jastrow import vmc as vmc_base
from my_research_libs.qmc_base.utils import recast_to_supercell
from my_research_libs.util.attr import Record
from . import model

__all__ = [
    'CoreFuncs',
    'Sampling',
    'StateError',
    'StateProp',
    'TPFParams',
    'core_funcs'
]

# Export symbols from base modules.
StateProp = qmc_base.vmc.StateProp
STAT_REJECTED = qmc_base.vmc.STAT_REJECTED


@attr.s(auto_attribs=True, frozen=True)
class TPFParams(qmc_base.jastrow.vmc.TPFParams, Record):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float
    lower_bound: float
    upper_bound: float


class TPFSpec(t.NamedTuple):
    """"""
    tpf_params: TPFParams


class CFCSpec(vmc_base.CFCSpec, t.NamedTuple):
    """The spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    tpf_params: TPFParams


class StateError(ValueError):
    """Flags errors related to the handling of a VMC state."""
    pass


@attr.s(auto_attribs=True, frozen=True)
class Sampling(qmc_base.jastrow.vmc.Sampling):
    """The spec of the VMC sampling."""

    model_spec: model.Spec

    move_spread: float

    rng_seed: t.Optional[int] = attr.ib(default=None)

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            super().__setattr__('rng_seed', rng_seed)

    @property
    def tpf_params(self):
        """"""
        move_spread = self.move_spread
        boson_number = self.model_spec.boson_number
        z_min, z_max = self.model_spec.boundaries
        tpf_params = TPFParams(boson_number=boson_number,
                               move_spread=move_spread,
                               lower_bound=z_min,
                               upper_bound=z_max)
        return tpf_params.as_record()

    @property
    def cfc_spec(self) -> CFCSpec:
        """"""
        model_spec = self.model_spec
        return CFCSpec(model_spec.params, model_spec.obf_params,
                       model_spec.tbf_params, self.tpf_params)

    def build_state(self, sys_conf: np.ndarray) -> qmc_base.vmc.State:
        """Builds a state for the sampling.

        The state includes the drift, the energies wne the weights of
        each one of the initial system configurations.

        :param sys_conf: The configuration of the state.
        """
        # noinspection PyTypeChecker
        sys_conf = np.asarray(sys_conf)
        if sys_conf.shape != self.model_spec.sys_conf_shape:
            raise StateError("sys_conf is not a valid configuration "
                             "of the model spec")

        cfc_spec = self.cfc_spec
        wf_abs_log = self.core_funcs.wf_abs_log(sys_conf, cfc_spec)
        return qmc_base.vmc.State(sys_conf, wf_abs_log, STAT_REJECTED)

    @property
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        # NOTE: Should we use a new CoreFuncs instance?
        return core_funcs


class CoreFuncs(qmc_base.jastrow.vmc.CoreFuncs):
    """The core functions to realize a VMC calculation.

    The VMC sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a uniform distribution function.
    """

    @property
    def wf_abs_log(self):
        """"""
        wf_abs_log = model.core_funcs.wf_abs_log

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
    def recast(self):
        """Apply the periodic boundary conditions on a configuration."""

        @jit(nopython=True)
        def _recast(z: float, tpf_params: TPFParams):
            """Apply the periodic boundary conditions on a configuration.

            :param z:
            :param tpf_params:
            :return:
            """
            z_min = tpf_params.lower_bound
            z_max = tpf_params.upper_bound
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
                              tpf_params: TPFParams):
            """Move the i-nth particle of the current configuration of the
            system under PBC.

            :param i_:
            :param ini_sys_conf: The current (initial) configuration.
            :param prop_sys_conf: The proposed configuration.
            :param tpf_params:.
            :return:
            """
            # Unpack data
            z_i = ini_sys_conf[pos_slot, i_]
            rnd_spread = rand_displace(tpf_params)
            z_i_upd = recast(z_i + rnd_spread, tpf_params)
            prop_sys_conf[pos_slot, i_] = z_i_upd

        return _ith_sys_conf_ppf


# Common reference to all the core functions.
core_funcs = CoreFuncs()
