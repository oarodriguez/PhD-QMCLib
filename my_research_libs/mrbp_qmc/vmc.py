import typing as t
from math import pi

import attr
import numpy as np
from cached_property import cached_property
from numba import jit

from my_research_libs import qmc_base, utils
from my_research_libs.qmc_base.jastrow import vmc as jsw_vmc_udf
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
class TPFParams(jsw_vmc_udf.TPFParams, Record):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a uniform distribution function.
    """
    boson_number: int
    move_spread: float
    lower_bound: float
    upper_bound: float


@attr.s(auto_attribs=True)
class SSFParams(jsw_vmc_udf.SSFParams, Record):
    """Static structure factor parameters."""
    num_modes: int
    supercell_size: float
    assume_none: bool = False


class CFCSpec(jsw_vmc_udf.CFCSpec, t.NamedTuple):
    """The spec of the core functions."""
    model_params: model.Params
    obf_params: model.OBFParams
    tbf_params: model.TBFParams
    tpf_params: TPFParams
    ssf_params: t.Optional[SSFParams] = None


class StateError(ValueError):
    """Flags errors related to the handling of a VMC state."""
    pass


@attr.s(auto_attribs=True)
class SSFEstSpec:
    """Structure factor estimator spec."""

    #: Number of modes to evaluate the structure factor S(k).
    num_modes: int


@attr.s(auto_attribs=True, frozen=True)
class Sampling(jsw_vmc_udf.Sampling):
    """The spec of the VMC sampling."""

    model_spec: model.Spec

    move_spread: float

    rng_seed: t.Optional[int] = attr.ib(default=None)

    ssf_est_spec: t.Optional[SSFEstSpec] = None

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
    def ssf_params(self):
        """Static structure factor parameters.

        :return:
        """
        model_spec = self.model_spec
        supercell_size = model_spec.supercell_size
        if self.ssf_est_spec is None:
            num_modes = 1
            assume_none = True
        else:
            num_modes = self.ssf_est_spec.num_modes
            assume_none = False

        ssf_params = \
            SSFParams(num_modes, supercell_size, assume_none=assume_none)
        return ssf_params.as_record()

    @property
    def cfc_spec(self) -> CFCSpec:
        """"""
        model_spec = self.model_spec
        return CFCSpec(model_spec.params,
                       model_spec.obf_params,
                       model_spec.tbf_params,
                       self.tpf_params,
                       self.ssf_params)

    @property
    def ssf_momenta(self):
        """Get the momenta to evaluate the static structure factor.

        :return:
        """
        if self.ssf_est_spec is None:
            raise TypeError('the static structure factor spec has no been '
                            'specified')
        else:
            model_spec = self.model_spec
            ssf_est_spec = self.ssf_est_spec
            num_modes = ssf_est_spec.num_modes
            supercell_size = model_spec.supercell_size
            return np.arange(num_modes) * 2 * pi / supercell_size

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


class CoreFuncs(jsw_vmc_udf.CoreFuncs):
    """The core functions to realize a VMC calculation.

    The VMC sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a uniform distribution function.
    """

    @property
    def model_core_funcs(self) -> model.CoreFuncs:
        """The core functions of the model."""
        return model.core_funcs

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

    @cached_property
    def init_obd_est_data(self):
        return None

    @cached_property
    def init_ssf_est_data(self):
        """

        :return:
        """
        ssf_num_parts = len(qmc_base.vmc.SSFPartSlot.__members__)

        @jit(nopython=True)
        def _init_ssf_est_data(num_steps_batch: int,
                               cfc_spec: CFCSpec):
            """

            :param cfc_spec:
            :return:
            """
            ssf_params = cfc_spec.ssf_params
            supercell_size = ssf_params.supercell_size
            if ssf_params.assume_none:
                num_modes = 1
                # A fake array will be created. It won't be used.
                i_ssf_shape = 1, 1, ssf_num_parts
            else:
                num_modes = ssf_params.num_modes
                i_ssf_shape = num_steps_batch, num_modes, ssf_num_parts

            ssf_momenta = np.arange(num_modes) * 2 * pi / supercell_size
            iter_ssf_array = np.empty(i_ssf_shape, dtype=np.float64)
            return qmc_base.vmc.SSFExecData(ssf_momenta, iter_ssf_array)

        return _init_ssf_est_data


# Common reference to all the core functions.
core_funcs = CoreFuncs()
