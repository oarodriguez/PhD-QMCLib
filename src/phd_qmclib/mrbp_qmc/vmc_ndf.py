import typing as t

import attr
from math import sqrt

from phd_qmclib import qmc_base, utils
from phd_qmclib.qmc_base import jastrow
from . import model, vmc as vmc_udf


class TPFParams(qmc_base.jastrow.vmc_ndf.TPFParams, t.NamedTuple):
    """Parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a normal (gaussian) distribution function.
    """
    boson_number: int
    sigma: float
    lower_bound: float
    upper_bound: float


@attr.s(auto_attribs=True, frozen=True)
class Sampling(vmc_udf.Sampling, jastrow.vmc_ndf.Sampling):
    """The spec of the VMC sampling."""

    model_spec: model.Spec
    time_step: float
    rng_seed: t.Optional[int] = attr.ib(default=None)
    ssf_est_spec: t.Optional[vmc_udf.SSFEstSpec] = None

    def __attrs_post_init__(self):
        """Post-initialization stage."""
        if self.rng_seed is None:
            rng_seed = int(utils.get_random_rng_seed())
            super().__setattr__('rng_seed', rng_seed)

    @property
    def tpf_params(self):
        """"""
        sigma = sqrt(self.time_step)
        boson_number = self.model_spec.boson_number
        z_min, z_max = self.model_spec.boundaries
        return TPFParams(boson_number, sigma=sigma,
                         lower_bound=z_min, upper_bound=z_max)

    @property
    def core_funcs(self) -> 'CoreFuncs':
        """The core functions of the sampling."""
        # NOTE: Should we use a new CoreFuncs instance?
        return core_funcs


class CoreFuncs(vmc_udf.CoreFuncs, jastrow.vmc_ndf.CoreFuncs):
    """The core functions to realize a VMC calculation.

    The VMC sampling is subject to periodic boundary conditions due to the
    multi-rods external potential. The random numbers used in the calculation
    are generated from a normal (gaussian) distribution function.
    """
    pass


core_funcs = CoreFuncs()
