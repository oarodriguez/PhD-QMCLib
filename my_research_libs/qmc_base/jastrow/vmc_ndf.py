import typing as t
from abc import ABCMeta
from math import sqrt

from my_research_libs.qmc_base import vmc_ndf
from my_research_libs.qmc_base.jastrow import model


class TPFSpecNT(vmc_ndf.TPFSpecNT, t.NamedTuple):
    """The parameters of the transition probability function.

    The parameters correspond to a sampling done with random numbers
    generated from a (normal) gaussian distribution function.
    """
    boson_number: int
    sigma: float


class Sampling(vmc_ndf.Sampling, metaclass=ABCMeta):
    """Spec for the VMC sampling of a Bijl-Jastrow model."""

    #: The spec of a concrete Jastrow model.
    model_spec: model.Spec

    time_step: float
    rng_seed: int

    @property
    def tpf_spec_nt(self):
        """"""
        sigma = sqrt(self.time_step)
        number = self.model_spec.boson_number
        return TPFSpecNT(number, sigma)


class CoreFuncs(vmc_ndf.CoreFuncs, metaclass=ABCMeta):
    """"""
    pass
