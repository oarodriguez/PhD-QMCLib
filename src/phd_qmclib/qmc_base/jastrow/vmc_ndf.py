from abc import ABCMeta

from phd_qmclib.qmc_base import vmc_ndf
from phd_qmclib.qmc_base.jastrow import model


class TPFParams(vmc_ndf.TPFParams):
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


class CoreFuncs(vmc_ndf.CoreFuncs, metaclass=ABCMeta):
    """"""
    pass
