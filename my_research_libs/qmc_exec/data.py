import typing as t

import attr

from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.qmc_data.dmc import SamplingData

__all__ = [
    'DMCProcResult'
]


@attr.s(auto_attribs=True, frozen=True)
class DMCProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The data generated during the sampling.
    data: t.Optional[SamplingData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[dmc_base.EstSampling] = None
