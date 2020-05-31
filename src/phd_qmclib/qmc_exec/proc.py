import typing as t
from abc import ABCMeta, abstractmethod

from phd_qmclib.qmc_base import (
    dmc as dmc_base, model as model_base, vmc as vmc_base_udf
)
from . import data

DMC_TASK_LOG_NAME = f'DMC Sampling'
VMC_SAMPLING_LOG_NAME = 'VMC Sampling'

T_Sampling = t.Union[dmc_base.Sampling, vmc_base_udf.Sampling]
T_State = t.Union[dmc_base.State, vmc_base_udf.State]
T_SamplingData = t.Union[data.vmc.SamplingData, data.dmc.SamplingData]


class ModelSysConfSpec(metaclass=ABCMeta):
    """Handler to build inputs from system configurations."""

    #:
    dist_type: str

    #:
    num_sys_conf: t.Optional[int]


class DensityEstSpec(metaclass=ABCMeta):
    """Density estimator basic config."""
    #: The number of bins to estimate the density.
    num_bins: int


class SSFEstSpec(metaclass=ABCMeta):
    """Structure factor estimator basic config."""
    #: The number of momenta modes.
    num_modes: int


class ProcInput(metaclass=ABCMeta):
    """Represents the input for the DMC calculation procedure."""
    # The state of the DMC procedure input.
    state: T_State

    @classmethod
    @abstractmethod
    def from_model_sys_conf_spec(cls, sys_conf_spec: ModelSysConfSpec,
                                 proc: 'Proc'):
        pass

    @classmethod
    @abstractmethod
    def from_result(cls, proc_result: 'ProcResult',
                    proc: 'Proc'):
        pass


class ProcResult(metaclass=ABCMeta):
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: T_State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: T_SamplingData


class Proc(metaclass=ABCMeta):
    """DMC sampling procedure spec."""

    #: The model spec.
    model_spec: model_base.Spec

    #: Keep the estimator values for all the time steps.
    keep_iter_data: bool

    # *** Estimators configuration ***
    #: Density estimator spec.
    density_spec: t.Optional[DensityEstSpec]

    #: Static structure factor estimator spec.
    ssf_spec: t.Optional[SSFEstSpec]

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        pass

    @abstractmethod
    def as_config(self) -> t.Dict:
        """Converts the procedure to a dictionary / mapping object."""
        pass

    @property
    def should_eval_density(self):
        """"""
        return False if self.density_spec is None else True

    @property
    def should_eval_ssf(self):
        """"""
        return False if self.ssf_spec is None else True

    @property
    @abstractmethod
    def sampling(self) -> T_Sampling:
        pass

    @abstractmethod
    def describe_model_spec(self):
        """Describe the spec of the model."""
        pass

    @abstractmethod
    def build_result(self, state: dmc_base.State,
                     sampling_data: T_SamplingData) -> ProcResult:
        """

        :param state: The last state of the sampling.
        :param sampling_data: The data generated during the sampling.
        :return:
        """
        pass

    def checkpoint(self):
        """"""
        pass

    @abstractmethod
    def exec(self, proc_input: ProcInput):
        """

        :param proc_input:
        :return:
        """
        pass
