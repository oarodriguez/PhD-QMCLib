import typing as t

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.multirods_qmc.bloch_phonon import dmc, model
from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.qmc_base.jastrow import SysConfDistType
from my_research_libs.qmc_data.dmc import SamplingData
from my_research_libs.qmc_exec import dmc as dmc_exec_base
from my_research_libs.qmc_exec.dmc import ProcInput
from my_research_libs.util.attr import (
    bool_validator, int_validator, opt_int_validator,
    str_validator
)

__all__ = [
    'HDF5FileHandler',
    'IOHandlerSpec',
    'ModelSysConfHandler',
    'Proc',
    'ProcIO',
    'SSFEstSpec'
]


@attr.s(auto_attribs=True, frozen=True)
class ModelSysConfHandler(dmc_exec_base.ModelSysConfHandler):
    """"""

    dist_type: str = attr.ib(validator=str_validator)

    def load(self):
        """"""
        raise NotImplementedError

    def save(self, data: 'ProcResult'):
        """"""
        raise NotImplementedError

    def get_dist_type(self):
        """

        :return:
        """
        dist_type = self.dist_type

        if dist_type is None:
            dist_type = SysConfDistType.RANDOM
        else:
            if dist_type not in SysConfDistType.__members__:
                raise ValueError

            dist_type = SysConfDistType[dist_type]

        return dist_type


@attr.s(auto_attribs=True, frozen=True)
class HDF5FileHandler(dmc_exec_base.HDF5FileHandler):
    """"""

    location: str = attr.ib(validator=str_validator)

    group: str = attr.ib(validator=str_validator)

    dataset: str = attr.ib(validator=str_validator)

    is_state: bool = attr.ib(default=True, validator=bool_validator)

    def load(self):
        """"""

    def save(self, data: 'ProcResult'):
        """"""
        pass


T_IOHandlerSpec = \
    t.Union[ModelSysConfHandler, HDF5FileHandler]

io_handler_spec_type_validator = [
    attr.validators.instance_of(str),
    attr.validators.in_(('MODEL_SYS_CONF', 'HDF5_FILE'))
]

io_handler_spec_types = \
    ModelSysConfHandler, HDF5FileHandler

# noinspection PyTypeChecker
io_handler_spec_validator = attr.validators.instance_of(io_handler_spec_types)


@attr.s(auto_attribs=True, frozen=True)
class IOHandlerSpec(dmc_exec_base.IOHandlerSpec):
    """"""

    type: str = attr.ib(validator=io_handler_spec_type_validator)

    spec: T_IOHandlerSpec = attr.ib(validator=io_handler_spec_validator)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        io_handler_type = config['type']
        io_handler = config['spec']

        if io_handler_type == 'MODEL_SYS_CONF':
            io_handler = ModelSysConfHandler(**io_handler)

        elif io_handler_type == 'HDF5_FILE':
            io_handler = HDF5FileHandler(**io_handler)

        else:
            raise ValueError

        return cls(io_handler_type, io_handler)


@attr.s(auto_attribs=True)
class ProcIO(dmc_exec_base.ProcIO):
    """"""
    #:
    input: IOHandlerSpec

    #:
    output: t.Optional[IOHandlerSpec] = None

    @classmethod
    def from_config(cls, config: t.Mapping):
        """"""
        input_spec_config = config['input']
        input_spec = IOHandlerSpec.from_config(input_spec_config)

        # Extract the output spec.
        output_spec_config = config['output']
        output_spec = IOHandlerSpec.from_config(output_spec_config)

        return cls(input_spec, output_spec)


model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec_base.SSFEstSpec):
    """Structure factor estimator basic config."""

    num_modes: int = attr.ib(validator=int_validator)

    as_pure_est: bool = attr.ib(default=True, validator=bool_validator)

    pfw_num_time_steps: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)


ssf_validator = attr.validators.instance_of(SSFEstSpec)
opt_ssf_validator = attr.validators.optional(ssf_validator)


@attr.s(auto_attribs=True, frozen=True)
class ProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The sampling object used to generate the results.
    sampling: dmc.EstSampling

    #: The data generated during the sampling.
    data: t.Optional[SamplingData] = None


@attr.s(auto_attribs=True, frozen=True)
class Proc(dmc_exec_base.Proc):
    """DMC sampling procedure."""

    model_spec: model.Spec = attr.ib(validator=None)

    time_step: float = attr.ib(converter=float)

    max_num_walkers: int = \
        attr.ib(default=512, validator=int_validator)

    target_num_walkers: int = \
        attr.ib(default=480, validator=int_validator)

    num_walkers_control_factor: t.Optional[float] = \
        attr.ib(default=0.5, converter=float)

    rng_seed: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    num_batches: int = \
        attr.ib(default=512, validator=int_validator)  # 2^9

    num_time_steps_batch: int = \
        attr.ib(default=512, validator=int_validator)  # 2^9

    burn_in_batches: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    keep_iter_data: bool = \
        attr.ib(default=False, validator=bool_validator)

    #: Remaining batches
    remaining_batches: t.Optional[int] = attr.ib(default=None, init=False)

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec] = \
        attr.ib(default=None, validator=None)

    verbose: bool = attr.ib(default=False, validator=bool_validator)

    def __attrs_post_init__(self):
        """"""
        pass

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        self_config = dict(config)

        # Extract the model spec.
        model_spec_config = self_config.pop('model_spec')
        model_spec = model.Spec(**model_spec_config)

        # Extract the spec of the static structure factor.
        ssf_est_config = self_config.pop('ssf_spec', None)
        if ssf_est_config is not None:
            ssf_est_spec = SSFEstSpec(**ssf_est_config)
        else:
            ssf_est_spec = None

        dmc_proc = cls(model_spec=model_spec,
                       ssf_spec=ssf_est_spec,
                       **self_config)

        return dmc_proc

    def evolve(self, config: t.Mapping):
        """

        :param config:
        :return:
        """
        self_config = dict(config)

        # Compound attributes of current instance.
        model_spec = self.model_spec
        ssf_est_spec = self.ssf_spec

        model_spec_config = self_config.pop('model_spec', None)
        if model_spec_config is not None:
            model_spec = attr.evolve(model_spec, **model_spec_config)

        # Extract the spec of the static structure factor.
        ssf_est_config = self_config.pop('ssf_spec', None)
        if ssf_est_config is not None:
            if ssf_est_spec is not None:
                ssf_est_spec = attr.evolve(ssf_est_spec, **ssf_est_config)

            else:
                ssf_est_spec = SSFEstSpec(**ssf_est_config)

        return attr.evolve(self, model_spec=model_spec,
                           ssf_spec=ssf_est_spec,
                           **self_config)

    @cached_property
    def sampling(self) -> dmc.EstSampling:
        """

        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est = dmc.SSFEstSpec(self.model_spec,
                                     ssf_spec.num_modes,
                                     ssf_spec.as_pure_est,
                                     ssf_spec.pfw_num_time_steps)

        else:
            ssf_est = None

        sampling = dmc.EstSampling(self.model_spec,
                                   self.time_step,
                                   self.max_num_walkers,
                                   self.target_num_walkers,
                                   self.num_walkers_control_factor,
                                   self.rng_seed,
                                   ssf_spec=ssf_est)
        return sampling

    def checkpoint(self):
        """"""
        pass

    def build_input(self, proc_io_input: IOHandlerSpec):
        """

        :param proc_io_input:
        :return:
        """
        model_spec = self.model_spec
        io_input = proc_io_input.spec

        if isinstance(io_input, ModelSysConfHandler):

            dist_type = io_input.get_dist_type()
            sys_conf_set = []
            for _ in range(self.target_num_walkers):
                sys_conf = model_spec.init_get_sys_conf(dist_type=dist_type)
                sys_conf_set.append(sys_conf)

            sys_conf_set = np.asarray(sys_conf_set)
            state = self.sampling.build_state(sys_conf_set)
            return ProcInput(state)

        elif isinstance(io_input, HDF5FileHandler):
            pass

        else:
            raise TypeError

    def build_result(self, state: dmc_base.State,
                     sampling: dmc.EstSampling,
                     data: SamplingData = None):
        """

        :param state:
        :param sampling:
        :param data:
        :return:
        """
        return ProcResult(state, sampling, data)


dmc_proc_validator = attr.validators.instance_of(Proc)
opt_dmc_proc_validator = attr.validators.optional(dmc_proc_validator)
