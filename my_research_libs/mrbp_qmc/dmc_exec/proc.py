import typing as t

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.mrbp_qmc import dmc, model
from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.qmc_base.jastrow import SysConfDistType
from my_research_libs.qmc_exec import dmc as dmc_exec
from my_research_libs.util.attr import (
    bool_converter, bool_validator, int_converter, int_validator,
    opt_int_converter, opt_int_validator
)

model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""

    num_modes: int = \
        attr.ib(converter=int_converter, validator=int_validator)

    as_pure_est: bool = attr.ib(default=True,
                                converter=bool_converter,
                                validator=bool_validator)

    pfw_num_time_steps: int = attr.ib(default=99999999,
                                      converter=int_converter,
                                      validator=int_validator)


ssf_validator = attr.validators.instance_of(SSFEstSpec)
opt_ssf_validator = attr.validators.optional(ssf_validator)


@attr.s(auto_attribs=True, frozen=True)
class ProcResult(dmc_exec.ProcResult):
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: dmc_exec.data.SamplingData


@attr.s(auto_attribs=True, frozen=True)
class Proc(dmc_exec.Proc):
    """DMC sampling procedure."""

    model_spec: model.Spec = attr.ib(validator=model_spec_validator)

    time_step: float = attr.ib(converter=float)

    max_num_walkers: int = \
        attr.ib(default=512, converter=int_converter, validator=int_validator)

    target_num_walkers: int = \
        attr.ib(default=480, converter=int_converter, validator=int_validator)

    num_walkers_control_factor: t.Optional[float] = \
        attr.ib(default=0.5, converter=float)

    rng_seed: t.Optional[int] = attr.ib(default=None,
                                        converter=opt_int_converter,
                                        validator=opt_int_validator)

    num_batches: int = attr.ib(default=512,
                               converter=int_converter,
                               validator=int_validator)  # 2^9

    num_time_steps_batch: int = attr.ib(default=512,
                                        converter=int_converter,
                                        validator=int_validator)  # 2^9

    burn_in_batches: t.Optional[int] = attr.ib(default=None,
                                               converter=opt_int_converter,
                                               validator=opt_int_validator)

    keep_iter_data: bool = attr.ib(default=False,
                                   converter=bool_converter,
                                   validator=bool_validator)

    # *** Estimators configuration ***
    ssf_spec: t.Optional[SSFEstSpec] = \
        attr.ib(default=None, validator=None)

    #: Parallel execution where possible.
    jit_parallel: bool = attr.ib(default=True,
                                 converter=bool_converter,
                                 validator=bool_validator)

    #: Use fastmath compiler directive.
    jit_fastmath: bool = attr.ib(default=False,
                                 converter=bool_converter,
                                 validator=bool_validator)

    verbose: bool = attr.ib(default=False,
                            converter=bool_converter,
                            validator=bool_validator)

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

        # Aliases for jit_parallel and jit_fastmath.
        if 'parallel' in self_config:
            parallel = self_config.pop('parallel')
            self_config['jit_parallel'] = parallel

        if 'fastmath' in self_config:
            fastmath = self_config.pop('fastmath')
            self_config['jit_fastmath'] = fastmath

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
    def sampling(self) -> dmc.Sampling:
        """

        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est_spec = dmc.SSFEstSpec(self.model_spec,
                                          ssf_spec.num_modes,
                                          ssf_spec.as_pure_est,
                                          ssf_spec.pfw_num_time_steps)

        else:
            ssf_est_spec = None

        sampling = dmc.Sampling(self.model_spec,
                                self.time_step,
                                self.max_num_walkers,
                                self.target_num_walkers,
                                self.num_walkers_control_factor,
                                self.rng_seed,
                                ssf_est_spec=ssf_est_spec)
        return sampling

    def checkpoint(self):
        """"""
        pass

    def build_input_from_model(self, sys_conf_dist_type: SysConfDistType):
        """

        :param sys_conf_dist_type:
        :return:
        """
        model_spec = self.model_spec

        sys_conf_set = []
        for _ in range(self.target_num_walkers):
            sys_conf = \
                model_spec.init_get_sys_conf(dist_type=sys_conf_dist_type)
            sys_conf_set.append(sys_conf)

        sys_conf_set = np.asarray(sys_conf_set)
        state = self.sampling.build_state(sys_conf_set)
        return dmc_exec.ProcInput(state)

    def build_input_from_result(self, proc_result: ProcResult):
        """

        :param proc_result:
        :return:
        """
        state = proc_result.state
        assert self.model_spec == proc_result.proc.model_spec
        return dmc_exec.ProcInput(state)

    def build_result(self, state: dmc_base.State,
                     sampling: dmc.Sampling,
                     data: dmc_exec.data.SamplingData):
        """

        :param state:
        :param sampling:
        :param data:
        :return:
        """
        proc = self
        return ProcResult(state, proc, data)
