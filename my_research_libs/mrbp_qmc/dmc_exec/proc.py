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
    opt_int_converter, opt_int_validator, opt_str_validator, str_validator
)

# String to use a ModelSysConfSpec instance as input for a Proc instance.
MODEL_SYS_CONF_TYPE = 'MODEL_SYS_CONF'

model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class ModelSysConfSpec(dmc_exec.proc.ModelSysConfSpec):
    """Handler to build inputs from system configurations."""

    #:
    dist_type: str = attr.ib(validator=str_validator)

    #:
    num_sys_conf: t.Optional[int] = attr.ib(default=None,
                                            validator=opt_int_validator)

    #: A tag to identify this handler.
    type: str = attr.ib(default=None, validator=opt_str_validator)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # This is the type tag, and must be fixed.
        object.__setattr__(self, 'type', f'{MODEL_SYS_CONF_TYPE}')

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        self_config = dict(config)
        return cls(**self_config)

    def dist_type_as_type(self) -> SysConfDistType:
        """

        :return:
        """
        dist_type = self.dist_type
        if dist_type is None:
            dist_type_enum = SysConfDistType.RANDOM
        else:
            if dist_type not in SysConfDistType.__members__:
                raise ValueError
            dist_type_enum = SysConfDistType[dist_type]
        return dist_type_enum


@attr.s(auto_attribs=True, frozen=True)
class DensityEstSpec(dmc_exec.DensityEstSpec):
    """Structure factor estimator basic config."""

    num_bins: int = \
        attr.ib(converter=int_converter, validator=int_validator)

    as_pure_est: bool = attr.ib(default=True,
                                converter=bool_converter,
                                validator=bool_validator)


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""

    num_modes: int = \
        attr.ib(converter=int_converter, validator=int_validator)

    as_pure_est: bool = attr.ib(default=True,
                                converter=bool_converter,
                                validator=bool_validator)


density_validator = attr.validators.instance_of(DensityEstSpec)
opt_density_validator = attr.validators.optional(density_validator)

ssf_validator = attr.validators.instance_of(SSFEstSpec)
opt_ssf_validator = attr.validators.optional(ssf_validator)


@attr.s(auto_attribs=True)
class ProcInput(dmc_exec.ProcInput):
    """Represents the input for the DMC calculation procedure."""
    # The state of the DMC procedure input.
    # NOTE: Is this class necessary? 🤔
    state: dmc_base.State

    @classmethod
    def from_model_sys_conf_spec(cls, sys_conf_spec: ModelSysConfSpec,
                                 proc: 'Proc'):
        """

        :param sys_conf_spec:
        :param proc:
        :return:
        """
        model_spec = proc.model_spec
        dist_type = sys_conf_spec.dist_type_as_type()
        num_sys_conf = sys_conf_spec.num_sys_conf

        sys_conf_set = []
        num_sys_conf = num_sys_conf or proc.target_num_walkers
        for _ in range(num_sys_conf):
            sys_conf = \
                model_spec.init_get_sys_conf(dist_type=dist_type)
            sys_conf_set.append(sys_conf)

        sys_conf_set = np.asarray(sys_conf_set)
        state = proc.sampling.build_state(sys_conf_set)
        return cls(state)

    @classmethod
    def from_result(cls, proc_result: 'ProcResult',
                    proc: 'Proc'):
        """

        :param proc_result:
        :param proc:
        :return:
        """
        state = proc_result.state
        assert proc.model_spec == proc_result.proc.model_spec
        return cls(state)


class ProcInputError(ValueError):
    """Flags an invalid input for a DMC calculation procedure."""
    pass


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
    # *** Estimators configuration ***
    density_spec: t.Optional[DensityEstSpec] = \
        attr.ib(default=None, validator=None)

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

        # Extract the spec of the density.
        density_est_config = self_config.pop('density_spec', None)
        if density_est_config is not None:
            density_est_spec = DensityEstSpec(**density_est_config)
        else:
            density_est_spec = None

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
                       density_spec=density_est_spec,
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
        pfw_num_time_steps = self.num_time_steps_batch

        if self.should_eval_density:
            density_spec = self.density_spec
            density_est_spec = \
                dmc.DensityEstSpec(density_spec.num_bins,
                                   density_spec.as_pure_est,
                                   pfw_num_time_steps)
        else:
            density_est_spec = None

        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est_spec = dmc.SSFEstSpec(ssf_spec.num_modes,
                                          ssf_spec.as_pure_est,
                                          pfw_num_time_steps)

        else:
            ssf_est_spec = None

        sampling = dmc.Sampling(self.model_spec,
                                self.time_step,
                                self.max_num_walkers,
                                self.target_num_walkers,
                                self.num_walkers_control_factor,
                                self.rng_seed,
                                density_est_spec=density_est_spec,
                                ssf_est_spec=ssf_est_spec)
        return sampling

    def checkpoint(self):
        """"""
        pass

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
