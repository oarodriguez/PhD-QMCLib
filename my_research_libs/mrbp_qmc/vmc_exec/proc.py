import typing as t

import attr
from cached_property import cached_property

from my_research_libs.constants import ER
from my_research_libs.qmc_base import vmc as vmc_udf_base
from my_research_libs.qmc_base.jastrow import SysConfDistType
from my_research_libs.qmc_exec import exec_logger, vmc as vmc_exec
from my_research_libs.util.attr import (
    bool_converter, bool_validator, int_converter, int_validator,
    opt_int_converter, opt_int_validator, opt_str_validator, str_validator
)
from .. import model, vmc as vmc_udf

# String to use a ModelSysConfSpec instance as input for a Proc instance.
MODEL_SYS_CONF_TYPE = 'MODEL_SYS_CONF'

model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class ModelSysConfSpec(vmc_exec.proc.ModelSysConfSpec):
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
class SSFEstSpec(vmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""

    num_modes: int = \
        attr.ib(converter=int_converter, validator=int_validator)


@attr.s(auto_attribs=True)
class ProcInput(vmc_exec.ProcInput):
    """Represents the input for the DMC calculation procedure."""

    #: The state of the DMC procedure input.
    state: vmc_udf_base.State

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
        sys_conf = \
            model_spec.init_get_sys_conf(dist_type=dist_type)

        state = proc.sampling.build_state(sys_conf)
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
class ProcResult(vmc_exec.ProcResult):
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: vmc_udf_base.State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: vmc_exec.data.SamplingData


@attr.s(auto_attribs=True, frozen=True)
class Proc(vmc_exec.Proc):
    """VMC Sampling."""

    model_spec: model.Spec = attr.ib(validator=model_spec_validator)

    move_spread: float = attr.ib(converter=float)

    rng_seed: t.Optional[int] = \
        attr.ib(default=None, converter=opt_int_converter,
                validator=opt_int_validator)

    num_batches: int = \
        attr.ib(default=8, converter=int_converter, validator=int_validator)

    num_steps_batch: int = \
        attr.ib(default=4096, converter=int_converter, validator=int_validator)

    burn_in_batches: t.Optional[int] = attr.ib(default=None,
                                               converter=opt_int_converter,
                                               validator=opt_int_validator)

    keep_iter_data: bool = attr.ib(default=False,
                                   converter=bool_converter,
                                   validator=bool_validator)

    # *** Estimators configuration ***
    # TODO: add proper validator.
    ssf_spec: t.Optional[vmc_udf_base.SSFEstSpec] = \
        attr.ib(default=None, validator=None)

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

        return cls(model_spec, **self_config)

    def evolve(self, config: t.Mapping):
        """

        :param config:
        :return:
        """
        self_config = dict(config)

        # Compound attributes of current instance.
        model_spec = self.model_spec

        # Extract the model spec.
        model_spec_config = self_config.pop('model_spec', None)
        if model_spec_config is not None:
            model_spec = attr.evolve(model_spec, **model_spec_config)

        return attr.evolve(self, model_spec=model_spec, **self_config)

    @cached_property
    def sampling(self) -> vmc_udf.Sampling:
        """

        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est_spec = vmc_udf.SSFEstSpec(ssf_spec.num_modes)
        else:
            ssf_est_spec = None

        vmc_sampling = vmc_udf.Sampling(self.model_spec,
                                        self.move_spread,
                                        self.rng_seed,
                                        ssf_est_spec=ssf_est_spec)
        return vmc_sampling

    def describe_model_spec(self):
        """

        :return:
        """
        model_spec = self.model_spec
        v_zero = model_spec.lattice_depth
        lr = model_spec.lattice_ratio
        gn = model_spec.interaction_strength
        nop = model_spec.boson_number
        sc_size = model_spec.supercell_size
        rm = model_spec.tbf_contact_cutoff

        exec_logger.info('Multi-Rods system parameters:')
        exec_logger.info(f'* Lattice depth: {v_zero / ER:.3G} ER')
        exec_logger.info(f'* Lattice ratio: {lr:.3G}')
        exec_logger.info(f'* Interaction strength: {gn / ER:.3G} ER')
        exec_logger.info(f'* Number of bosons: {nop:d}')
        exec_logger.info(f'* Supercell size: {sc_size:.3G} LKP')
        exec_logger.info(f'* Variational parameters:')
        exec_logger.info(f'  * RM: {rm:.3G} LKP')

    def build_result(self, state: vmc_udf_base.State,
                     sampling: vmc_udf_base.Sampling,
                     data: vmc_exec.data.SamplingData) -> ProcResult:
        """

        :param state:
        :param sampling:
        :param data:
        :return:
        """
        proc = self
        return ProcResult(state, proc, data)


vmc_proc_validator = attr.validators.instance_of(Proc)
opt_vmc_proc_validator = attr.validators.optional(vmc_proc_validator)
