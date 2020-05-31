import typing as t
import warnings

import attr
import numpy as np
from cached_property import cached_property

from phd_qmclib.constants import ER
from phd_qmclib.qmc_base.jastrow import SysConfDistType
from phd_qmclib.qmc_exec import (
    data as qmc_data, dmc as dmc_exec, exec_logger, proc as proc_base
)
from phd_qmclib.util.attr import (
    bool_converter, bool_validator, int_converter, int_validator,
    opt_int_converter, opt_int_validator, opt_str_validator, str_validator
)
from .. import dmc, model

# String to use a ModelSysConfSpec instance as input for a Proc instance.
MODEL_SYS_CONF_TYPE = 'MODEL_SYS_CONF'

model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class ModelSysConfSpec(dmc_exec.ModelSysConfSpec):
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
    state: dmc.State

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
        # assert proc.model_spec.boson_number == \
        #        proc_result.proc.model_spec.boson_number  # noqa
        return cls(state)


@attr.s(auto_attribs=True, frozen=True)
class ProcResult(proc_base.ProcResult):
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc.State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: qmc_data.dmc.SamplingData


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

    num_blocks: int = attr.ib(default=512,
                              converter=int_converter,
                              validator=int_validator)  # 2^9

    num_time_steps_block: int = attr.ib(default=512,
                                        converter=int_converter,
                                        validator=int_validator)  # 2^9

    burn_in_blocks: t.Optional[int] = attr.ib(default=None,
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

        # Add support for num_batches alias for num_blocks.
        if 'num_batches' in self_config:
            # WARNING ⚠⚠⚠
            warnings.warn("num_batches attribute is deprecated, use "
                          "num_blocks instead", DeprecationWarning)
            # WARNING ⚠⚠⚠
            num_blocks = self_config.pop('num_batches')
            self_config['num_blocks'] = num_blocks

        # Add support for num_time_steps_batch alias for num_time_steps_block.
        if 'num_time_steps_batch' in self_config:
            # WARNING ⚠⚠⚠
            warnings.warn("num_time_steps_batch attribute is deprecated, use "
                          "num_time_steps_block instead", DeprecationWarning)
            # WARNING ⚠⚠⚠
            nts_block = self_config.pop('num_time_steps_batch')
            self_config['num_time_steps_block'] = nts_block

        # Add support for burn_in_batches alias for burn_in_blocks.
        if 'burn_in_batches' in self_config:
            # WARNING ⚠⚠⚠
            warnings.warn("burn_in_batches attribute is deprecated, use "
                          "burn_in_blocks instead", DeprecationWarning)
            # WARNING ⚠⚠⚠
            nts_block = self_config.pop('burn_in_batches')
            self_config['burn_in_blocks'] = nts_block

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
            # NOTE: This item was removed from SSFEstSpec constructor.
            ssf_est_config.pop('pfw_num_time_steps', None)
            ssf_est_spec = SSFEstSpec(**ssf_est_config)
        else:
            ssf_est_spec = None

        # Aliases for jit_parallel and jit_fastmath.
        if 'parallel' in self_config:
            num_blocks = self_config.pop('parallel')
            self_config['jit_parallel'] = num_blocks

        if 'fastmath' in self_config:
            nts_block = self_config.pop('fastmath')
            self_config['jit_fastmath'] = nts_block

        dmc_proc = cls(model_spec=model_spec,
                       density_spec=density_est_spec,
                       ssf_spec=ssf_est_spec,
                       **self_config)

        return dmc_proc

    def as_config(self):
        """

        :return:
        """
        return attr.asdict(self, filter=attr.filters.exclude(type(None)))

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
        pfw_num_time_steps = self.num_time_steps_block

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

    def build_result(self, state: dmc.State,
                     data: qmc_data.dmc.SamplingData):
        """

        :param state:
        :param data:
        :return:
        """
        proc = self
        return ProcResult(state, proc, data)
