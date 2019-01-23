import typing as t

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.qmc_exec import dmc as dmc_exec, exec_logger
from .dmc import EstSampling, SSFEstSpec
from .model import CSWFOptimizer, Spec
from .vmc import Sampling

__all__ = [
    'DMCProc',
    'DMCSSFEstSpec',
    'VMCProc',
    'WFOptProc'
]

float_validator = attr.validators.instance_of((float, int))
int_validator = attr.validators.instance_of(int)
str_validator = attr.validators.instance_of(str)
bool_validator = attr.validators.instance_of(bool)

opt_float_validator = attr.validators.optional(float_validator)
opt_int_validator = attr.validators.optional(int_validator)
opt_str_validator = attr.validators.optional(str_validator)
opt_bool_validator = attr.validators.optional(bool_validator)

model_spec_validator = attr.validators.instance_of(Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class VMCProc(dmc_exec.VMCProc):
    """VMC Sampling."""

    model_spec: Spec = attr.ib(validator=model_spec_validator)

    move_spread: float = attr.ib(converter=float)

    rng_seed: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    num_batches: int = \
        attr.ib(default=8, validator=int_validator)

    num_steps_batch: int = \
        attr.ib(default=4096, validator=int_validator)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        self_config = dict(config)
        # Extract the model spec.
        model_spec_config = self_config.pop('model_spec')
        model_spec = Spec(**model_spec_config)

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
    def sampling(self) -> Sampling:
        """

        :return:
        """
        return Sampling(self.model_spec,
                        self.move_spread,
                        self.rng_seed)


vmc_proc_validator = attr.validators.instance_of(VMCProc)
opt_vmc_proc_validator = attr.validators.optional(vmc_proc_validator)


@attr.s(auto_attribs=True, frozen=True)
class WFOptProc(dmc_exec.WFOptProc):
    """Wave function optimization."""

    #: The number of configurations used in the process.
    num_sys_confs: int = \
        attr.ib(default=1024, validator=int_validator)

    #: The energy of reference to minimize the variance of the local energy.
    ref_energy: t.Optional[float] = \
        attr.ib(default=None, converter=float)

    #: Use threads or multiple process.
    use_threads: bool = attr.ib(default=True, validator=bool_validator)

    #: Number of threads or process to use.
    num_workers: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    #: Display log messages or not.
    verbose: bool = attr.ib(default=False, validator=bool_validator)

    def exec(self, model_spec: Spec,
             sys_conf_set: np.ndarray,
             ini_wf_abs_log_set: np.ndarray):
        """

        :param model_spec:
        :param sys_conf_set: he system configurations used for the
            minimization process.
        :param ini_wf_abs_log_set: The initial wave function values. Used
            to calculate the weights.
        :return:
        """
        num_sys_confs = self.num_sys_confs

        exec_logger.info('Starting wave function optimization...')
        exec_logger.info(f'Using {num_sys_confs} configurations to '
                         f'minimize the variance...')

        sys_conf_set = sys_conf_set[-num_sys_confs:]
        ini_wf_abs_log_set = ini_wf_abs_log_set[-num_sys_confs:]

        optimizer = CSWFOptimizer(model_spec,
                                  sys_conf_set,
                                  ini_wf_abs_log_set,
                                  self.ref_energy,
                                  self.use_threads,
                                  self.num_workers,
                                  self.verbose)
        opt_result = optimizer.exec()

        exec_logger.info('Wave function optimization completed.')

        return opt_result


wf_opt_proc_validator = attr.validators.instance_of(WFOptProc)
opt_wf_opt_proc_validator = attr.validators.optional(wf_opt_proc_validator)


@attr.s(auto_attribs=True, frozen=True)
class DMCSSFEstSpec(dmc_exec.SSFEstSpec):
    """Structure factor estimator basic config."""

    num_modes: int = attr.ib(validator=int_validator)

    as_pure_est: bool = attr.ib(default=True, validator=bool_validator)

    pfw_num_time_steps: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)


ssf_validator = attr.validators.instance_of(dmc_exec.SSFEstSpec)
opt_ssf_validator = attr.validators.optional(ssf_validator)


@attr.s(auto_attribs=True, frozen=True)
class DMCProc(dmc_exec.DMCProc):
    """DMC sampling procedure."""

    model_spec: Spec = attr.ib(validator=model_spec_validator)

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
    ssf_spec: t.Optional[DMCSSFEstSpec] = \
        attr.ib(default=None, validator=opt_ssf_validator)

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
        model_spec = Spec(**model_spec_config)

        # Extract the spec of the static structure factor.
        ssf_est_config = self_config.pop('ssf_spec', None)
        if ssf_est_config is not None:
            ssf_est_spec = DMCSSFEstSpec(**ssf_est_config)
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
                ssf_est_spec = DMCSSFEstSpec(**ssf_est_config)

        return attr.evolve(self, model_spec=model_spec,
                           ssf_spec=ssf_est_spec,
                           **self_config)

    @cached_property
    def sampling(self) -> EstSampling:
        """

        :return:
        """
        if self.should_eval_ssf:
            ssf_spec = self.ssf_spec
            ssf_est = SSFEstSpec(self.model_spec,
                                 ssf_spec.num_modes,
                                 ssf_spec.as_pure_est,
                                 ssf_spec.pfw_num_time_steps)

        else:
            ssf_est = None

        sampling = EstSampling(self.model_spec,
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


dmc_proc_validator = attr.validators.instance_of(DMCProc)
opt_dmc_proc_validator = attr.validators.optional(dmc_proc_validator)
