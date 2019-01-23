import typing as t

import attr
from cached_property import cached_property

from my_research_libs.multirods_qmc.bloch_phonon import dmc, model
from my_research_libs.qmc_exec import dmc as dmc_exec_base
from my_research_libs.util.attr import (
    bool_validator, int_validator, opt_int_validator
)

__all__ = [
    'Proc',
    'SSFEstSpec'
]

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


dmc_proc_validator = attr.validators.instance_of(Proc)
opt_dmc_proc_validator = attr.validators.optional(dmc_proc_validator)
