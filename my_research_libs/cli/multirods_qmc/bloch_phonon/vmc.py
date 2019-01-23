import typing as t

import attr
from cached_property import cached_property

from my_research_libs.multirods_qmc.bloch_phonon import model, vmc
from my_research_libs.qmc_exec import vmc as vmc_exec_base
from my_research_libs.util.attr import int_validator, opt_int_validator

model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)

__all__ = [
    'Proc'
]


@attr.s(auto_attribs=True, frozen=True)
class Proc(vmc_exec_base.Proc):
    """VMC Sampling."""

    model_spec: model.Spec = attr.ib(validator=model_spec_validator)

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
    def sampling(self) -> vmc.Sampling:
        """

        :return:
        """
        return vmc.Sampling(self.model_spec,
                            self.move_spread,
                            self.rng_seed)


vmc_proc_validator = attr.validators.instance_of(Proc)
opt_vmc_proc_validator = attr.validators.optional(vmc_proc_validator)
