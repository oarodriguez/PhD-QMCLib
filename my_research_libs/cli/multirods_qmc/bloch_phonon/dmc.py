import os
import typing as t

import attr
import h5py
import numpy as np
from cached_property import cached_property

from my_research_libs.multirods_qmc.bloch_phonon import dmc, model
from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.qmc_base.jastrow import SysConfDistType
from my_research_libs.qmc_data.dmc import SamplingData
from my_research_libs.qmc_exec import dmc as dmc_exec_base, exec_logger
from my_research_libs.qmc_exec.dmc import ProcInput
from my_research_libs.util.attr import (
    bool_converter, bool_validator, int_converter, int_validator,
    opt_int_converter, opt_int_validator, str_validator
)

__all__ = [
    'RawHDF5FileHandler',
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

    #: A tag to identify this handler.
    type: str = attr.ib(validator=str_validator)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # This is the type tag, and must be fixed.
        object.__setattr__(self, 'type', 'MODEL_SYS_CONF')

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        return cls(**config)

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
class RawHDF5FileHandler(dmc_exec_base.RawHDF5FileHandler):
    """ handler for HDF5 files without a specific structure."""

    location: str = attr.ib(validator=str_validator)

    group: str = attr.ib(validator=str_validator)

    dataset: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str = attr.ib(validator=str_validator)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # This is the type tag, and must be fixed.
        object.__setattr__(self, 'type', 'RAW_HDF5_FILE')

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        return cls(**config)

    def load(self):
        pass

    def save(self, data: 'ProcResult'):
        pass


@attr.s(auto_attribs=True, frozen=True)
class HDF5FileHandler(dmc_exec_base.HDF5FileHandler):
    """A handler for structured HDF5 files to save DMC procedure results."""

    #: Path to the file.
    location: str = attr.ib(validator=str_validator)

    #: The HDF5 group in the file to read and/or write data.
    group: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str = attr.ib(validator=str_validator)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # This is the type tag, and must be fixed.
        object.__setattr__(self, 'type', 'HDF5_FILE')

        # Get an absolute location for the file.
        abs_location = os.path.abspath(os.path.expandvars(self.location))
        object.__setattr__(self, 'location', abs_location)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        return cls(**config)

    def load(self):
        """Load a pro

        :return:
        """
        h5_file = h5py.File(self.location, 'r')
        with h5_file:
            state = self.load_state(h5_file)
            proc = self.load_proc(h5_file)
            data_blocks = self.load_data_blocks(h5_file)

        data_series = None  # For now...
        sampling_data = SamplingData(data_blocks, series=data_series)
        return ProcResult(state, proc, sampling_data)

    def get_proc_config(self, proc: 'Proc'):
        """Converts the procedure to a dictionary / mapping object.

        :param proc:
        :return:
        """
        return attr.asdict(proc, filter=attr.filters.exclude(type(None)))

    def load_proc(self, h5_file: h5py.File):
        """Load the procedure results from the file.

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        proc_group = base_group.require_group('dmc/proc_spec')

        model_spec_group = proc_group.require_group('model_spec')
        model_spec_config = dict(model_spec_group.attrs.items())

        ssf_spec_group: h5py.Group = proc_group.get('ssf_spec')
        if ssf_spec_group is not None:
            ssf_spec_config = dict(ssf_spec_group.attrs.items())
        else:
            ssf_spec_config = None

        # Build a config object.
        proc_config = {
            'model_spec': model_spec_config,
            'ssf_spec': ssf_spec_config
        }
        proc_config.update(proc_group.attrs.items())

        return Proc.from_config(proc_config)


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

    proc_id: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    def __attrs_post_init__(self):
        """

        :return:
        """
        self_spec = self.spec

        if isinstance(self_spec, HDF5FileHandler):

            proc_id = self.proc_id
            spec_group = self_spec.group

            if proc_id is not None:
                group_suffix = 'proc-id-#' + str(proc_id)
                spec_group = '_'.join([spec_group, group_suffix])

            self_spec = attr.evolve(self_spec, group=spec_group)
            object.__setattr__(self, 'spec', self_spec)

        # Reset proc_id to None.
        object.__setattr__(self, 'proc_id', None)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        io_handler_type = config['type']
        io_handler = config['spec']
        proc_id = config.get('proc_id', None)

        if io_handler_type == 'MODEL_SYS_CONF':
            io_handler = ModelSysConfHandler(**io_handler)

        elif io_handler_type == 'HDF5_FILE':
            io_handler = HDF5FileHandler(**io_handler)

        else:
            raise ValueError

        return cls(io_handler_type, io_handler, proc_id)


@attr.s(auto_attribs=True)
class ProcIO(dmc_exec_base.ProcIO):
    """"""
    #:
    input: IOHandlerSpec

    #:
    output: t.Optional[IOHandlerSpec] = None

    #: The procedure id.
    #: Used to store a ProcResult in different HDF5 groups.
    proc_id: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    def __attrs_post_init__(self):
        """"""
        proc_id = self.proc_id

        if proc_id is not None:
            self_input = attr.evolve(self.input, proc_id=proc_id)
            object.__setattr__(self, 'input', self_input)

            if self.output is not None:
                self_output = attr.evolve(self.output, proc_id=proc_id)
                object.__setattr__(self, 'output', self_output)

            # Reset proc_id to None.
            object.__setattr__(self, 'proc_id', None)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """

        input_spec_config = dict(config['input'])
        input_handler = IOHandlerSpec.from_config(input_spec_config)

        # Extract the output spec.
        output_spec_config = dict(config['output'])
        output_handler = IOHandlerSpec.from_config(output_spec_config)

        if not isinstance(output_handler.spec, HDF5FileHandler):
            raise TypeError('only the HDF5_FILE is supported as '
                            'output handler')

        proc_id = config.get('proc_id', None)
        return cls(input_handler, output_handler, proc_id)


model_spec_validator = attr.validators.instance_of(model.Spec)
opt_model_spec_validator = attr.validators.optional(model_spec_validator)


@attr.s(auto_attribs=True, frozen=True)
class SSFEstSpec(dmc_exec_base.SSFEstSpec):
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

T_IOHandler = \
    t.Union[ModelSysConfHandler, HDF5FileHandler]


@attr.s(auto_attribs=True, frozen=True)
class ProcResult(dmc_exec_base.ProcResult):
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The sampling object used to generate the results.
    proc: 'Proc'

    #: The data generated during the sampling.
    data: SamplingData


@attr.s(auto_attribs=True, frozen=True)
class Proc(dmc_exec_base.Proc):
    """DMC sampling procedure."""

    model_spec: model.Spec = attr.ib(validator=None)

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

    def build_input(self, io_handler: T_IOHandler):
        """

        :param io_handler:
        :return:
        """
        model_spec = self.model_spec

        if isinstance(io_handler, ModelSysConfHandler):

            dist_type = io_handler.get_dist_type()
            sys_conf_set = []
            for _ in range(self.target_num_walkers):
                sys_conf = model_spec.init_get_sys_conf(dist_type=dist_type)
                sys_conf_set.append(sys_conf)

            sys_conf_set = np.asarray(sys_conf_set)
            state = self.sampling.build_state(sys_conf_set)
            return ProcInput(state)

        elif isinstance(io_handler, HDF5FileHandler):

            proc_result = io_handler.load()
            input_state = proc_result.state
            # input_proc = proc_result.proc

            # TODO: We need a reasonable check for the input_state.
            # assert self == input_proc

            return ProcInput(input_state)

        else:
            raise TypeError

    def build_result(self, state: dmc_base.State,
                     sampling: dmc.EstSampling,
                     data: SamplingData):
        """

        :param state:
        :param sampling:
        :param data:
        :return:
        """
        proc = self
        return ProcResult(state, proc, data)


dmc_proc_validator = attr.validators.instance_of(Proc)
opt_dmc_proc_validator = attr.validators.optional(dmc_proc_validator)


def get_io_handler(config: t.Mapping):
    """

    :param config:
    :return:
    """
    handler_spec = dict(config)
    handler_type = handler_spec['type']

    if handler_type == 'MODEL_SYS_CONF':
        return ModelSysConfHandler(**handler_spec)

    elif handler_type == 'HDF5_FILE':
        return HDF5FileHandler(**handler_spec)

    else:
        raise TypeError(f"unknown handler type {handler_type}")


@attr.s(auto_attribs=True)
class ProcSpec:
    """"""

    #: Procedure spec.
    proc: Proc

    #: Input spec.
    input: T_IOHandler

    #: Output spec.
    output: T_IOHandler = None

    #: Procedure id.
    proc_id: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    #: Tag the input.
    tag_input: bool = attr.ib(default=False, validator=bool_validator)

    #: Tag the output.
    tag_output: bool = attr.ib(default=True, validator=bool_validator)

    def __attrs_post_init__(self):
        """"""
        # Tag the input and output handlers.
        if self.tag_input:
            input_ = self.tag_io_handler(self.input)
            object.__setattr__(self, 'input', input_)

        if self.output is not None:
            if self.tag_output:
                output = self.tag_io_handler(self.output)
                object.__setattr__(self, 'output', output)

        # Reset attributes to None.
        object.__setattr__(self, 'tag_input', False)
        object.__setattr__(self, 'tag_output', False)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        proc_config = config['proc']
        proc = Proc.from_config(proc_config)

        # Procedure id...
        proc_id = config.get('proc_id', 0)
        tag_output = config.get('tag_output', True)

        input_handler_config = config['input']
        input_handler = get_io_handler(input_handler_config)

        # Extract the output spec.
        output_handler_config = config['output']
        output_handler = get_io_handler(output_handler_config)

        if not isinstance(output_handler, HDF5FileHandler):
            raise TypeError('only the HDF5_FILE is supported as '
                            'output handler')

        return cls(proc=proc, input=input_handler,
                   output=output_handler, proc_id=proc_id,
                   tag_output=tag_output)

    def tag_io_handler(self, io_handler: T_IOHandler):
        """

        :param io_handler:
        :return:
        """
        proc_id = self.proc_id
        if isinstance(io_handler, HDF5FileHandler):
            spec_group = io_handler.group
            group_suffix = 'proc-ID' + str(proc_id)
            spec_group = '_'.join([spec_group, group_suffix])
            return attr.evolve(io_handler, group=spec_group)

    def evolve(self, config: t.Mapping):
        """

        :param config:
        :return:
        """
        evolve_config = dict(config)

        proc_config = evolve_config.pop('proc')
        proc = self.proc.evolve(proc_config)

        # IO configuration is never evolved directly.
        input_config = evolve_config.pop('input', None)
        if input_config is None:
            input_handler = attr.evolve(self.input)
        else:
            input_handler = get_io_handler(input_config)

        # Extract the output spec.
        output_config = evolve_config.pop('output', None)
        if output_config is None:
            output_handler = attr.evolve(self.output)
        else:
            output_handler = get_io_handler(output_config)

        return attr.evolve(self, proc=proc, input=input_handler,
                           output=output_handler, **evolve_config)

    def exec(self):
        """

        :return:
        """
        proc_input = self.proc.build_input(self.input)
        return self.proc.exec(proc_input)


proc_spec_validator = attr.validators.instance_of(ProcSpec)


def proc_cli_tags_converter(tag_or_tags: t.Union[str, t.Sequence[str]]):
    """

    :param tag_or_tags:
    :return:
    """
    if isinstance(tag_or_tags, str):
        return tag_or_tags

    hashed_tags = ['#' + str(tag) for tag in tag_or_tags]
    return ' - '.join(hashed_tags)


@attr.s(auto_attribs=True)
class ProcCLI:
    """Entry point for the CLI."""

    #:
    name: str = attr.ib(validator=str_validator)

    #:
    description: str = attr.ib(validator=str_validator)

    #:
    author: str = attr.ib(validator=str_validator)

    #:
    author_email: str = attr.ib(validator=str_validator)

    #:
    institution: str = attr.ib(validator=str_validator)

    #:
    category: str = attr.ib(validator=str_validator)

    #:
    tags: str = attr.ib(converter=proc_cli_tags_converter,
                        validator=str_validator)

    #:
    main: ProcSpec = attr.ib(validator=proc_spec_validator)

    #:
    main_post_exec: t.Optional[t.Sequence[ProcSpec]] = None

    #:
    process_proc_id: bool = attr.ib(default=True, validator=bool_validator)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """Initializes a ProcCLI instance from a mapping object.

        :param config:
        :return:
        """
        self_config = dict(config.items())

        # Get the main config.
        main_config = dict(self_config.pop('main'))
        tag_input = main_config.pop('tag_input', None)
        tag_output = main_config.pop('tag_output', None)
        main_proc_id = main_config.pop('proc_id', None)

        # These attributes override any other specified in
        # main_post_exec_config.
        tag_input = False if tag_input is None else tag_input
        tag_output = True if tag_output is None else tag_output
        main_proc_id = 0 if main_proc_id is None else main_proc_id

        # The reference spec should not be tagged.
        main_config['tag_input'] = False
        main_config['tag_output'] = False
        main_config['proc_id'] = main_proc_id

        # The reference procedure spec.
        proc_spec_ref = ProcSpec.from_config(main_config)

        # The main procedure spec.
        proc_spec = attr.evolve(proc_spec_ref, tag_input=tag_input,
                                tag_output=tag_output)

        main_post_exec_config = self_config.pop('main_post_exec', None)
        if main_post_exec_config is not None:
            #
            main_post_proc_set = []

            for post_id, post_config in enumerate(main_post_exec_config, 1):

                post_proc_id = main_proc_id + post_id

                # Extended config.
                ext_config = {'tag_input': tag_input,
                              'tag_output': tag_output,
                              'proc_id': post_proc_id}
                ext_config.update(post_config)

                # Update the reference spec to reflect the current spec.
                post_proc = proc_spec_ref.evolve(ext_config)

                # Append...
                main_post_proc_set.append(post_proc)

        else:

            main_post_proc_set = None

        proc_cli = cls(main=proc_spec,
                       main_post_exec=main_post_proc_set,
                       **self_config)

        return proc_cli

    def exec(self):
        """

        :return:
        """
        main_post_exec = self.main_post_exec or []
        len_self = 1 + len(main_post_exec)

        exec_logger.info(f'Starting the QMC calculations...')
        exec_logger.info(f'Starting the execution of a set of '
                         f'{len_self} QMC calculations...')

        main_proc = self.main
        proc_proc_id = main_proc.proc_id

        exec_logger.info("*** *** ->> ")
        exec_logger.info(f'Starting procedure ID{proc_proc_id}...')

        result = main_proc.exec()

        # Create the necessary directories.
        output_dir = \
            os.path.abspath(os.path.dirname(main_proc.output.location))
        os.makedirs(output_dir, exist_ok=True)

        main_proc.output.save(result)

        exec_logger.info(f'Procedure ID{proc_proc_id} completed.')
        exec_logger.info("<<- *** ***")

        for proc_id, proc in enumerate(main_post_exec, 1):

            exec_logger.info("*** *** ->> ")
            exec_logger.info(f'Starting procedure ID{proc_id}...')

            # Create the necessary directories.
            output_dir = \
                os.path.abspath(os.path.dirname(proc.output.location))
            os.makedirs(output_dir, exist_ok=True)

            result = proc.exec()
            proc.output.save(result)

            exec_logger.info(f'Procedure ID{proc_id} completed.')
            exec_logger.info("<<- *** ***")

        exec_logger.info(f'All the QMC calculations have completed.')
