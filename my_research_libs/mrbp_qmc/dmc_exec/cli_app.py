import typing as t
from pathlib import Path

import attr

from my_research_libs.qmc_exec import dmc as dmc_exec, exec_logger
from my_research_libs.util.attr import (
    opt_int_validator, opt_path_validator, seq_validator, str_validator
)
from .io import (
    HDF5FileHandler, ModelSysConfHandler, T_IOHandler, get_io_handler,
    io_handler_validator
)
from .proc import Proc, ProcResult

proc_validator = attr.validators.instance_of(Proc)
opt_proc_validator = attr.validators.optional(proc_validator)


# TODO: We need a better name for this class.
@attr.s(auto_attribs=True)
class AppSpec:
    """"""

    #: Procedure spec.
    proc: Proc = attr.ib(validator=proc_validator)

    #: Input spec.
    proc_input: T_IOHandler = attr.ib(validator=io_handler_validator)

    #: Output spec.
    # TODO: Update the accepted output handlers.
    proc_output: T_IOHandler = attr.ib(validator=io_handler_validator)

    #: Procedure id.
    proc_id: t.Optional[int] = \
        attr.ib(default=None, validator=opt_int_validator)

    def __attrs_post_init__(self):
        """"""
        # Tag the input and output handlers.
        pass

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        # Own copy(shallow).
        self_config = dict(config)

        proc_config = self_config['proc']
        proc = Proc.from_config(proc_config)

        # Procedure id...
        proc_id = self_config.get('proc_id', 0)

        # Aliases for proc_input and proc_output.
        # TODO: Deprecate these fields.
        if 'input' in self_config:
            proc_input = self_config.pop('input')
            self_config['proc_input'] = proc_input

        if 'output' in self_config:
            proc_output = self_config.pop('output')
            self_config['proc_output'] = proc_output

        input_handler_config = self_config['proc_input']
        input_handler = get_io_handler(input_handler_config)

        # Extract the output spec.
        output_handler_config = self_config['proc_output']
        output_handler = get_io_handler(output_handler_config)

        if not isinstance(output_handler, HDF5FileHandler):
            raise TypeError('only the HDF5_FILE is supported as '
                            'output handler')

        return cls(proc=proc, proc_input=input_handler,
                   proc_output=output_handler, proc_id=proc_id)

    def build_input(self):
        """

        :return:
        """
        input_handler = self.proc_input

        if isinstance(input_handler, ModelSysConfHandler):
            sys_conf_dist_type = input_handler.dist_type_enum
            num_sys_conf = input_handler.num_sys_conf
            return self.proc.input_from_model(sys_conf_dist_type,
                                              num_sys_conf)

        if isinstance(input_handler, HDF5FileHandler):
            proc_result = input_handler.load()
            return self.proc.input_from_result(proc_result)

        raise TypeError

    def dump_output(self, proc_result: ProcResult):
        """

        :param proc_result:
        :return:
        """
        self.proc_output.dump(proc_result)

    def exec(self, proc_input: dmc_exec.ProcInput) -> ProcResult:
        """

        :return:
        """
        return self.proc.exec(proc_input)


def proc_cli_tags_converter(tag_or_tags: t.Union[str, t.Sequence[str]]):
    """

    :param tag_or_tags:
    :return:
    """
    if isinstance(tag_or_tags, str):
        return tag_or_tags

    hashed_tags = ['#' + str(tag) for tag in tag_or_tags]
    return ' - '.join(hashed_tags)


proc_spec_validator = attr.validators.instance_of(AppSpec)


@attr.s(auto_attribs=True)
class CLIApp:
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
    main_proc_set: t.Sequence[AppSpec] = attr.ib(validator=seq_validator)

    #:
    base_path: t.Optional[Path] = \
        attr.ib(default=None, validator=opt_path_validator)

    def __attrs_post_init__(self):
        """"""
        if self.base_path is None:
            base_path = Path('.')
            object.__setattr__(self, 'base_path', base_path)

    @classmethod
    def from_config(cls, config: t.Mapping):
        """Initializes a CLIApp instance from a mapping object.

        :param config:
        :return:
        """
        self_config = dict(config.items())

        # Get the main config.
        main_proc_set = self_config.pop('main_proc_set')

        main_proc_spec_set = []
        for proc_num, proc_config in enumerate(main_proc_set):

            proc_id = proc_config.get('proc_id', None)
            proc_id = proc_num if proc_id is None else proc_id

            # The reference spec should not be tagged.
            proc_config = dict(proc_config)
            proc_config['proc_id'] = proc_id

            proc_spec = AppSpec.from_config(proc_config)

            # Append...
            main_proc_spec_set.append(proc_spec)

        base_path = self_config.pop('base_path', None)
        if base_path is not None:
            base_path = Path(base_path)

        return cls(main_proc_set=main_proc_spec_set,
                   base_path=base_path, **self_config)

    def exec(self):
        """

        :return:
        """
        main_proc_set = self.main_proc_set
        len_self = len(main_proc_set)

        exec_logger.info(f'Starting the QMC calculations...')
        exec_logger.info(f'Starting the execution of a set of '
                         f'{len_self} QMC calculations...')

        for proc_id, proc_spec in enumerate(main_proc_set, 1):

            exec_logger.info("*** *** ->> ")
            exec_logger.info(f'Starting procedure ID{proc_id}...')

            proc_input = proc_spec.build_input()
            result = proc_spec.exec(proc_input)
            proc_spec.dump_output(result)

            exec_logger.info(f'Procedure ID{proc_id} completed.')
            exec_logger.info("<<- *** ***")

        exec_logger.info(f'All the QMC calculations have completed.')
