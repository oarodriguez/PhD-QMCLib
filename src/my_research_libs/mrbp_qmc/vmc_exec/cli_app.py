import typing as t

import attr

from my_research_libs.qmc_exec import cli_app
from my_research_libs.util.attr import (
    opt_int_validator, seq_validator
)
from .io import HDF5FileHandler
from .proc import (
    MODEL_SYS_CONF_TYPE, ModelSysConfSpec, Proc, ProcInput, ProcResult
)

proc_validator = attr.validators.instance_of(Proc)
opt_proc_validator = attr.validators.optional(proc_validator)

# Helpers for the AppSpec.proc_input handling.
T_ProcInput = t.Union[HDF5FileHandler, ModelSysConfSpec]
proc_input_types = (ModelSysConfSpec, HDF5FileHandler)
# noinspection PyTypeChecker
proc_input_validator = \
    attr.validators.instance_of(proc_input_types)

# Helpers for the AppSpec.proc_output handling.
T_ProcOutput = HDF5FileHandler
# NOTE: Using base class or specific classes...
proc_output_types = (HDF5FileHandler,)
proc_output_validator = \
    attr.validators.instance_of(proc_output_types)


# TODO: We need a better name for this class.
@attr.s(auto_attribs=True)
class AppSpec(cli_app.AppSpec):
    """"""

    #: Procedure spec.
    proc: Proc = attr.ib(validator=proc_validator)

    #: Input spec.
    proc_input: T_ProcInput = \
        attr.ib(validator=proc_input_validator)

    #: Output spec.
    # TODO: Update the accepted output handlers.
    proc_output: T_ProcOutput = \
        attr.ib(validator=proc_output_validator)

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
        proc_input = self.proc_input
        if isinstance(proc_input, ModelSysConfSpec):
            return ProcInput.from_model_sys_conf_spec(proc_input, self.proc)
        if isinstance(proc_input, HDF5FileHandler):
            proc_result = proc_input.load()
            return ProcInput.from_result(proc_result, self.proc)
        else:
            raise TypeError

    def exec(self, dump_output: bool = True) -> ProcResult:
        """

        :return:
        """
        self_input = self.build_input()
        proc_result = self.proc.exec(self_input)
        if dump_output:
            self.proc_output.dump(proc_result)
        return proc_result


def get_io_handler(config: t.Mapping):
    """

    :param config:
    :return:
    """
    handler_config = dict(config)
    handler_type = handler_config['type']

    if handler_type == MODEL_SYS_CONF_TYPE:
        return ModelSysConfSpec(**handler_config)

    elif handler_type == 'HDF5_FILE':
        return HDF5FileHandler.from_config(handler_config)

    else:
        raise TypeError(f"unknown handler type {handler_type}")


@attr.s(auto_attribs=True)
class CLIApp(cli_app.CLIApp):
    """Entry point for the CLI."""

    #: Metadata.
    meta: cli_app.AppMeta

    #:
    app_spec: t.Sequence[AppSpec] = attr.ib(validator=seq_validator)

    def __attrs_post_init__(self):
        """"""
        pass

    @classmethod
    def from_config(cls, config: t.Mapping):
        """Initializes a CLIApp instance from a mapping object.

        :param config:
        :return:
        """
        self_config = dict(config.items())

        app_meta = cli_app.AppMeta(**self_config['meta'])
        app_spec_data = self_config.pop('app_spec')

        app_spec_set = []
        for proc_num, app_spec_config in enumerate(app_spec_data):

            proc_id = app_spec_config.get('proc_id', None)
            proc_id = proc_num if proc_id is None else proc_id

            app_spec_config = dict(app_spec_config)
            app_spec_config['proc_id'] = proc_id
            app_spec = AppSpec.from_config(app_spec_config)
            app_spec_set.append(app_spec)

        return cls(meta=app_meta, app_spec=app_spec_set)
