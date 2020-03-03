import os
import pathlib
import typing as t
from abc import abstractmethod
from collections import Sequence

import jinja2
import toml
from ruamel_yaml import YAML

CONFIG_FILE_EXTENSIONS = ('.yml', '.yaml', '.toml')
YAML_EXTENSIONS = ('.yml', '.yaml')

UNIX_NEWLINE = '\n'

# yaml object definition.
yaml = YAML()
yaml.allow_unicode = True
yaml.indent = 4
yaml.default_flow_style = False


class Loader:
    """Load the configuration for a QMC procedure."""

    #: Valid extensions for configuration files.
    file_extensions: t.Tuple[str, ...]

    #: Valid input-output handler types.
    io_file_handler_types: t.Tuple[str, ...]

    def load(self, location: t.Union[str, pathlib.Path]):
        """

        :param location:
        :return:
        """
        path = pathlib.Path(location)
        suffix = path.suffix
        if not suffix:
            raise IOError('config file has no extension')

        if suffix not in self.file_extensions:
            raise IOError('unknown file extension')

        # NOTE: Should we base this choice in file extensions.¿
        if suffix in YAML_EXTENSIONS:
            assume_yaml = True
        else:
            assume_yaml = False

        with path.open('r') as fp:
            if assume_yaml:
                config_data = yaml.load(fp)
            else:
                config_data = toml.load(fp)

        # Keep support for old config files.
        if 'main_proc_set' in config_data:
            config_data['app_spec'] = config_data.pop('main_proc_set')

        app_spec_data = config_data['app_spec']
        if isinstance(app_spec_data, Sequence):
            app_spec_config_set = []
            for app_spec_config in app_spec_data:
                app_spec_config_set.append(app_spec_config)
        else:
            app_spec_config_set = [app_spec_data]

        path = path.absolute()
        loc_parent = path.parent
        for app_spec_conf in app_spec_config_set:
            self.fix_app_spec_locations(app_spec_conf, loc_parent)

        config_data['app_spec'] = app_spec_config_set
        return config_data

    def fix_app_spec_locations(self, app_spec_config: t.MutableMapping,
                               config_path: pathlib.Path):
        """Fix any relative path in the AppSpec configuration.

        Relative paths are relative to the location of the configuration
        file.

        :param app_spec_config: The configuration of the application.
        :param config_path: The location of the configuration file.
        :return:
        """
        # Aliases for proc_input and proc_output.
        # TODO: Deprecate these fields.
        if 'input' in app_spec_config:
            app_spec_config['proc_input'] = app_spec_config.pop('input')
        if 'output' in app_spec_config:
            app_spec_config['proc_output'] = app_spec_config.pop('output')

        proc_input = app_spec_config['proc_input']
        handler_type = proc_input['type']
        if handler_type in self.io_file_handler_types:
            # If input_location is absolute, base_path is discarded
            # automatically.
            input_location = proc_input['location']
            proc_input['location'] = str(config_path / input_location)

        proc_output = app_spec_config['proc_output']
        handler_type = proc_output['type']
        if handler_type in self.io_file_handler_types:
            output_location = proc_output['location']
            proc_output['location'] = str(config_path / output_location)


class Template:
    """"""

    #:
    path: pathlib.Path

    @property
    def name(self):
        """"""
        return self.path.name

    @property
    def dirname(self):
        """"""
        return self.path.parent

    @property
    @abstractmethod
    def vars(self):
        """"""
        pass

    @property
    def loader(self):
        """"""
        cwd = os.getcwd()
        self_dir = str(self.dirname)
        return jinja2.FileSystemLoader([self_dir, cwd], followlinks=True)

    @property
    def environ(self):
        """"""
        self_loader = self.loader
        return jinja2.Environment(loader=self_loader, trim_blocks=True,
                                  lstrip_blocks=True)

    def render(self, context: t.Mapping):
        """

        :param context:
        :return:
        """
        # self_path = str(self.path)
        template = self.environ.get_template(self.name)
        return template.render(context)

    def save(self, config_path: pathlib.Path):
        """

        :param config_path:
        :return:
        """
        context = dict(self.vars)
        context.update({
            'template_name': self.path.stem,
            'config_filename': config_path.stem
        })

        config_file = \
            config_path.open('w', encoding='utf-8', newline=UNIX_NEWLINE)

        # Save the configuration data.
        # NOTE: We could use config_file.write instead...
        config = yaml.load(self.render(context))
        with config_file:
            yaml.dump(config, stream=config_file)
