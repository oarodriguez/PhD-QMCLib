import os
import pathlib
from collections import Mapping
from math import pi

import attr
import jinja2
import yaml

UNIX_NEWLINE = '\n'


@attr.s(auto_attribs=True, frozen=True)
class Variables:
    """"""
    #:
    LKP: float = 1.

    #:
    UE: float = 1.

    #:
    ER: float = UE * pi ** 2

    #:
    K_OPT: float = pi / LKP


@attr.s(auto_attribs=True, frozen=True)
class Template:
    """"""

    #:
    path: pathlib.Path

    def __attrs_post_init__(self):
        """"""
        path = pathlib.Path(self.path)
        object.__setattr__(self, 'path', path)

    @property
    def name(self):
        """"""
        return self.path.name

    @property
    def dirname(self):
        """"""
        return self.path.parent

    @property
    def vars(self):
        """"""
        return Variables()

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
        return jinja2.Environment(loader=self_loader)

    def render(self, context: Mapping):
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
        context = attr.asdict(self.vars)
        context.update({
            'template_name': self.path.stem,
            'config_filename': config_path.stem
        })

        config_file = \
            config_path.open('w', encoding='utf-8', newline=UNIX_NEWLINE)

        # Save the configuration data.
        # NOTE: We could use config_file.write instead...
        config = yaml.safe_load(self.render(context))
        with config_file:
            yaml.safe_dump(config, stream=config_file,
                           default_flow_style=False,
                           indent=4, allow_unicode=True)
