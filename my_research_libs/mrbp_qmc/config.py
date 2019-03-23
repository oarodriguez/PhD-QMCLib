import pathlib
import typing as t
from math import pi

import attr

from my_research_libs.qmc_exec import config

CONFIG_FILE_EXTENSIONS = ('.yml', '.yaml', '.toml')
YAML_EXTENSIONS = ('.yml', '.yaml')

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
class Loader(config.Loader):
    """"""
    #:
    file_extensions: t.Tuple[str, ...]

    #:
    io_file_handler_types: t.Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True)
class Template(config.Template):
    """"""

    #:
    path: pathlib.Path

    def __attrs_post_init__(self):
        """"""
        path = pathlib.Path(self.path)
        object.__setattr__(self, 'path', path)

    @property
    def vars(self):
        """"""
        return attr.asdict(Variables())
