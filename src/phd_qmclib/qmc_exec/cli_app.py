import typing as t
from abc import abstractmethod

import attr

from phd_qmclib.util.attr import str_validator
from .logging import exec_logger
from .proc import Proc


# TODO: We need a better name for this class.
class AppSpec:
    """Spec for an QMC application."""
    #: Procedure spec.
    proc: Proc

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        """"""
        pass

    @abstractmethod
    def build_input(self):
        """"""
        pass

    @abstractmethod
    def exec(self, dump_output: bool = True):
        """"""
        pass


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
class AppMeta:
    """Metadata of the application."""

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


class CLIApp:
    """Entry point for the CLI."""

    #: Metadata.
    meta: AppMeta

    #:
    app_spec: t.Sequence[AppSpec]

    def __attrs_post_init__(self):
        """"""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping) -> 'CLIApp':
        """Initializes a CLIApp instance from a mapping object.

        :param config:
        :return:
        """
        pass

    def exec(self):
        """Execute the application tasks.

        :return:
        """
        app_spec_set = self.app_spec
        len_spec_set = len(app_spec_set)

        exec_logger.info(f'Starting the QMC calculations...')
        exec_logger.info(f'Starting the execution of a set of '
                         f'{len_spec_set} QMC calculations...')

        for proc_num, app_spec in enumerate(app_spec_set, 1):

            exec_logger.info("*** *** ->> ")
            exec_logger.info(f'Starting procedure ID{proc_num}...')

            result = app_spec.exec()

            exec_logger.info(f'Procedure ID{proc_num} completed.')
            exec_logger.info("<<- *** ***")

        exec_logger.info(f'All the QMC calculations have completed.')
