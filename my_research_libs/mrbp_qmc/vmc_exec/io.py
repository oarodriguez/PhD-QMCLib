import pathlib
import typing as t

import attr

from my_research_libs.qmc_exec import vmc as vmc_exec
from my_research_libs.util.attr import (
    bool_validator, opt_str_validator, str_validator
)
from .proc import Proc, ProcResult

IO_HANDLER_TYPES = ('HDF5_FILE',)
IO_FILE_HANDLER_TYPES = ('HDF5_FILE',)


@attr.s(auto_attribs=True, frozen=True)
class RawHDF5FileHandler(vmc_exec.io.RawHDF5FileHandler):
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

    def dump(self, data: 'ProcResult'):
        pass


# A ``pathlib.Path`` instance is a valid location.
valid_path_types = pathlib.Path, str
# noinspection PyTypeChecker
loc_validator = attr.validators.instance_of(valid_path_types)


@attr.s(auto_attribs=True, frozen=True)
class HDF5FileHandler(vmc_exec.io.HDF5FileHandler):
    """A handler for structured HDF5 files to save DMC procedure results."""

    #: Path to the file.
    location: str = attr.ib(validator=loc_validator)

    #: The HDF5 group in the file to read and/or write data.
    group: str = attr.ib(validator=str_validator)

    #: Replace any existing data in the file.
    dump_replace: bool = attr.ib(default=False, validator=bool_validator)

    #: A tag to identify this handler.
    type: t.Optional[str] = attr.ib(default=None, validator=opt_str_validator)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # This is the type tag, and must be fixed.
        object.__setattr__(self, 'type', 'HDF5_FILE')

        location = self.location
        if isinstance(location, pathlib.Path):
            object.__setattr__(self, 'location', str(location))

        location_path = self.location_path
        if location_path.is_dir():
            raise ValueError(f"location {location_path} is a directory, "
                             f"not a file")

    @classmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        return cls(**config)

    def build_proc(self, proc_config: t.Dict):
        """

        :param proc_config:
        :return:
        """
        return Proc.from_config(proc_config)
