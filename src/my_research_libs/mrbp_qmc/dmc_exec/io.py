import pathlib
import typing as t

import attr

from my_research_libs.qmc_exec import data as qmc_data, dmc as dmc_exec
from my_research_libs.util.attr import (
    bool_validator, opt_str_validator, str_validator
)
from .proc import Proc, ProcResult
from .. import dmc

IO_HANDLER_TYPES = ('HDF5_FILE',)
IO_FILE_HANDLER_TYPES = ('HDF5_FILE',)

# A ``pathlib.Path`` instance is a valid location.
valid_path_types = pathlib.Path, str
loc_validator = attr.validators.instance_of(valid_path_types)


@attr.s(auto_attribs=True, frozen=True)
class HDF5FileHandler(dmc_exec.HDF5FileHandler):
    """A handler for structured HDF5 files to save DMC procedure results."""

    location: str = attr.ib(validator=loc_validator)

    #: Path to the file.
    group: str = attr.ib(validator=str_validator)

    #: The HDF5 group in the file to read and/or write data.
    dump_replace: bool = attr.ib(default=False, validator=bool_validator)

    #: Replace any existing data in the file.
    type: t.Optional[str] = attr.ib(default=None, validator=opt_str_validator)

    #: A tag to identify this handler.
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

    def build_result(self, state: dmc.State,
                     proc_inst: Proc,
                     sampling_data: qmc_data.dmc.SamplingData):
        """

        :param state:
        :param proc_inst:
        :param sampling_data:
        :return:
        """
        return ProcResult(state, proc_inst, sampling_data)
