import pathlib
import typing as t

import attr
import h5py

from my_research_libs.qmc_exec import dmc as dmc_exec
from my_research_libs.util.attr import (
    bool_validator, opt_str_validator, str_validator
)
from .proc import Proc, ProcResult

IO_HANDLER_TYPES = ('HDF5_FILE',)
IO_FILE_HANDLER_TYPES = ('HDF5_FILE',)


@attr.s(auto_attribs=True, frozen=True)
class RawHDF5FileHandler(dmc_exec.io.RawHDF5FileHandler):
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
loc_validator = attr.validators.instance_of(valid_path_types)


@attr.s(auto_attribs=True, frozen=True)
class HDF5FileHandler(dmc_exec.io.HDF5FileHandler):
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

    def load(self):
        """Load the contents of the file.

        :return:
        """
        h5_file = h5py.File(self.location_path, 'r')
        with h5_file:
            #
            dmc_group = h5_file.get(f'{self.group}/dmc')
            state_group = dmc_group.get('state')
            proc_group = dmc_group.get('proc_spec')
            data_group = dmc_group.get('data')

            state = self.load_state(state_group)
            proc = self.load_proc(proc_group)
            blocks = dmc_exec.data.SamplingData.from_hdf5_data(data_group)

        data_series = None  # For now...
        sampling_data = dmc_exec.data.SamplingData(blocks, series=data_series)
        return ProcResult(state, proc, sampling_data)

    def load_proc(self, group: h5py.Group):
        """Load the procedure results from the file.

        :param group:
        :return:
        """
        model_spec_group = group.get('model_spec')
        model_spec_config = dict(model_spec_group.attrs.items())

        density_spec_group: h5py.Group = group.get('density_spec')
        if density_spec_group is not None:
            density_spec_config = dict(density_spec_group.attrs.items())
        else:
            density_spec_config = None

        ssf_spec_group: h5py.Group = group.get('ssf_spec')
        if ssf_spec_group is not None:
            ssf_spec_config = dict(ssf_spec_group.attrs.items())
        else:
            ssf_spec_config = None

        # Build a config object.
        proc_config = {
            'model_spec': model_spec_config,
            'density_spec': density_spec_config,
            'ssf_spec': ssf_spec_config
        }
        proc_config.update(group.attrs.items())

        return Proc.from_config(proc_config)
