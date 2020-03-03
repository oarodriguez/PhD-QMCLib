from abc import ABCMeta
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import vmc as vmc_base_udf
from my_research_libs.util.attr import str_validator
from .. import io as io_base
from ..data.vmc import SamplingData


class HDF5FileHandler(io_base.HDF5FileHandler, metaclass=ABCMeta):
    """A handler for properly structured HDF5 files."""

    location: str

    #: Path to the file.
    group: str

    #: The HDF5 group in the file to read and/or write data.
    dump_replace: bool

    #: Replace any existing data in the file.
    type: str

    #: A tag to identify this handler.
    @property
    def location_path(self):
        """Return the file location as a ``pathlib.Path`` object."""
        return Path(self.location).absolute()

    @property
    def sampling_type(self):
        return 'vmc'

    def save_state(self, state: vmc_base_udf.State,
                   group: h5py.Group):
        """

        :param state:
        :param group:
        :return:
        """
        group.create_dataset('sys_conf', data=state.sys_conf)
        group.attrs.update({
            'wf_abs_log': state.wf_abs_log,
            'move_stat': state.move_stat
        })

    def load_state(self, group: h5py.Group):
        """

        :param group:
        :return:
        """
        sys_conf = group.get('sys_conf').value
        return vmc_base_udf.State(sys_conf=sys_conf, **group.attrs)

    def load_sampling_data(self, group: h5py.Group):
        """

        :param group:
        :return:
        """
        return SamplingData.from_hdf5_data(group)

    def save_sampling_data(self, sampling_data: SamplingData,
                           group: h5py.Group):
        """

        :param sampling_data:
        :param group:
        :return:
        """
        sampling_data.hdf5_export(group)


@attr.s(auto_attribs=True, frozen=True)
class NpyFileHandler(io_base.IOHandler, metaclass=ABCMeta):
    """"""
    # NOTE: It could be useful in the future...

    location: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str


@attr.s(auto_attribs=True, frozen=True)
class RawHDF5FileHandler(io_base.IOHandler, metaclass=ABCMeta):
    """A handler for HDF5 files without a specific structure."""

    location: str = attr.ib(validator=str_validator)

    group: str = attr.ib(validator=str_validator)

    dataset: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str
