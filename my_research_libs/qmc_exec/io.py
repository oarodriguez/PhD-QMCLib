import typing as t
from abc import ABCMeta, abstractmethod
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import dmc as dmc_base, vmc as vmc_base_udf
from my_research_libs.util.attr import str_validator
from . import data, proc

T_State = t.Union[dmc_base.State, vmc_base_udf.State]
T_SamplingData = t.Union[data.vmc.SamplingData, data.dmc.SamplingData]


class IOHandler(metaclass=ABCMeta):
    """Abstract Base Input-Output handler."""

    #: A tag to identify this handler.
    type: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: t.Mapping):
        """

        :param config:
        :return:
        """
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def dump(self, proc_result: proc.ProcResult):
        """

        :param proc_result:
        :return:
        """
        pass


class HDF5FileHandlerGroupError(ValueError):
    """Flags an error occurring when saving data to an HDF5 file."""
    pass


class HDF5FileHandler(IOHandler, metaclass=ABCMeta):
    """A handler for properly structured HDF5 files."""

    #: Path to the file.
    location: str

    #: The HDF5 group in the file to read and/or write data.
    group: str

    #: Replace any existing data in the file.
    dump_replace: bool

    #: A tag to identify this handler.
    type: str

    @property
    def location_path(self):
        """Return the file location as a ``pathlib.Path`` object."""
        return Path(self.location).absolute()

    @property
    @abstractmethod
    def sampling_type(self):
        pass

    def load(self):
        """Load the contents of the file.

        :return:
        """
        h5_file = h5py.File(self.location_path, 'r')
        with h5_file:
            #
            qmc_group = h5_file.get(f'{self.group}/{self.sampling_type}')
            state_group = qmc_group.get('state')
            proc_group = qmc_group.get('proc_spec')
            data_group = qmc_group.get('data')

            state = self.load_state(state_group)
            proc_inst = self.load_proc(proc_group)
            sampling_data = self.load_sampling_data(data_group)

        return self.build_result(state, proc_inst, sampling_data)

    def dump(self, proc_result: proc.ProcResult):
        """Save a QMC procedure result to file.

        :param proc_result:
        :return:
        """
        h5_file = h5py.File(self.location_path)
        with h5_file:
            #
            base_group = h5_file.require_group(self.group)
            sampling_type = self.sampling_type
            if sampling_type in base_group:
                if self.dump_replace:
                    del base_group[sampling_type]
                else:
                    group_exists_msg = (
                        f"Unable to create '{sampling_type}' group (name "
                        f"already exists)"
                    )
                    raise HDF5FileHandlerGroupError(group_exists_msg)

            # Create the required groups.
            qmc_group = base_group.require_group(sampling_type)
            state_group = qmc_group.require_group('state')
            proc_group = qmc_group.require_group('proc_spec')
            data_group = qmc_group.require_group('data')

            state = proc_result.state
            proc_config = proc_result.proc.as_config()
            sampling_data = proc_result.data

            # Export the data.
            self.save_state(state, state_group)
            self.save_proc(proc_config, proc_group)
            self.save_sampling_data(sampling_data, data_group)

            # Do not for get to flush.
            h5_file.flush()

    @abstractmethod
    def build_result(self, state: T_State,
                     proc_inst: proc.ProcResult,
                     sampling_data: T_SamplingData) -> proc.ProcResult:
        """

        :param state:
        :param proc_inst:
        :param sampling_data:
        :return:
        """
        pass

    @abstractmethod
    def load_state(self, group: h5py.Group):
        pass

    @abstractmethod
    def save_state(self, state: T_State, group: h5py.Group):
        pass

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
        return self.build_proc(proc_config)

    @staticmethod
    def save_proc(config: t.Dict, group: h5py.Group):
        """

        :param config:
        :param group:
        :return:
        """
        model_spec = config.pop('model_spec')

        model_spec_group = group.require_group('model_spec')
        model_spec_group.attrs.update(**model_spec)

        density_spec_config = config.pop('density_spec', None)
        if density_spec_config is not None:
            density_spec_group = group.require_group('density_spec')
            density_spec_group.attrs.update(**density_spec_config)

        ssf_spec_config = config.pop('ssf_spec', None)
        if ssf_spec_config is not None:
            ssf_spec_group = group.require_group('ssf_spec')
            ssf_spec_group.attrs.update(**ssf_spec_config)

        group.attrs.update(config)

    @abstractmethod
    def build_proc(self, proc_config: t.Dict):
        pass

    @abstractmethod
    def load_sampling_data(self, group: h5py.Group):
        """"""
        pass

    @abstractmethod
    def save_sampling_data(self, sampling_data: T_SamplingData,
                           group: h5py.Group):
        """"""
        pass


@attr.s(auto_attribs=True, frozen=True)
class NpyFileHandler(IOHandler, metaclass=ABCMeta):
    """"""
    # NOTE: It could be useful in the future...

    location: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str


@attr.s(auto_attribs=True, frozen=True)
class RawHDF5FileHandler(IOHandler, metaclass=ABCMeta):
    """A handler for HDF5 files without a specific structure."""

    location: str = attr.ib(validator=str_validator)

    group: str = attr.ib(validator=str_validator)

    dataset: str = attr.ib(validator=str_validator)

    #: A tag to identify this handler.
    type: str
