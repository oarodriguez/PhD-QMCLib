import typing as t
from abc import ABCMeta, abstractmethod
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.util.attr import str_validator
from .proc import ProcResult


class IOHandler(metaclass=ABCMeta):
    """"""

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
    def dump(self, data: 'ProcResult'):
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

    def dump(self, proc_result: 'ProcResult'):
        """Save a DMC procedure result to file.

        :param proc_result:
        :return:
        """
        h5_file = h5py.File(self.location_path)
        with h5_file:
            #
            base_group = h5_file.require_group(self.group)
            if 'dmc' in base_group:
                if self.dump_replace:
                    del base_group['dmc']
                else:
                    group_exists_msg = "Unable to create 'dmc' group " \
                                       "(name already exists)"
                    raise HDF5FileHandlerGroupError(group_exists_msg)

            # Create the required groups.
            dmc_group = base_group.require_group('dmc')
            state_group = dmc_group.require_group('state')
            proc_group = dmc_group.require_group('proc_spec')
            data_group = dmc_group.require_group('data')

            state = proc_result.state
            proc_config = proc_result.proc.as_config()
            proc_result_data = proc_result.data

            # Export the data.
            self.save_state(state, state_group)
            self.save_proc(proc_config, proc_group)
            proc_result_data.hdf5_export(data_group)

            # Do not for get to flush.
            h5_file.flush()

    @staticmethod
    def save_state(state: dmc_base.State,
                   group: h5py.Group):
        """

        :param state:
        :param group:
        :return:
        """
        group.create_dataset('branching_spec', data=state.branching_spec)
        group.create_dataset('confs', data=state.confs)
        group.create_dataset('props', data=state.props)

        group.attrs.update({
            'energy': state.energy,
            'weight': state.weight,
            'num_walkers': state.num_walkers,
            'ref_energy': state.ref_energy,
            'accum_energy': state.accum_energy,
            'max_num_walkers': state.max_num_walkers
        })

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
    def load_proc(self, group: h5py.Group):
        """"""
        pass

    @staticmethod
    def load_state(group: h5py.Group):
        """

        :return:
        """
        branching_spec = group.get('branching_spec').value
        state_confs = group.get('confs').value
        state_props = group.get('props').value

        return dmc_base.State(confs=state_confs,
                              props=state_props,
                              branching_spec=branching_spec,
                              **group.attrs)


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
