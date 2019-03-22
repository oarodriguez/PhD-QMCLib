import typing as t
from abc import ABCMeta, abstractmethod
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import vmc as vmc_base_udf
from my_research_libs.util.attr import str_validator
from .proc import ProcResult

# HDF5 groups.
VMC_BASE_GROUP = 'vmc'
VMC_PROC_SPEC = 'vmc/proc_spec'
VMC_STATE = 'vmc/state'
VMC_DATA = 'vmc/data'
VMC_DATA_BLOCKS = 'vmc/data/blocks'
VMC_DATA_BLOCKS_ENERGY = 'vmc/data/blocks/energy'
VMC_DATA_BLOCKS_SS_FACTOR = 'vmc/data/blocks/ss_factor'
SS_FACTOR_FDK_SQR_ABS = 'ss_factor/fdk_sqr_abs'
SS_FACTOR_FDK_REAL = 'ss_factor/fdk_real'
SS_FACTOR_FDK_IMAG = 'ss_factor/fdk_imag'


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

    def init_main_groups(self, h5_file: h5py.File):
        """Initialize sub-groups to store the data.

        :param h5_file:
        :return:
        """
        base_group = h5_file.require_group(self.group)

        if VMC_BASE_GROUP in base_group:
            raise HDF5FileHandlerGroupError("Unable to create 'vmc' group "
                                            "(name already exists)")

        base_group.require_group(VMC_PROC_SPEC)
        base_group.require_group(VMC_STATE)
        base_group.require_group(VMC_DATA_BLOCKS)
        base_group.require_group(VMC_DATA_BLOCKS_ENERGY)

    def dump(self, proc_result: 'ProcResult'):
        """Save a DMC procedure result to file.

        :param proc_result:
        :return:
        """
        h5_file = h5py.File(self.location_path)
        with h5_file:
            #
            base_group = h5_file.require_group(self.group)
            if 'vmc' in base_group:
                if self.dump_replace:
                    del base_group['vmc']
                else:
                    group_exists_msg = "Unable to create 'vmc' group " \
                                       "(name already exists)"
                    raise HDF5FileHandlerGroupError(group_exists_msg)

            vmc_group = base_group.require_group('vmc')
            state_group = vmc_group.require_group('state')
            proc_group = vmc_group.require_group('proc_spec')
            data_group = vmc_group.require_group('data')

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
    def save_state(state: vmc_base_udf.State,
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

    @staticmethod
    def save_proc(proc_config: t.Dict, group: h5py.Group):
        """

        :param proc_config:
        :param group:
        :return:
        """
        model_spec = proc_config.pop('model_spec')

        model_spec_group = group.require_group('model_spec')
        model_spec_group.attrs.update(**model_spec)

        ssf_spec_config = proc_config.pop('ssf_spec', None)
        if ssf_spec_config is not None:
            ssf_spec_group = group.require_group('ssf_spec')
            ssf_spec_group.attrs.update(**ssf_spec_config)

        group.attrs.update(proc_config)

    @abstractmethod
    def load_proc(self, h5_file):
        """"""
        pass

    @staticmethod
    def load_state(group: h5py.Group):
        """

        :param group:
        :return:
        """
        sys_conf = group.get('sys_conf').value
        return vmc_base_udf.State(sys_conf=sys_conf, **group.attrs)


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
