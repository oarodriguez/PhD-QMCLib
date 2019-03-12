import typing as t
from abc import ABCMeta, abstractmethod
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import vmc as vmc_base_udf
from my_research_libs.util.attr import str_validator
from .data import (
    EnergyBlocks, PropBlocks, PropsDataBlocks, SSFBlocks,
    SSFPartBlocks, SamplingData
)
from .proc import Proc, ProcResult

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

    def dump(self, data: 'ProcResult'):
        """Save a DMC procedure result to file.

        :param data:
        :return:
        """
        self_path = self.location_path
        h5_file = h5py.File(self_path)
        with h5_file:
            #
            self.init_main_groups(h5_file)

            self.save_proc(data.proc, h5_file)

            self.save_state(data.state, h5_file)

            self.save_data_blocks(data.data, h5_file)

            h5_file.flush()

    @abstractmethod
    def get_proc_config(self, proc: 'Proc'):
        """

        :param proc:
        :return:
        """
        pass

    def save_state(self, state: vmc_base_udf.State,
                   h5_file: h5py.File):
        """

        :param state:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name)
        state_group = base_group.require_group(VMC_STATE)

        state_group.create_dataset('sys_conf', data=state.sys_conf)
        state_group.attrs.update({
            'wf_abs_log': state.wf_abs_log,
            'move_stat': state.move_stat
        })

    def save_proc(self, proc: 'Proc', h5_file: h5py.File):
        """

        :param proc:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        proc_group = base_group.require_group(VMC_PROC_SPEC)

        proc_config = self.get_proc_config(proc)
        model_spec = proc_config.pop('model_spec')

        model_spec_group = proc_group.require_group('model_spec')
        model_spec_group.attrs.update(**model_spec)

        ssf_spec_config = proc_config.pop('ssf_spec', None)
        if ssf_spec_config is not None:
            ssf_spec_group = proc_group.require_group('ssf_spec')
            ssf_spec_group.attrs.update(**ssf_spec_config)

        proc_group.attrs.update(proc_config)

    def save_data_blocks(self, data: SamplingData,
                         h5_file: h5py.File):
        """

        :param data:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name)
        blocks_group = base_group.require_group(VMC_DATA_BLOCKS)

        data_blocks = data.blocks
        energy_blocks = data_blocks.energy
        energy_group = blocks_group.require_group('energy')
        self.save_prop_blocks(energy_blocks, energy_group)

        ssf_blocks = data_blocks.ss_factor
        if ssf_blocks is not None:
            # Save each part of S(k).
            fdk_sqr_abs_group = \
                blocks_group.require_group(SS_FACTOR_FDK_SQR_ABS)

            self.save_prop_blocks(ssf_blocks.fdk_sqr_abs_part,
                                  fdk_sqr_abs_group)

            fdk_real_group = \
                blocks_group.require_group(SS_FACTOR_FDK_REAL)

            self.save_prop_blocks(ssf_blocks.fdk_real_part, fdk_real_group)

            fdk_imag_group = \
                blocks_group.require_group(SS_FACTOR_FDK_IMAG)

            self.save_prop_blocks(ssf_blocks.fdk_imag_part, fdk_imag_group)

    @staticmethod
    def save_prop_blocks(blocks: PropBlocks,
                         group: h5py.Group):
        """

        :param blocks:
        :param group:
        :return:
        """
        group.create_dataset('totals', data=blocks.totals)
        group.attrs.update({
            'num_blocks': blocks.num_blocks,
            'num_steps_block': blocks.num_steps_block
        })

    @abstractmethod
    def load_proc(self, h5_file):
        """"""
        pass

    def load_state(self, h5_file: h5py.File):
        """

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name, None)
        if base_group is None:
            raise HDF5FileHandlerGroupError(
                    f"unable to read '{group_name}' group (name "
                    f"does not exists)"
            )

        state_group = base_group.require_group(VMC_STATE)
        sys_conf = state_group.get('sys_conf').value
        return vmc_base_udf.State(sys_conf=sys_conf, **state_group.attrs)

    def load_data_blocks(self, h5_file: h5py.File):
        """

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        blocks_group = base_group.require_group(VMC_DATA_BLOCKS)
        energy_group = base_group.require_group(VMC_DATA_BLOCKS_ENERGY)

        blocks_data = self.load_prop_blocks_data(energy_group)
        energy_blocks = EnergyBlocks(**blocks_data)

        ss_factor_group = base_group.get(VMC_DATA_BLOCKS_SS_FACTOR, None)
        if ss_factor_group is not None:
            #
            fdk_sqr_abs_group = \
                blocks_group.require_group(SS_FACTOR_FDK_SQR_ABS)
            blocks_data = self.load_prop_blocks_data(fdk_sqr_abs_group)
            fdk_sqr_abs_blocks = SSFPartBlocks(**blocks_data)

            fdk_real_group = \
                blocks_group.require_group(SS_FACTOR_FDK_REAL)
            blocks_data = self.load_prop_blocks_data(fdk_real_group)
            fdk_real_blocks = SSFPartBlocks(**blocks_data)

            fdk_imag_group = \
                blocks_group.require_group(SS_FACTOR_FDK_IMAG)
            blocks_data = self.load_prop_blocks_data(fdk_imag_group)
            fdk_imag_blocks = SSFPartBlocks(**blocks_data)

            ssf_blocks = SSFBlocks(fdk_sqr_abs_blocks,
                                   fdk_real_blocks,
                                   fdk_imag_blocks)

        else:
            ssf_blocks = None

        return PropsDataBlocks(energy_blocks,
                               ssf_blocks)

    @staticmethod
    def load_prop_blocks_data(group: h5py.Group):
        """

        :param group:
        :return:
        """
        totals = group.get('totals').value
        prop_blocks_data = {'totals': totals}
        prop_blocks_data.update(group.attrs.items())
        return prop_blocks_data


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
