import typing as t
from abc import ABCMeta, abstractmethod
from pathlib import Path

import attr
import h5py

from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.util.attr import str_validator
from .data import (
    EnergyBlocks, NumWalkersBlocks, PropBlocks, PropsDataBlocks, SSFBlocks,
    SSFPartBlocks, SamplingData, WeightBlocks
)
from .proc import Proc, ProcResult


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
    location: Path

    #: The HDF5 group in the file to read and/or write data.
    group: str

    #: Replace any existing data in the file.
    dump_replace: bool

    #: A tag to identify this handler.
    type: str

    def init_main_groups(self, h5_file: h5py.File):
        """Initialize sub-groups to store the data.

        :param h5_file:
        :return:
        """
        base_group = h5_file.require_group(self.group)

        if 'dmc' in base_group:
            raise HDF5FileHandlerGroupError("Unable to create 'dmc' group "
                                            "(name already exists)")

        dmc_group = base_group.require_group('dmc')
        dmc_group.require_group('proc_spec')
        dmc_group.require_group('state')

        data_group = dmc_group.require_group('data')
        blocks_group = data_group.require_group('blocks')
        blocks_group.require_group('energy')
        blocks_group.require_group('weight')
        blocks_group.require_group('num_walkers')

    def dump(self, data: 'ProcResult'):
        """Save a DMC procedure result to file.

        :param data:
        :return:
        """
        file_path = self.location.absolute()
        h5_file = h5py.File(file_path)
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

    def save_state(self, state: dmc_base.State,
                   h5_file: h5py.File):
        """

        :param state:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.get(group_name)
        state_group = base_group.require_group('dmc/state')

        state_group.create_dataset('branching_spec', data=state.branching_spec)
        state_group.create_dataset('confs', data=state.confs)
        state_group.create_dataset('props', data=state.props)

        state_group.attrs.update({
            'energy': state.energy,
            'weight': state.weight,
            'num_walkers': state.num_walkers,
            'ref_energy': state.ref_energy,
            'accum_energy': state.accum_energy,
            'max_num_walkers': state.max_num_walkers
        })

    def save_proc(self, proc: 'Proc', h5_file: h5py.File):
        """

        :param proc:
        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        proc_group = base_group.require_group('dmc/proc_spec')

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
        blocks_group = base_group.require_group('dmc/data/blocks')

        data_blocks = data.blocks
        energy_blocks = data_blocks.energy
        energy_group = blocks_group.require_group('energy')
        self.save_prop_blocks(energy_blocks, energy_group)

        weight_blocks = data_blocks.weight
        weight_group = blocks_group.require_group('weight')
        self.save_prop_blocks(weight_blocks, weight_group,
                              has_weight_totals=False)

        num_walkers_blocks = data_blocks.num_walkers
        num_walkers_group = blocks_group.require_group('num_walkers')
        self.save_prop_blocks(num_walkers_blocks, num_walkers_group,
                              has_weight_totals=False)

        ssf_blocks = data_blocks.ss_factor
        if ssf_blocks is not None:
            # Save each part of S(k).
            fdk_sqr_abs_group = \
                blocks_group.require_group('ss_factor/fdk_sqr_abs')

            self.save_prop_blocks(ssf_blocks.fdk_sqr_abs_part,
                                  fdk_sqr_abs_group)

            fdk_real_group = \
                blocks_group.require_group('ss_factor/fdk_real')

            self.save_prop_blocks(ssf_blocks.fdk_real_part, fdk_real_group)

            fdk_imag_group = \
                blocks_group.require_group('ss_factor/fdk_imag')

            self.save_prop_blocks(ssf_blocks.fdk_imag_part, fdk_imag_group)

    @staticmethod
    def save_prop_blocks(blocks: PropBlocks,
                         group: h5py.Group,
                         has_weight_totals: bool = True):
        """

        :param blocks:
        :param group:
        :param has_weight_totals:
        :return:
        """
        group.create_dataset('totals', data=blocks.totals)

        if has_weight_totals:
            group.create_dataset('weight_totals', data=blocks.weight_totals)

        group.attrs.update({
            'num_blocks': blocks.num_blocks,
            'num_time_steps_block': blocks.num_time_steps_block
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

        state_group = base_group.require_group('dmc/state')

        branching_spec = state_group.get('branching_spec').value
        state_confs = state_group.get('confs').value
        state_props = state_group.get('props').value

        return dmc_base.State(confs=state_confs,
                              props=state_props,
                              branching_spec=branching_spec,
                              **state_group.attrs)

    def load_data_blocks(self, h5_file: h5py.File):
        """

        :param h5_file:
        :return:
        """
        group_name = self.group
        base_group = h5_file.require_group(group_name)
        blocks_group = base_group.require_group('dmc/data/blocks')

        energy_group = blocks_group.require_group('energy')
        blocks_data = self.load_prop_blocks_data(energy_group)
        energy_blocks = EnergyBlocks(**blocks_data)

        weight_group = blocks_group.require_group('weight')
        blocks_data = self.load_prop_blocks_data(weight_group,
                                                 has_weight_totals=False)
        weight_blocks = WeightBlocks(**blocks_data)

        num_walkers_group = blocks_group.require_group('num_walkers')
        blocks_data = self.load_prop_blocks_data(num_walkers_group,
                                                 has_weight_totals=False)
        num_walkers_blocks = NumWalkersBlocks(**blocks_data)

        ss_factor_group = blocks_group.get('ss_factor', None)
        if ss_factor_group is not None:
            #
            fdk_sqr_abs_group = \
                blocks_group.require_group('ss_factor/fdk_sqr_abs')
            blocks_data = self.load_prop_blocks_data(fdk_sqr_abs_group)
            fdk_sqr_abs_blocks = SSFPartBlocks(**blocks_data)

            fdk_real_group = \
                blocks_group.require_group('ss_factor/fdk_real')
            blocks_data = self.load_prop_blocks_data(fdk_real_group)
            fdk_real_blocks = SSFPartBlocks(**blocks_data)

            fdk_imag_group = \
                blocks_group.require_group('ss_factor/fdk_imag')
            blocks_data = self.load_prop_blocks_data(fdk_imag_group)
            fdk_imag_blocks = SSFPartBlocks(**blocks_data)

            ssf_blocks = SSFBlocks(fdk_sqr_abs_blocks,
                                   fdk_real_blocks,
                                   fdk_imag_blocks)

        else:
            ssf_blocks = None

        return PropsDataBlocks(energy_blocks,
                               weight_blocks,
                               num_walkers_blocks,
                               ssf_blocks)

    @staticmethod
    def load_prop_blocks_data(group: h5py.Group,
                              has_weight_totals: bool = True):
        """

        :param group:
        :param has_weight_totals:
        :return:
        """
        totals = group.get('totals').value
        prop_blocks_data = {'totals': totals}

        prop_blocks_data.update(group.attrs.items())

        if has_weight_totals:
            weight_totals = group.get('weight_totals').value
            prop_blocks_data.update(weight_totals=weight_totals)

        return prop_blocks_data


class ModelSysConfHandler(IOHandler, metaclass=ABCMeta):
    """"""

    dist_type: str


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
