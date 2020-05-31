import typing as t
from typing import Mapping

import attr
import h5py
import numpy as np
from cached_property import cached_property

from phd_qmclib.qmc_base import vmc as vmc_udf_base
from phd_qmclib.qmc_base.vmc import SSFPartSlot
from phd_qmclib.stats import reblock


@attr.s(auto_attribs=True, frozen=True)
class PropBlock:
    """Represent a single block of data."""
    total: float


T_PropBlocksItem = t.Union['PropBlock', 'PropBlocks']


@attr.s(auto_attribs=True, frozen=True)
class PropBlocks(Mapping):
    """Abstract class to represent data in blocks."""

    totals: np.ndarray

    @property
    def mean(self):
        """Mean value of the blocks."""
        return self.reblock.mean

    @property
    def mean_error(self):
        """Error of the mean value of the blocks."""
        return self.reblock.mean_eff_error

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFObject.from_non_obj_data(self.totals)

    def hdf5_export(self, group: h5py.Group):
        """Export the data to an HDF5 group object.

        :param group:
        :return:
        """
        # Array data go to a dataset.
        group.create_dataset('totals', data=self.totals)

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """Create an instance from the data to an HDF5 group object.

        :param group:
        :return:
        """
        data = {
            'totals': group.get('totals').value
        }
        return cls(**data)

    def __getitem__(self, index) -> T_PropBlocksItem:
        """Retrieve single blocks, or a whole series."""
        if isinstance(index, tuple):
            if len(index) > 1:
                raise TypeError("only one-element tuples are allowed")

        if isinstance(index, int):
            total = self.totals[index]
            return PropBlock(total)

        totals = self.totals[index]
        # TODO: Retrieve an instance of type(self) for now.
        return PropBlocks(totals)

    def __len__(self) -> int:
        """Number of blocks."""
        return len(self.totals)

    def __iter__(self):
        """Iterable interface."""
        for index, total in enumerate(self.totals):
            total = self.totals[index]
            yield PropBlock(total)

    def __add__(self, other: t.Any):
        """Concatenate the current data blocks with another data blocks."""
        if not isinstance(other, PropBlocks):
            return NotImplemented
        try:
            all_totals = self.totals, other.totals
            totals = np.concatenate(all_totals, axis=0)
        except ValueError as e:
            raise ValueError("'totals' are incompatible "
                             "between instances") from e
        return PropBlocks(totals)


@attr.s(auto_attribs=True, frozen=True)
class EnergyBlocks(PropBlocks):
    """Energy data in blocks."""

    totals: np.ndarray

    @classmethod
    def from_data(cls, data: vmc_udf_base.PropsData,
                  reduce_data: bool = True):
        """

        :param data:
        :param reduce_data:
        :return:
        """
        energy_data = data.energy
        if reduce_data:
            totals = energy_data.mean(axis=1)
        else:
            totals = energy_data
        return cls(totals)


@attr.s(auto_attribs=True, frozen=True)
class DensityBlocks(PropBlocks):
    """Density data in blocks."""

    totals: np.ndarray

    @classmethod
    def from_data(cls, density_data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param density_data:
        :param reduce_data:
        :return:
        """
        if reduce_data:
            totals = density_data.mean(axis=1)
        else:
            totals = density_data
        return cls(totals)


@attr.s(auto_attribs=True, frozen=True)
class SSFPartBlocks(PropBlocks):
    """Structure Factor data in blocks."""

    totals: np.ndarray

    @classmethod
    def from_data(cls, ssf_data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param reduce_data:
        :param ssf_data:
        :return:
        """
        if reduce_data:
            totals = ssf_data.mean(axis=1)
        else:
            totals = ssf_data
        return cls(totals)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFSet.from_non_obj_data(self.totals)


@attr.s(auto_attribs=True, frozen=True)
class SSFBlocks:
    """Structure Factor data in blocks."""

    #: Squared module of the Fourier density component.
    fdk_sqr_abs_part: SSFPartBlocks

    #: Real part of the Fourier density component.
    fdk_real_part: SSFPartBlocks

    #: Imaginary part of the Fourier density component.
    fdk_imag_part: SSFPartBlocks

    @classmethod
    def from_data(cls, ssf_data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param reduce_data:
        :param ssf_data:
        :return:
        """
        if reduce_data:
            totals = ssf_data.mean(axis=1)
        else:
            totals = ssf_data

        # The totals of every part.
        fdk_sqr_abs_totals = totals[:, :, SSFPartSlot.FDK_SQR_ABS]
        fdk_real_totals = totals[:, :, SSFPartSlot.FDK_REAL]
        fdk_imag_totals = totals[:, :, SSFPartSlot.FDK_IMAG]

        fdk_sqr_abs_part_blocks = \
            SSFPartBlocks(fdk_sqr_abs_totals)

        fdk_real_part_blocks = \
            SSFPartBlocks(fdk_real_totals)

        fdk_imag_part_blocks = \
            SSFPartBlocks(fdk_imag_totals)

        return cls(fdk_sqr_abs_part_blocks,
                   fdk_real_part_blocks,
                   fdk_imag_part_blocks)

    @property
    def mean(self):
        """Mean value of the static structure factor."""
        fdk_sqr_abs_part = self.fdk_sqr_abs_part
        fdk_real_part = self.fdk_real_part
        fdk_imag_part = self.fdk_imag_part

        return (fdk_sqr_abs_part.mean -
                fdk_real_part.mean ** 2 - fdk_imag_part.mean ** 2)

    @property
    def mean_error(self):
        """Error of the mean value of the static structure factor."""
        fdk_sqr_abs_part = self.fdk_sqr_abs_part
        fdk_real_part = self.fdk_real_part
        fdk_imag_part = self.fdk_imag_part

        fdk_real_part_mean = fdk_real_part.mean
        rdk_real_part_mean_error = fdk_real_part.mean_error
        fdk_imag_part_mean = fdk_imag_part.mean
        fdk_imag_part_mean_error = fdk_imag_part.mean_error

        # TODO: Check expressions for the error.
        return (fdk_sqr_abs_part.mean_error +
                2 * (fdk_real_part_mean * rdk_real_part_mean_error +
                     fdk_imag_part_mean * fdk_imag_part_mean_error))

    def hdf5_export(self, group: h5py.Group):
        """

        :param group:
        :return:
        """
        # Create the three groups.
        fdk_sqr_abs_group = group.require_group('fdk_sqr_abs')
        fdk_real_group = group.require_group('fdk_real')
        fdk_imag_group = group.require_group('fdk_imag')

        self.fdk_sqr_abs_part.hdf5_export(fdk_sqr_abs_group)
        self.fdk_real_part.hdf5_export(fdk_real_group)
        self.fdk_imag_part.hdf5_export(fdk_imag_group)

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """

        :param group:
        :return:
        """
        # Create the three groups.
        fdk_sqr_abs_group = group.get('fdk_sqr_abs')
        fdk_real_group = group.get('fdk_real')
        fdk_imag_group = group.get('fdk_imag')

        fdk_sqr_abs = SSFPartBlocks.from_hdf5_data(fdk_sqr_abs_group)
        fdk_real = SSFPartBlocks.from_hdf5_data(fdk_real_group)
        fdk_imag = SSFPartBlocks.from_hdf5_data(fdk_imag_group)

        return cls(fdk_sqr_abs, fdk_real, fdk_imag)

    def __add__(self, other: t.Any):
        """Concatenate the current data blocks with another data blocks."""
        if not isinstance(other, SSFBlocks):
            return NotImplemented
        fdk_sqr_abs_part = self.fdk_sqr_abs_part + other.fdk_sqr_abs_part
        fdk_real_part = self.fdk_real_part + other.fdk_real_part
        fdk_imag_part = self.fdk_imag_part + other.fdk_imag_part
        return SSFBlocks(fdk_sqr_abs_part, fdk_real_part, fdk_imag_part)


@attr.s(auto_attribs=True, frozen=True)
class PropsDataSeries:
    """The data from a VMC sampling."""

    #: The blocks of data of the sampling basic properties.
    iter_props_blocks: np.ndarray

    #: The  blocks of data of the static structure factor.
    ssf_blocks: t.Optional[np.ndarray] = None

    @cached_property
    def props(self):
        """"""
        source_data = self.iter_props_blocks
        return np.hstack(source_data)

    @property
    def wf_abs_log(self):
        """"""
        return self.props[vmc_udf_base.IterProp.WF_ABS_LOG]

    @property
    def energy(self):
        """"""
        return self.props[vmc_udf_base.IterProp.ENERGY]

    @property
    def move_state(self):
        """"""
        return self.props[vmc_udf_base.IterProp.MOVE_STAT]

    @cached_property
    def ss_factor(self):
        """"""
        if self.ssf_blocks is None:
            return None
        return np.vstack(self.ssf_blocks)


@attr.s(auto_attribs=True, frozen=True)
class PropsDataBlocks:
    """Results of a DMC sampling grouped in block totals."""

    energy: EnergyBlocks
    density: t.Optional[DensityBlocks] = None
    ss_factor: t.Optional[SSFBlocks] = None

    def hdf5_export(self, group: h5py.Group):
        """Export the data blocks to the given HDF5 group.

        :param group:
        :return:
        """
        energy_group = group.require_group('energy')
        self.energy.hdf5_export(energy_group)

        if self.density is not None:
            density_group = group.require_group('density')
            self.density.hdf5_export(density_group)

        if self.ss_factor is not None:
            ssf_group = group.require_group('ss_factor')
            self.ss_factor.hdf5_export(ssf_group)

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """

        :param group:
        :return:
        """
        energy_group = group.get('energy')
        energy_blocks = EnergyBlocks.from_hdf5_data(energy_group)

        density_group = group.get('density')
        if density_group is not None:
            density_blocks = DensityBlocks.from_hdf5_data(density_group)
        else:
            density_blocks = None

        ssf_group = group.get('ss_factor')
        if ssf_group is not None:
            ssf_blocks = SSFBlocks.from_hdf5_data(ssf_group)
        else:
            ssf_blocks = None

        return cls(energy_blocks, density_blocks, ssf_blocks)

    def merge(self, other: 'PropsDataBlocks'):
        """Concatenate the current data blocks with another data blocks."""
        if not isinstance(other, PropsDataBlocks):
            raise TypeError("'other' must be an instance of "
                            "'PropsDataBlocks'")
        energy = self.energy + other.energy
        density = self.density
        if density is None:
            density = other.density if other.density is not None else None
        else:
            if other.density is not None:
                density = density + other.density
        ssf = self.ss_factor
        if ssf is None:
            ssf = other.ss_factor if other.ss_factor is not None else None
        else:
            if other.ss_factor is not None:
                ssf = ssf + other.ss_factor
        return PropsDataBlocks(energy, density, ssf)


@attr.s(auto_attribs=True, frozen=True)
class SamplingData:
    """The data from a DMC sampling."""

    #: Data blocks.
    blocks: PropsDataBlocks

    #: Full data.
    series: t.Optional[PropsDataSeries] = None

    def hdf5_export(self, group: h5py.Group):
        """Export the data blocks to the given HDF5 group.

        :param group:
        :return:
        """
        # TODO: Export series...
        blocks_group = group.require_group('blocks')
        self.blocks.hdf5_export(blocks_group)

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """

        :param group:
        :return:
        """
        # TODO: Load series...
        blocks_group = group.get('blocks')
        props_data_blocks = PropsDataBlocks.from_hdf5_data(blocks_group)

        return cls(props_data_blocks, series=None)


# TODO: Delete this class.
@attr.s(auto_attribs=True, frozen=True)
class ProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: vmc_udf_base.State

    #: The data generated during the sampling.
    data: t.Optional[SamplingData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[vmc_udf_base.Sampling] = None
