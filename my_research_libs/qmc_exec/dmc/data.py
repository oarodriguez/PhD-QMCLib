import typing as t
from abc import ABCMeta, abstractmethod

import attr
import h5py
import numpy as np
from cached_property import cached_property

from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.qmc_base.dmc import SSFPartSlot
from my_research_libs.stats import reblock

__all__ = [
    'DensityBlocks',
    'ProcResult',
    'EnergyBlocks',
    'NumWalkersBlocks',
    'PropBlocks',
    'PropsDataBlocks',
    'PropsDataSeries',
    'SamplingData',
    'SSFBlocks',
    'SSFPartBlocks',
    'UnWeightedPropBlocks',
    'WeightBlocks'
]


class PropBlocks(metaclass=ABCMeta):
    """Abstract class to represent data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: t.Optional[np.ndarray] = None

    @classmethod
    @abstractmethod
    def from_data(cls, *args, **kwargs):
        pass

    @property
    def mean(self):
        """Mean value of the blocks."""
        energy_rbc = self.reblock
        weight_rbc = self.weight_reblock

        if weight_rbc is None:
            return self.reblock.mean
        return energy_rbc.mean / weight_rbc.mean

    @property
    def mean_error(self):
        """Error of the mean value of the blocks."""
        ow_rbc = self.reblock

        ow_mean = ow_rbc.mean
        ow_var = ow_rbc.var
        ow_eff_size = ow_rbc.eff_size
        mean = self.mean

        if self.weight_reblock is None:
            #
            w_mean = 1.
            w_var = 0.
            oww_mean = ow_mean
            w_eff_size = 0.5
            oww_eff_size = 0.5

        else:
            #
            w_rbc = self.weight_reblock
            oww_rbc = self.cross_weight_reblock

            w_mean = w_rbc.mean
            w_var = w_rbc.var
            oww_mean = oww_rbc.mean
            w_eff_size = w_rbc.eff_size
            oww_eff_size = oww_rbc.eff_size

        err_ow = ow_var / ow_mean ** 2
        err_w = w_var / w_mean ** 2
        err_oww = (oww_mean - ow_mean * w_mean) / (ow_mean * w_mean)

        return np.abs(mean) * np.sqrt(err_ow / ow_eff_size +
                                      err_w / w_eff_size -
                                      2 * err_oww / oww_eff_size)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFObject.from_non_obj_data(self.totals)

    @property
    def weight_reblock(self):
        """Reblocking of the totals of the weights of every block."""
        if self.weight_totals is None:
            return None
        return reblock.OTFObject.from_non_obj_data(self.weight_totals)

    @property
    def cross_weight_reblock(self):
        """Reblocking of the total * weight_total of every block."""
        totals = self.totals
        weight_totals = self.weight_totals
        if weight_totals is None:
            return None
        cross_totals = totals * weight_totals
        return reblock.OTFObject.from_non_obj_data(cross_totals)

    def hdf5_export(self, group: h5py.Group):
        """Export the data to an HDF5 group object.

        :param group:
        :return:
        """
        # Array data go to a dataset.
        group.create_dataset('totals', data=self.totals)
        group.create_dataset('weight_totals', data=self.weight_totals)

        # Save attributes.
        group.attrs.update({
            'num_blocks': self.num_blocks,
            'num_time_steps_block': self.num_time_steps_block
        })

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """Create an instance from the data to an HDF5 group object.

        :param group:
        :return:
        """
        data = {
            'totals': group.get('totals').value,
            'weight_totals': group.get('weight_totals').value
        }
        data.update(group.attrs.items())
        # noinspection PyArgumentList
        return cls(**data)


class UnWeightedPropBlocks(metaclass=ABCMeta):
    """Abstract class to represent data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray

    @classmethod
    @abstractmethod
    def from_data(cls, *args, **kwargs):
        pass

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
        """

        :param group:
        :return:
        """
        # Create the necessary data sets and export data.
        group.create_dataset('totals', data=self.totals)
        group.attrs.update({
            'num_blocks': self.num_blocks,
            'num_time_steps_block': self.num_time_steps_block
        })

    @classmethod
    def from_hdf5_data(cls, group: h5py.Group):
        """

        :param group:
        :return:
        """
        data = {
            'totals': group.get('totals').value,
        }
        data.update(group.attrs.items())
        # noinspection PyArgumentList
        return cls(**data)


@attr.s(auto_attribs=True, frozen=True)
class NumWalkersBlocks(UnWeightedPropBlocks):
    """Number of walkers data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        weight_data = data[dmc_base.IterProp.NUM_WALKERS]
        if reduce_data:
            weight_totals = weight_data.sum(axis=1)
        else:
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class WeightBlocks(UnWeightedPropBlocks):
    """Weight data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        weight_data = data[dmc_base.IterProp.WEIGHT]
        if reduce_data:
            weight_totals = weight_data.sum(axis=1)
        else:
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class EnergyBlocks(PropBlocks):
    """Energy data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: np.ndarray

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param num_blocks:
        :param num_time_steps_block:
        :param data:
        :param reduce_data:
        :return:
        """
        energy_data = data[dmc_base.IterProp.ENERGY]
        weight_data = data[dmc_base.IterProp.WEIGHT]
        if reduce_data:
            totals = energy_data.sum(axis=1)
            weight_totals = weight_data.sum(axis=1)
        else:
            totals = energy_data
            weight_totals = weight_data

        return cls(num_blocks,
                   num_time_steps_block,
                   totals,
                   weight_totals)


@attr.s(auto_attribs=True, frozen=True)
class DensityBlocks(PropBlocks):
    """Density data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: np.ndarray
    as_pure_est: bool = True  # NOTE: Maybe we do not need this.

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  density_data: np.ndarray,
                  props_data: np.ndarray,
                  reduce_data: bool = True,
                  as_pure_est: bool = True,
                  pure_est_reduce_factor: np.ndarray = None):
        """

        :param reduce_data:
        :param num_blocks:
        :param num_time_steps_block:
        :param density_data:
        :param props_data:
        :param as_pure_est:
        :param pure_est_reduce_factor:
        :return:
        """
        nts_block = num_time_steps_block
        weight_data = props_data[dmc_base.IterProp.WEIGHT]

        if not as_pure_est:

            if reduce_data:
                totals = density_data.sum(axis=1)
                weight_totals = weight_data.sum(axis=1)

            else:
                totals = density_data
                weight_totals = weight_data

        else:
            # Normalize the pure estimator.
            if reduce_data:

                # Reductions are not used in pure estimators.
                # We just take the last element.
                totals = density_data[:, nts_block - 1, :]
                weight_totals = weight_data[:, nts_block - 1]

            else:
                totals = density_data
                weight_totals = weight_data * pure_est_reduce_factor

        # Add an extra dimension.
        weight_totals = weight_totals[:, np.newaxis]

        return cls(num_blocks, num_time_steps_block,
                   totals, weight_totals, as_pure_est)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFSet.from_non_obj_data(self.totals)

    @property
    def weight_reblock(self):
        """Reblocking of the totals of the weights of every block."""
        if self.weight_totals is None:
            return None
        return reblock.OTFSet.from_non_obj_data(self.weight_totals)

    @property
    def cross_weight_reblock(self):
        """Reblocking of the total * weight_total of every block."""
        totals = self.totals
        weight_totals = self.weight_totals
        if weight_totals is None:
            return None
        cross_totals = totals * weight_totals
        return reblock.OTFSet.from_non_obj_data(cross_totals)


@attr.s(auto_attribs=True, frozen=True)
class SSFPartBlocks(PropBlocks):
    """Structure Factor data in blocks."""

    num_blocks: int
    num_time_steps_block: int
    totals: np.ndarray
    weight_totals: np.ndarray
    as_pure_est: bool = True  # NOTE: Maybe we do not need this.

    @classmethod
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  sf_data: np.ndarray,
                  props_data: np.ndarray,
                  reduce_data: bool = True,
                  as_pure_est: bool = True,
                  pure_est_reduce_factor: np.ndarray = None):
        """

        :param reduce_data:
        :param num_blocks:
        :param num_time_steps_block:
        :param sf_data:
        :param props_data:
        :param as_pure_est:
        :param pure_est_reduce_factor:
        :return:
        """
        nts_block = num_time_steps_block
        weight_data = props_data[dmc_base.IterProp.WEIGHT]

        if not as_pure_est:

            if reduce_data:
                totals = sf_data.sum(axis=1)
                weight_totals = weight_data.sum(axis=1)

            else:
                totals = sf_data
                weight_totals = weight_data

        else:
            # Normalize the pure estimator.
            if reduce_data:

                # Reductions are not used in pure estimators.
                # We just take the last element.
                totals = sf_data[:, nts_block - 1, :]
                weight_totals = weight_data[:, nts_block - 1]

            else:
                totals = sf_data
                weight_totals = weight_data * pure_est_reduce_factor

        # Add an extra dimension.
        weight_totals = weight_totals[:, np.newaxis]

        return cls(num_blocks, num_time_steps_block,
                   totals, weight_totals, as_pure_est)

    @property
    def reblock(self):
        """Reblocking of the totals of every block."""
        return reblock.OTFSet.from_non_obj_data(self.totals)

    @property
    def weight_reblock(self):
        """Reblocking of the totals of the weights of every block."""
        if self.weight_totals is None:
            return None
        return reblock.OTFSet.from_non_obj_data(self.weight_totals)

    @property
    def cross_weight_reblock(self):
        """Reblocking of the total * weight_total of every block."""
        totals = self.totals
        weight_totals = self.weight_totals
        if weight_totals is None:
            return None
        cross_totals = totals * weight_totals
        return reblock.OTFSet.from_non_obj_data(cross_totals)


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
    def from_data(cls, num_blocks: int,
                  num_time_steps_block: int,
                  sf_data: np.ndarray,
                  props_data: np.ndarray,
                  reduce_data: bool = True,
                  as_pure_est: bool = True,
                  pure_est_reduce_factor: np.ndarray = None):
        """

        :param reduce_data:
        :param num_blocks:
        :param num_time_steps_block:
        :param sf_data:
        :param props_data:
        :param as_pure_est:
        :param pure_est_reduce_factor:
        :return:
        """
        nts_block = num_time_steps_block
        weight_data = props_data[dmc_base.IterProp.WEIGHT]

        if not as_pure_est:

            if reduce_data:
                totals = sf_data.sum(axis=1)
                weight_totals = weight_data.sum(axis=1)

            else:
                totals = sf_data
                weight_totals = weight_data

        else:
            # Normalize the pure estimator.
            if reduce_data:

                # Reductions are not used in pure estimators.
                # We just take the last element.
                totals = sf_data[:, nts_block - 1, :]
                weight_totals = weight_data[:, nts_block - 1]

            else:
                totals = sf_data
                weight_totals = weight_data * pure_est_reduce_factor

        # Add an extra dimension.
        weight_totals = weight_totals[:, np.newaxis]

        # The totals of every part.
        fdk_sqr_abs_totals = totals[:, :, SSFPartSlot.FDK_SQR_ABS]
        fdk_real_totals = totals[:, :, SSFPartSlot.FDK_REAL]
        fdk_imag_totals = totals[:, :, SSFPartSlot.FDK_IMAG]

        fdk_sqr_abs_part_blocks = \
            SSFPartBlocks(num_blocks, num_time_steps_block,
                          fdk_sqr_abs_totals, weight_totals, as_pure_est)

        fdk_real_part_blocks = \
            SSFPartBlocks(num_blocks, num_time_steps_block,
                          fdk_real_totals, weight_totals, as_pure_est)

        fdk_imag_part_blocks = \
            SSFPartBlocks(num_blocks, num_time_steps_block,
                          fdk_imag_totals, weight_totals, as_pure_est)

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


@attr.s(auto_attribs=True, frozen=True)
class PropsDataSeries:
    """The data from a DMC sampling."""

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
    def energy(self):
        """"""
        return self.props[dmc_base.IterProp.ENERGY]

    @property
    def weight(self):
        """"""
        return self.props[dmc_base.IterProp.WEIGHT]

    @property
    def num_walkers(self):
        """"""
        return self.props[dmc_base.IterProp.NUM_WALKERS]

    @property
    def ref_energy(self):
        """"""
        return self.props[dmc_base.IterProp.REF_ENERGY]

    @property
    def accum_energy(self):
        """"""
        return self.props[dmc_base.IterProp.ACCUM_ENERGY]

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
    weight: WeightBlocks
    num_walkers: NumWalkersBlocks
    density: t.Optional[DensityBlocks] = None
    ss_factor: t.Optional[SSFBlocks] = None

    def hdf5_export(self, group: h5py.Group):
        """Export the data blocks to the given HDF5 group.

        :param group:
        :return:
        """
        energy_group = group.require_group('energy')
        self.energy.hdf5_export(energy_group)

        weight_group = group.require_group('weight')
        self.weight.hdf5_export(weight_group)

        num_walkers_group = group.require_group('num_walkers')
        self.num_walkers.hdf5_export(num_walkers_group)

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

        weight_group = group.get('weight')
        weight_blocks = WeightBlocks.from_hdf5_data(weight_group)

        num_walkers_group = group.get('num_walkers')
        num_walkers_blocks = NumWalkersBlocks.from_hdf5_data(num_walkers_group)

        density_group = group.get('density')
        if density_group is not None:
            density_blocks = DensityBlocks.from_hdf5_data(density_group)
        else:
            density_blocks = None

        ssf_group = group.get('ss_factor')
        if ssf_group is not None:
            ssf_block = SSFBlocks.from_hdf5_data(ssf_group)
        else:
            ssf_block = None

        return cls(energy_blocks, weight_blocks,
                   num_walkers_blocks, density_blocks, ssf_block)


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


@attr.s(auto_attribs=True, frozen=True)
class ProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: dmc_base.State

    #: The data generated during the sampling.
    data: t.Optional[SamplingData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[dmc_base.Sampling] = None
