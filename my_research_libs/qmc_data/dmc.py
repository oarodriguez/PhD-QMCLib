import typing as t
from abc import ABCMeta, abstractmethod

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.qmc_base import dmc as dmc_base
from my_research_libs.stats import reblock


class DataBlocks(metaclass=ABCMeta):
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

        return mean * np.sqrt(err_ow / ow_eff_size +
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


@attr.s(auto_attribs=True, frozen=True)
class NumWalkersBlocks(DataBlocks):
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
class WeightBlocks(DataBlocks):
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
class EnergyBlocks(DataBlocks):
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
class SSFBlocks(DataBlocks):
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
class DMCESDataSeries:
    """The data from a DMC sampling."""

    #: The blocks / batches of data.
    props_blocks: np.ndarray

    #:
    ssf_blocks: t.Optional[np.ndarray] = None

    @cached_property
    def props(self):
        """"""
        source_data = self.props_blocks
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
class DMCESDataBlocks:
    """Results of a DMC sampling grouped in block totals."""

    energy: EnergyBlocks
    weight: WeightBlocks
    num_walkers: NumWalkersBlocks
    ss_factor: t.Optional[SSFBlocks] = None


@attr.s(auto_attribs=True, frozen=True)
class DMCESData:
    """The data from a DMC sampling."""

    #: Data blocks.
    blocks: DMCESDataBlocks

    #: Full data.
    series: t.Optional[DMCESDataSeries] = None
