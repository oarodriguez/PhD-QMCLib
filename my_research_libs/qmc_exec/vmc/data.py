import typing as t
from abc import ABCMeta, abstractmethod

import attr
import numpy as np
from cached_property import cached_property

from my_research_libs.qmc_base import vmc as vmc_udf_base
from my_research_libs.qmc_base.vmc import SSFPartSlot
from my_research_libs.stats import reblock


class PropBlocks(metaclass=ABCMeta):
    """Abstract class to represent data in blocks."""

    num_blocks: int
    num_steps_block: int
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


@attr.s(auto_attribs=True, frozen=True)
class EnergyBlocks(PropBlocks):
    """Energy data in blocks."""

    num_blocks: int
    num_steps_block: int
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
        energy_data = data[vmc_udf_base.IterProp.ENERGY]
        if reduce_data:
            totals = energy_data.sum(axis=1)
        else:
            totals = energy_data

        return cls(num_blocks,
                   num_time_steps_block,
                   totals)


@attr.s(auto_attribs=True, frozen=True)
class SSFPartBlocks(PropBlocks):
    """Structure Factor data in blocks."""

    num_blocks: int
    num_steps_block: int
    totals: np.ndarray

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
        if reduce_data:
            totals = sf_data.sum(axis=1)
        else:
            totals = sf_data
        return cls(num_blocks, num_time_steps_block, totals)

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
    def from_data(cls, num_blocks: int,
                  num_steps_block: int,
                  sf_data: np.ndarray,
                  reduce_data: bool = True):
        """

        :param reduce_data:
        :param num_blocks:
        :param num_steps_block:
        :param sf_data:
        :return:
        """
        if reduce_data:
            totals = sf_data.sum(axis=1)
        else:
            totals = sf_data

        # The totals of every part.
        fdk_sqr_abs_totals = totals[:, :, SSFPartSlot.FDK_SQR_ABS]
        fdk_real_totals = totals[:, :, SSFPartSlot.FDK_REAL]
        fdk_imag_totals = totals[:, :, SSFPartSlot.FDK_IMAG]

        fdk_sqr_abs_part_blocks = \
            SSFPartBlocks(num_blocks, num_steps_block,
                          fdk_sqr_abs_totals)

        fdk_real_part_blocks = \
            SSFPartBlocks(num_blocks, num_steps_block,
                          fdk_real_totals)

        fdk_imag_part_blocks = \
            SSFPartBlocks(num_blocks, num_steps_block,
                          fdk_imag_totals)

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
        return self.props[vmc_udf_base.IterProp.ENERGY]

    @property
    def weight(self):
        """"""
        return self.props[vmc_udf_base.IterProp.WEIGHT]

    @property
    def num_walkers(self):
        """"""
        return self.props[vmc_udf_base.IterProp.NUM_WALKERS]

    @property
    def ref_energy(self):
        """"""
        return self.props[vmc_udf_base.IterProp.REF_ENERGY]

    @property
    def accum_energy(self):
        """"""
        return self.props[vmc_udf_base.IterProp.ACCUM_ENERGY]

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
    ss_factor: t.Optional[SSFBlocks] = None


@attr.s(auto_attribs=True, frozen=True)
class SamplingData:
    """The data from a DMC sampling."""

    #: Data blocks.
    blocks: PropsDataBlocks

    #: Full data.
    series: t.Optional[PropsDataSeries] = None


@attr.s(auto_attribs=True, frozen=True)
class ProcResult:
    """Result of the DMC estimator sampling."""

    #: The last state of the sampling.
    state: vmc_udf_base.State

    #: The data generated during the sampling.
    data: t.Optional[SamplingData] = None

    #: The sampling object used to generate the results.
    sampling: t.Optional[vmc_udf_base.Sampling] = None
