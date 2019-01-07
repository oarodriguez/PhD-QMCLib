import typing as t
from abc import ABCMeta, abstractmethod
from collections import Mapping
from enum import Enum, unique
from math import ceil, floor, log, log2, sqrt
from typing import Iterator
from warnings import warn

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from scipy.optimize import curve_fit

__all__ = [
    'IACFitParams',
    'IACTimeFit',
    'Object',
    'ObjectBase',
    'OTFObject',
    'OTFObjectDataField',
    'OTFSet',
    'SetBase',
    'on_the_fly_obj_data_init',
    'on_the_fly_obj_create',
    'on_the_fly_extend_obj_data_set',
    'on_the_fly_obj_data_order'
]


class IACFitParams(t.NamedTuple):
    """"""
    iac_time: float
    eac_time: float
    c_time: float


class IACFitErrors(t.NamedTuple):
    """"""
    iac_time: float
    eac_time: float
    c_time: float


@attr.s(auto_attribs=True, frozen=True)
class IACTimeFit:
    """Fit for the integrated autocorrelation time."""
    # TODO: Add docstrings...
    times: np.ndarray
    iac_times: np.ndarray
    results: t.Tuple = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        try:
            self_fit = curve_fit(self.__func__, self.times, self.iac_times)
        except TypeError as e:
            exc_msg = f'attempt to fit data to target function failed'
            raise TypeError(exc_msg) from e

        super().__setattr__('results', self_fit)

    @staticmethod
    def __func__(time, iac_time, eac_time, const):
        """A function to approximate the correlation times..

        :param time: The size of the block.
        :param iac_time:  The integrated autocorrelation time.
        :param eac_time:  The exponential autocorrelation time.
        :param const: The fitting constant.
        :return:
        """
        return iac_time - const * np.exp(-time / eac_time)

    def __call__(self, times):
        """Callable interface."""
        return self.__func__(times, *self.params)

    @property
    def params(self):
        """The fit parameters."""
        return IACFitParams(*self.results[0])

    @property
    def cov_matrix(self):
        """Covariance matrix of the fit."""
        return self.results[1]

    @property
    def errors(self):
        """The errors of the fit parameters."""
        return IACFitErrors(*np.sqrt(np.diag(self.cov_matrix)))

    @property
    def iac_time(self):
        """Integrated autocorrelation time."""
        return self.params.iac_time

    @property
    def eac_time(self):
        """Exponential autocorrelation time."""
        return self.params.eac_time


@attr.s(auto_attribs=True, frozen=True)
class BlockedData:
    """"""
    num_blocks: int
    block_size: int
    data: np.ndarray


class ObjectBase(metaclass=ABCMeta):
    """Base class for the reblocking of a series of data."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Variance delta degrees of freedom.
    var_ddof: int

    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the source data."""
        pass

    @cached_property
    def mean(self):
        """The mean of the source data."""
        return self.source_data.mean(axis=0)

    @cached_property
    def var(self):
        """The variance of the source data."""
        return self.source_data.var(axis=0, ddof=self.var_ddof)

    @property
    @abstractmethod
    def block_sizes(self) -> np.ndarray:
        """The sizes of the blocks used in the reblocking."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def means(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def vars(self) -> np.ndarray:
        pass

    @property
    def errors(self):
        """Errors of the mean for each of the block sizes."""
        return np.sqrt(self.vars / self.num_blocks)

    @property
    def iac_times(self):
        """Integrated autocorrelation times for each of the block sizes."""
        self_vars = self.vars
        block_sizes = self.block_sizes
        data_var = self.var

        # Correlation times.
        return 0.5 * block_sizes * self_vars / data_var

    @property
    def opt_block_size(self):
        """The optimal block size."""
        block_sizes = self.block_sizes
        data_size = self.size
        int_corr_times = self.iac_times
        # B^3 > 2N * (2 \tau)^2
        criterion = block_sizes ** 3 > 8 * data_size * int_corr_times ** 2
        opt_block_size = block_sizes[criterion].min()
        return opt_block_size

    @property
    def opt_iac_time(self):
        """The optimal integrated autocorrelation time."""
        criterion = self.block_sizes == self.opt_block_size
        opt_corr_time = self.iac_times[criterion]
        return opt_corr_time[0]

    @property
    def eff_size(self):
        """Returns the effective size of the source data.

        The calculation takes into account the temporal correlations
        between the data to get the effective size of the statistical
        sample.
        """
        data_size = self.size
        # NOTE: Should we return an int or a float?
        return data_size / (2 * self.opt_iac_time)

    @property
    def mean_eff_error(self):
        """The effective error of the mean of the data.

        This value takes into account the temporal correlations between
        the data to get the effective error of the mean of the statistical
        sample.
        """
        eff_size = self.eff_size
        return sqrt(self.var / eff_size)

    @property
    def iac_time_fit(self):
        """Fit of the integrated autocorrelation time."""
        return IACTimeFit(self.block_sizes, self.iac_times)


class SetBase(Mapping):
    """Base class for the reblocking of a set of tabular data."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Variance delta degrees of freedom.
    var_ddof: int

    @property
    @abstractmethod
    def size(self) -> np.ndarray:
        """The size of the source data."""
        pass

    @cached_property
    def mean(self):
        """The mean of the source data."""
        return self.source_data.mean(axis=0)

    @cached_property
    def var(self):
        """The variance of the source data."""
        return self.source_data.var(axis=0, ddof=self.var_ddof)

    @property
    @abstractmethod
    def block_sizes(self) -> np.ndarray:
        """The sizes of the blocks used in the reblocking."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def means(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def vars(self) -> np.ndarray:
        pass

    @property
    def errors(self):
        """Errors of the mean for each of the block sizes."""
        return np.sqrt(self.vars / self.num_blocks)

    @cached_property
    def iac_times(self):
        """Integrated autocorrelation times for each of the block sizes."""
        self_vars = self.vars
        block_sizes = self.block_sizes
        data_var = self.var[:, np.newaxis]

        # Correlation times.
        return 0.5 * block_sizes * self_vars / data_var

    @property
    @abstractmethod
    def opt_block_size(self):
        """The optimal block size."""
        pass

    @property
    @abstractmethod
    def opt_iac_time(self):
        """The optimal integrated autocorrelation time."""
        pass

    @property
    def eff_size(self):
        """Returns the effective size of the source data.

        The calculation takes into account the temporal correlations
        between the data to get the effective size of the statistical
        sample.
        """
        data_size = self.size
        # NOTE: Should we return an int or a float?
        return data_size / (2 * self.opt_iac_time)

    @property
    def mean_eff_error(self):
        """The effective error of the mean of the data.

        This value takes into account the temporal correlations between
        the data to get the effective error of the mean of the statistical
        sample.
        """
        eff_size = self.eff_size
        return np.sqrt(self.var / eff_size)


@attr.s(auto_attribs=True, frozen=True)
class Object(ObjectBase):
    """Realizes a blocking/binning analysis of serially correlated data."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Minimum number of blocks.
    min_num_blocks: int = 2

    #: Variance delta degrees of freedom.
    var_ddof: int = attr.ib(default=1, init=False)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        # NOTE: Allow only 1d arrays by now.
        assert len(self.source_data.shape) == 1

        # NOTE: Set var_ddof attribute to one always.
        # TODO: Maybe we should remove the attribute from init.
        var_ddof = 1
        super().__setattr__('var_ddof', var_ddof)

        if self.min_num_blocks < 2:
            raise ValueError('the minimum number of blocks of the reblocking '
                             'is two')

    @property
    def size(self):
        """The size of the source data."""
        return len(self.source_data)

    @cached_property
    def block_sizes(self):
        """The sizes of the blocks used in the reblocking."""
        min_num_blocks = self.min_num_blocks
        data_length = len(self.source_data)
        max_order = int(floor(log2(data_length)))
        min_order = int(ceil(log2(min_num_blocks)))

        if max_order < min_order:
            raise ValueError('source data cannot be grouped in the minimum '
                             'number of blocks requested')

        # Powers of 2 for the block sizes.
        block_sizes = 1 << np.arange(max_order - min_order + 1)
        return block_sizes.astype(np.int64)

    @cached_property
    def num_blocks(self):
        """The number of blocks for each block size."""
        data_size = self.size
        num_blocks = data_size // np.array(self.block_sizes, dtype=np.int64)
        return num_blocks.astype(np.int64)

    @cached_property
    def blocked_data(self) -> t.List[BlockedData]:
        """The source data reshaped in blocks."""
        self_data = self.source_data
        block_sizes = self.block_sizes
        data_size = self.size

        blocked_data = []
        for size in block_sizes:
            num_blocks = data_size // size
            eff_size = num_blocks * size
            data_shape = (num_blocks, size)
            shaped_data = self_data[:eff_size].reshape(data_shape)
            data = BlockedData(num_blocks, size, shaped_data)
            blocked_data.append(data)
        return blocked_data

    @cached_property
    def means(self):
        """Means of each of the blocks."""
        blocked_data = self.blocked_data
        data_means = []
        for block in blocked_data:
            shaped_data = block.data
            data_mean = shaped_data.mean(axis=1).mean(axis=0)
            data_means.append(data_mean)
        return np.array(data_means)

    @cached_property
    def vars(self):
        """Variances of each of the blocks."""
        var_ddof = self.var_ddof
        blocked_data = self.blocked_data
        data_vars = []
        for block in blocked_data:
            shaped_data = block.data
            data_mean = shaped_data.mean(axis=1).var(axis=0, ddof=var_ddof)
            data_vars.append(data_mean)
        return np.array(data_vars)


@unique
class OTFObjectDataField(str, Enum):
    """Fields of the reblocking accumulated array."""
    BLOCK_SIZE = 'BLOCK_SIZE'
    MEANS = 'MEANS'
    MEANS_SQR = 'MEANS_SQR'
    NUM_BLOCKS = 'NUM_BLOCKS'


BLOCK_SIZE_FIELD = OTFObjectDataField.BLOCK_SIZE.value
NUM_BLOCKS_FIELD = OTFObjectDataField.NUM_BLOCKS.value
MEANS_SQR_FIELD = OTFObjectDataField.MEANS_SQR.value
MEANS_FIELD = OTFObjectDataField.MEANS.value

otf_data_dtype = np.dtype([
    (BLOCK_SIZE_FIELD, np.int64),
    (MEANS_FIELD, np.float64),
    (MEANS_SQR_FIELD, np.float64),
    (NUM_BLOCKS_FIELD, np.int64)
])

T_ObjectDataSet = t.Union[np.ndarray, t.Sequence[np.ndarray]]


@nb.njit
def on_the_fly_obj_data_order(source_data: np.ndarray):
    """Estimates the maximum order of an on-the-fly reblocking.

    The maximum order determines the size of the output array of the
    on-the-fly reblocking process.

    :param source_data:
    :return:
    """
    data_length = source_data.shape[0]
    return int(floor(log(data_length) / log(2)))


def on_the_fly_obj_data_init(order: int, num_cols: int = None) -> np.ndarray:
    """Initializes the reblocking array.

    If ``num_cols`` is an integer, the data will be a 2d structured array
    with as as many rows as ``num_cols`` and as many columns as ``order + 1``.
    If it is ``None``, a single array for a unique reblocking will be
    returned.

    :param order:
    :param num_cols:
    :return:
    """
    if num_cols is None:
        # Return a single array to save the data of a reblocking.
        return _on_the_fly_obj_data_init(order, num_cols=1)[0]

    return _on_the_fly_obj_data_init(order, num_cols)


def on_the_fly_obj_create(source_data: np.ndarray):
    """Realizes a reblocking analysis at increasing levels.

    This function calculates a reblocking analysis at increasing levels.
    Each subsequent level takes twice the number of elements of
    source_data than the previous to construct a block and to calculate
    the averages.

    :param source_data:
    :return:
    """
    # Convert the source data to a 2d array always.
    source_data = np.asarray(source_data)

    assert len(source_data.shape) >= 1

    is_1d_array = len(source_data.shape) == 1
    if is_1d_array:
        source_data = source_data[:, np.newaxis]

    otf_data_array = _on_the_fly_obj_create(source_data)
    if is_1d_array:
        # Return a single reblocking array.
        return otf_data_array[0]

    return otf_data_array


@nb.njit
def _on_the_fly_obj_data_init(order: int, num_cols: int) -> np.ndarray:
    """Initializes the reblocking array.

    :param order:
    :param num_cols:
    :return:
    """
    otf_data_array = np.zeros((num_cols, order + 1), dtype=otf_data_dtype)
    block_size_array = otf_data_array[BLOCK_SIZE_FIELD]
    for order in range(order + 1):
        block_size = 1 << order
        block_size_array[:, order] = block_size

    return otf_data_array


@nb.njit
def _on_the_fly_obj_create(source_data: np.ndarray):
    """Realizes a reblocking analysis at increasing levels.

    This function calculates a reblocking analysis at increasing levels.
    Each subsequent level takes twice the number of elements of
    source_data than the previous to construct a block and to calculate
    the averages.

    :param source_data:
    :return:
    """
    max_order = on_the_fly_obj_data_order(source_data)
    data_length, num_cols = source_data.shape

    # Initialize arrays.
    otf_data_array = _on_the_fly_obj_data_init(max_order, num_cols)
    means_array = np.zeros((num_cols, max_order + 1, 2), dtype=np.float64)

    # TODO: We only need one block_size_array and num_blocks_array for
    #  the whole reblocking.
    block_size_array = otf_data_array[BLOCK_SIZE_FIELD]
    means_sum_array = otf_data_array[MEANS_FIELD]
    means_sqr_sum_array = otf_data_array[MEANS_SQR_FIELD]
    num_blocks_array = otf_data_array[NUM_BLOCKS_FIELD]

    # The size of the next block is twice the previous.
    ini_order = order = 0
    ini_block_size = block_size = block_size_array[0, order]
    ini_next_block_size = next_block_size = block_size << 1

    data_length = source_data.shape[0]
    for index in range(data_length):
        #
        block_index = index
        mean_index = block_index % 2

        for nc in range(num_cols):
            #
            data_mean = source_data[index, nc]
            means_sum_array[nc, order] += data_mean
            means_sqr_sum_array[nc, order] += data_mean ** 2
            means_array[nc, order, mean_index] = data_mean
            num_blocks_array[nc, order] += 1

        while True:
            #
            if order >= max_order:
                #
                break

            if not (index + 1) % next_block_size:
                #
                order += 1
                block_size = block_size << 1
                block_index = (index + 1) // block_size - 1
                mean_index = block_index % 2

                for nc in range(num_cols):
                    #
                    data_mean = means_array[nc, order - 1].mean()
                    means_sum_array[nc, order] += data_mean
                    means_sqr_sum_array[nc, order] += data_mean ** 2
                    means_array[nc, order, mean_index] = data_mean
                    num_blocks_array[nc, order] += 1

                # Doubles the current block size for the next order.
                next_block_size = block_size << 1

            else:
                #
                break

        # Reset order to the initial order.
        # NOTE: This will be executed always the if loop breaks.
        #  This does not seems good.
        order = ini_order
        block_size = ini_block_size
        next_block_size = ini_next_block_size

    return otf_data_array


@nb.njit
def recursive_reblocking(means_array: np.ndarray,
                         block_size_array: np.ndarray,
                         means_sum_array: np.ndarray,
                         means_sqr_sum_array: np.ndarray,
                         num_blocks_array: np.ndarray,
                         index: int,
                         order: int,
                         max_order: int,
                         reblocking: np.ndarray):
    """Does a reblocking analysis at higher orders recursively.

    :param means_array:
    :param block_size_array:
    :param means_sum_array:
    :param means_sqr_sum_array:
    :param num_blocks_array:
    :param index:
    :param order:
    :param max_order:
    :param reblocking:
    :return:
    """
    data_mean = means_array[order - 1].mean()
    means_sum_array[order] += data_mean
    means_sqr_sum_array[order] += data_mean ** 2
    num_blocks_array[order] += 1

    block_size = 1 << order
    block_index = (index + 1) // block_size - 1
    mean_index = block_index % 2
    means_array[order, mean_index] = data_mean

    if order < max_order:
        next_order = order + 1
        next_block_size = block_size << 1

        if not (index + 1) % next_block_size:
            recursive_reblocking(means_array, block_size_array,
                                 means_sum_array, means_sqr_sum_array,
                                 num_blocks_array, index, next_order,
                                 max_order, reblocking=reblocking)


@attr.s(auto_attribs=True, frozen=True)
class OTFObject(ObjectBase):
    """Realizes a reblocking analysis at increasing levels."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Minimum number of blocks.
    min_num_blocks: t.Optional[int] = 2

    #: Variance delta degrees of freedom.
    var_ddof: int = 1

    def __attrs_post_init__(self):
        """"""
        # Only allow reblocking accumulated values.
        self_source_data = self.source_data
        if not self_source_data.dtype == otf_data_dtype:
            raise TypeError("source_data is not a reblocking table.")

        if not len(self_source_data.shape) == 1:
            raise ValueError("source_data must be a 1d array")

        # NOTE: Set var_ddof attribute to one always.
        # TODO: Maybe we should remove the attribute from init.
        var_ddof = 1
        super().__setattr__('var_ddof', var_ddof)

        min_num_blocks = self.min_num_blocks or 2
        if min_num_blocks < 2:
            raise ValueError('the minimum number of blocks of the reblocking '
                             'is two')
        object.__setattr__(self, 'min_num_blocks', min_num_blocks)

        data_num_blocks = self_source_data[NUM_BLOCKS_FIELD]
        criterion = data_num_blocks >= min_num_blocks
        if not np.count_nonzero(criterion):
            raise ValueError('the source data is empty for the requested '
                             'minimum number of blocks.')

        source_data = self_source_data[criterion]
        super().__setattr__('source_data', source_data)

    @classmethod
    def from_non_obj_data(cls, seq: t.Union[t.Sequence, np.ndarray],
                          min_num_blocks: int = None):
        """

        :param seq:
        :param min_num_blocks:
        :return:
        """
        obj_data = on_the_fly_obj_create(seq)
        return cls(obj_data, min_num_blocks=min_num_blocks)

    @classmethod
    def from_obj_data_set(cls, obj_data_set: T_ObjectDataSet,
                          min_num_blocks: int = None):
        """

        :param obj_data_set:
        :param min_num_blocks:
        :return:
        """
        obj_data = on_the_fly_extend_obj_data_set(obj_data_set)
        return cls(obj_data, min_num_blocks=min_num_blocks)

    @cached_property
    def size(self):
        """"""
        return self.num_blocks[0]

    @cached_property
    def mean(self):
        """The mean of the source data."""
        return self.means[0]

    @cached_property
    def var(self):
        """The variance of the source data."""
        # The variance at first order, as an array of one element.
        return self.vars[0]

    @property
    def block_sizes(self):
        """"""
        return self.source_data[BLOCK_SIZE_FIELD]

    @property
    def num_blocks(self):
        return self.source_data[NUM_BLOCKS_FIELD]

    @cached_property
    def means(self):
        """"""
        means_sum = self.source_data[MEANS_FIELD]
        return means_sum / self.num_blocks

    @cached_property
    def vars(self):
        """"""
        num_blocks = self.num_blocks
        means_sqr_sum = self.source_data[MEANS_SQR_FIELD]
        means_sqr = means_sqr_sum / num_blocks
        ddof_num_blocks = num_blocks - self.var_ddof
        return num_blocks * (means_sqr - self.means ** 2) / ddof_num_blocks


@attr.s(auto_attribs=True, frozen=True)
class OTFSet(SetBase):
    """Realizes a reblocking analysis at increasing levels."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Minimum number of blocks.
    min_num_blocks: t.Optional[int] = 2

    #: Variance delta degrees of freedom.
    var_ddof: int = 1

    def __attrs_post_init__(self):
        """"""
        # Only allow reblocking accumulated values.
        self_source_data = self.source_data
        if not self_source_data.dtype == otf_data_dtype:
            raise TypeError("source_data is not a reblocking table.")

        if not len(self_source_data.shape) == 2:
            # Re-raise with a nice message 🙂.
            raise ValueError("source_data must be a 2d array")

        # Start with a 2d array.
        block_size_set = self_source_data[BLOCK_SIZE_FIELD]

        # Check that the data has a consistent format.
        assert np.all(np.diff(block_size_set, axis=0) == 0)

        # Pick only one element as they are all equal.
        data_num_blocks = self_source_data[NUM_BLOCKS_FIELD][0, :]

        # NOTE: Set var_ddof attribute to one always.
        # TODO: Maybe we should remove the attribute from init.
        var_ddof = 1
        super().__setattr__('var_ddof', var_ddof)

        min_num_blocks = self.min_num_blocks or 2
        if min_num_blocks < 2:
            raise ValueError('the minimum number of blocks of the reblocking '
                             'is two')
        object.__setattr__(self, 'min_num_blocks', min_num_blocks)

        criterion = data_num_blocks >= min_num_blocks
        if not np.count_nonzero(criterion):
            raise ValueError('the source data is empty for the requested '
                             'minimum number of blocks.')

        source_data = self_source_data[:, criterion]
        super().__setattr__('source_data', source_data)

    @classmethod
    def from_non_obj_data(cls, seq: t.Union[t.Sequence, np.ndarray],
                          min_num_blocks: int = None):
        """

        :param seq:
        :param min_num_blocks:
        :return:
        """
        obj_data = on_the_fly_obj_create(seq)
        return cls(obj_data, min_num_blocks=min_num_blocks)

    @classmethod
    def from_obj_data_set(cls, obj_data_set: T_ObjectDataSet,
                          min_num_blocks: int = None):
        """

        :param obj_data_set:
        :param min_num_blocks:
        :return:
        """
        obj_data = on_the_fly_extend_obj_data_set(obj_data_set)
        return cls(obj_data, min_num_blocks=min_num_blocks)

    @cached_property
    def size(self) -> np.ndarray:
        """"""
        return self.num_blocks[:, 0]

    @cached_property
    def mean(self):
        """The mean of the source data."""
        return self.means[:, 0]

    @cached_property
    def var(self):
        """The variance of the source data."""
        # The variance at first order, as an array of one element.
        return self.vars[:, 0]

    @property
    def block_sizes(self):
        """"""
        return self.source_data[BLOCK_SIZE_FIELD]

    @property
    def num_blocks(self):
        return self.source_data[NUM_BLOCKS_FIELD]

    @cached_property
    def means(self):
        """The means of the means of the reblocking."""
        means_sum = self.source_data[MEANS_FIELD]
        return means_sum / self.num_blocks

    @cached_property
    def vars(self):
        """The variances of the means of the reblocking."""
        num_blocks = self.num_blocks
        means_sqr_sum = self.source_data[MEANS_SQR_FIELD]
        means_sqr = means_sqr_sum / num_blocks
        ddof_num_blocks = num_blocks - self.var_ddof
        return num_blocks * (means_sqr - self.means ** 2) / ddof_num_blocks

    @cached_property
    def opt_block_size(self):
        """The optimal block size."""
        block_sizes = self.block_sizes
        data_size = self.size[:, np.newaxis]
        iac_times = self.iac_times

        # B^3 > 2N * (2 \tau)^2
        criterion = block_sizes ** 3 > 8 * data_size * iac_times ** 2

        opt_block_sizes = []
        for row_idx, row_positions in enumerate(criterion):
            valid_sizes = block_sizes[row_idx, row_positions]
            if not np.count_nonzero(valid_sizes):
                row_opt_block_size = block_sizes.max()
                warn("the optimum block size criterion is not satisfied by "
                     "any of the autocorrelation times. The maximum block "
                     "size will be treated as the optimal one. You may try "
                     "to gather more data to suppress this warning.",
                     RuntimeWarning)

            else:
                row_opt_block_size = valid_sizes.min()
            opt_block_sizes.append(row_opt_block_size)
        return np.array(opt_block_sizes)

    @property
    def opt_iac_time(self):
        """The optimal integrated autocorrelation time."""
        criterion = self.block_sizes == self.opt_block_size[:, np.newaxis]
        opt_corr_times = []
        for row_idx, row_positions in enumerate(criterion):
            opt_corr_time = self.iac_times[row_idx, row_positions]
            opt_corr_times.append(opt_corr_time[0])

        return np.array(opt_corr_times)

    def __getitem__(self, index) -> OTFObject:
        """Collection interface"""
        return OTFObject(self.source_data[index],
                         min_num_blocks=self.min_num_blocks)

    def __len__(self) -> int:
        """Sized interface."""
        return self.source_data.shape[0]

    def __iter__(self) -> Iterator[OTFObject]:
        """Iterable interface."""
        for index in range(len(self)):
            yield self[index]


def on_the_fly_obj_data_update(obj_data: np.ndarray,
                               ext_obj_data: np.ndarray):
    """Updates the accumulated data of a reblocking with other accumulated.

    This function serves to update the accumulated data of a given
    reblocking with the accumulated data of a new, compatible reblocking.

    :param obj_data:
    :param ext_obj_data:
    :return:
    """
    # Shapes must be equal.
    assert obj_data.shape == ext_obj_data.shape

    accum_block_size = obj_data[BLOCK_SIZE_FIELD]
    extra_block_size = ext_obj_data[BLOCK_SIZE_FIELD]

    assert np.all(accum_block_size == extra_block_size)

    obj_data[MEANS_FIELD] += ext_obj_data[MEANS_FIELD]
    obj_data[MEANS_SQR_FIELD] += ext_obj_data[MEANS_SQR_FIELD]
    obj_data[NUM_BLOCKS_FIELD] += ext_obj_data[NUM_BLOCKS_FIELD]


def _on_the_fly_obj_data_from_obj_data_set(obj_data_set: T_ObjectDataSet):
    """Does a reblocking from a collection of reblocking data.

    :param obj_data_set:
    :return:
    """
    # Checks that the data has a valid dtype.
    obj_data_set: np.ndarray = np.asarray(obj_data_set)
    assert obj_data_set.dtype == otf_data_dtype

    # Check that the data has a consistent format.
    block_size_set = obj_data_set[BLOCK_SIZE_FIELD]
    assert np.all(np.diff(block_size_set, axis=0) == 0)

    # Calculate the accumulated extension. We need the accumulated data
    # corresponding to the greatest order of each one of the accumulated
    # data in the set.
    last_means_set = obj_data_set[MEANS_FIELD]
    accum_extension = _on_the_fly_obj_create(last_means_set)

    # Fix up the block sizes of the extension.
    last_block_size = obj_data_set[BLOCK_SIZE_FIELD][0]
    block_size_ext = accum_extension[BLOCK_SIZE_FIELD]

    # Fix up the shape of last_block_size...
    block_size_ext *= last_block_size[:, np.newaxis]

    # NOTE: The number of blocks of the extension array are not fixed up.
    return accum_extension[:, 1:]


def on_the_fly_extend_obj_data_set(obj_data_set: T_ObjectDataSet):
    """

    :param obj_data_set:
    :return:
    """
    # Checks that the data has a valid dtype.
    obj_data_set: np.ndarray = np.asarray(obj_data_set)

    assert obj_data_set.dtype == otf_data_dtype

    try:
        is_2d_array = False
        num_data, num_cols, max_order = obj_data_set.shape
    except ValueError:
        num_cols = 1
        is_2d_array = True
        num_data, max_order = obj_data_set.shape
        obj_data_set = obj_data_set[:, np.newaxis, :]

    data_total = on_the_fly_obj_data_init(max_order - 1, num_cols)

    dataset_last_order_data = []

    for data_index in range(num_data):
        ext_data: np.ndarray = obj_data_set[data_index]
        on_the_fly_obj_data_update(data_total, ext_data)
        # Take the data of the highest order for all the columns in
        # the current reblock.
        last_order_data: np.ndarray = ext_data[:, max_order - 1]
        dataset_last_order_data.append(last_order_data)

    data_ext = _on_the_fly_obj_data_from_obj_data_set(dataset_last_order_data)
    ext_data_set = np.hstack((data_total, data_ext))
    # NOTE: We could choose whether to extend the initial table set.

    if is_2d_array:
        return ext_data_set[0]

    return ext_data_set
