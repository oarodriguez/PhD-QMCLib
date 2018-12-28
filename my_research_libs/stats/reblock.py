import typing as t
from abc import ABCMeta, abstractmethod
from enum import Enum, unique
from math import ceil, floor, log, log2, sqrt

import attr
import numba as nb
import numpy as np
from cached_property import cached_property
from scipy.optimize import curve_fit

__all__ = [
    'CorrFitParams',
    'CorrTimeFit',
    'Reblocking',
    'ReblockingField',
    'get_reblocking_order',
    'init_reblocking_accum_array',
    'on_the_fly_reblocking_accum'
]


class CorrFitParams(t.NamedTuple):
    """"""
    int_time: float
    exp_time: float
    const: float


class CorrFitErrors(t.NamedTuple):
    """"""
    int_time: float
    exp_time: float
    const: float


@attr.s(auto_attribs=True, frozen=True)
class CorrTimeFit:
    """"""
    times: np.ndarray
    int_corr_times: np.ndarray
    results: t.Tuple = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        """Post initialization stage."""
        self_fit = curve_fit(self.__func__, self.times, self.int_corr_times)
        super().__setattr__('results', self_fit)

    @staticmethod
    def __func__(times, int_time, exp_time, const):
        """A function to approximate the correlation times..

        :param times: The size of the block.
        :param int_time:  The integrated correlation time.
        :param exp_time:  The exponential correlation time.
        :param const: The fitting constant.
        :return:
        """
        return int_time - const * np.exp(-times / exp_time)

    def __call__(self, times):
        """Callable interface."""
        return self.__func__(times, *self.params)

    @property
    def params(self):
        """The fit parameters."""
        return CorrFitParams(*self.results[0])

    @property
    def cov_matrix(self):
        """Covariance matrix of the fit."""
        return self.results[1]

    @property
    def errors(self):
        """The errors of the fit parameters."""
        return CorrFitErrors(*np.sqrt(np.diag(self.cov_matrix)))

    @property
    def int_time(self):
        """Integrated correlation time."""
        return self.params.int_time

    @property
    def exp_time(self):
        """Exponential correlation time."""
        return self.params.exp_time


@attr.s(auto_attribs=True, frozen=True)
class BlockedData:
    """"""
    num_blocks: int
    block_size: int
    data: np.ndarray


class ReblockingBase(metaclass=ABCMeta):
    """"""
    #: The data to analyze.
    source_data: np.ndarray

    #: Variance delta degrees of freedom.
    var_ddof: int

    @property
    @abstractmethod
    def source_data_size(self) -> int:
        """The size of the source data."""
        pass

    @cached_property
    def source_data_mean(self):
        """The mean of the source data."""
        return self.source_data.mean(axis=0)

    @cached_property
    def source_data_var(self):
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
    def int_corr_times(self):
        """Integrated correlation times for each of the block sizes."""
        self_vars = self.vars
        block_sizes = self.block_sizes
        data_var = self.source_data_var

        # Correlation times.
        return 0.5 * block_sizes * self_vars / data_var

    @property
    def opt_block_size(self):
        """The optimal block size."""
        block_sizes = self.block_sizes
        data_size = self.source_data_size
        int_corr_times = self.int_corr_times
        # B^3 > 2N * (2 \tau)^2
        criterion = block_sizes ** 3 > 8 * data_size * int_corr_times ** 2
        opt_block_size = block_sizes[criterion].min()
        return opt_block_size

    @property
    def opt_int_corr_time(self):
        """The optimal integrated correlation time."""
        criterion = self.block_sizes == self.opt_block_size
        opt_corr_time = self.int_corr_times[criterion]
        return opt_corr_time[0]

    @property
    def source_data_eff_size(self):
        """Returns the effective size of the source data.

        The calculation takes into account the temporal correlations
        between the data to get the effective size of the statistical
        sample.
        """
        data_size = self.source_data_size
        # NOTE: Should we return an int or a float?
        return data_size / (2 * self.opt_int_corr_time)

    @property
    def source_data_mean_eff_error(self):
        """The effective error of the mean of the data.

        This value takes into account the temporal correlations between
        the data to get the effective error of the mean of the statistical
        sample.
        """
        eff_data_size = self.source_data_eff_size
        return sqrt(self.source_data_var / eff_data_size)

    @property
    def corr_time_fit(self):
        """Fit of the integrated correlation time."""
        return CorrTimeFit(self.block_sizes, self.int_corr_times)


@attr.s(auto_attribs=True, frozen=True)
class Reblocking(ReblockingBase):
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

        if self.var_ddof < 0:
            raise ValueError('delta degrees of freedom must be a positive '
                             'integer')

        min_num_blocks = self.min_num_blocks
        less_than_min = self.num_blocks < min_num_blocks
        if np.count_nonzero(less_than_min):
            raise ValueError(f'the minimum number of data blocks for the '
                             f'analysis is {min_num_blocks} is not respected '
                             'with the given block_sizes')

    @property
    def source_data_size(self):
        """The size of the source data."""
        return len(self.source_data)

    @cached_property
    def block_sizes(self):
        """The sizes of the blocks used in the reblocking."""
        min_num_blocks = self.min_num_blocks
        data_size = self.source_data_size
        max_num_blocks = int(floor(log2(data_size)))
        min_num_blocks = int(ceil(log2(min_num_blocks)))

        # Powers of 2 for the block sizes.
        block_sizes = 1 << np.arange(max_num_blocks - min_num_blocks + 1)
        return block_sizes.astype(np.int64)

    @cached_property
    def num_blocks(self):
        """The number of blocks for each block size."""
        data_size = self.source_data_size
        num_blocks = data_size // np.array(self.block_sizes, dtype=np.int64)
        return num_blocks.astype(np.int64)

    @cached_property
    def data(self) -> t.List[BlockedData]:
        """The source data reshaped in blocks."""
        self_data = self.source_data
        block_sizes = self.block_sizes
        data_size = self.source_data_size

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
        blocked_data = self.data
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
        blocked_data = self.data
        data_vars = []
        for block in blocked_data:
            shaped_data = block.data
            data_mean = shaped_data.mean(axis=1).var(axis=0, ddof=var_ddof)
            data_vars.append(data_mean)
        return np.array(data_vars)


@unique
class ReblockingField(str, Enum):
    """Fields of the reblocking array."""
    BLOCK_SIZE = 'BLOCK_SIZE'
    MEANS_SUM = 'MEANS_SUM'
    MEANS_SQR_SUM = 'MEANS_SQR_SUM'
    NUM_BLOCKS = 'NUM_BLOCKS'


BLOCK_SIZE_FIELD = ReblockingField.BLOCK_SIZE.value
NUM_BLOCKS_FIELD = ReblockingField.NUM_BLOCKS.value
MEANS_SQR_SUM_FIELD = ReblockingField.MEANS_SQR_SUM.value
MEANS_SUM_FIELD = ReblockingField.MEANS_SUM.value

accum_array_dtype = np.dtype([
    (BLOCK_SIZE_FIELD, np.int64),
    (MEANS_SUM_FIELD, np.float64),
    (MEANS_SQR_SUM_FIELD, np.float64),
    (NUM_BLOCKS_FIELD, np.int64)
])


@nb.njit
def get_reblocking_order(source_data: np.ndarray,
                         min_num_blocks: int = 1,
                         max_order: int = None):
    """Estimates the maximum order of an on-the-fly reblocking.

    The maximum order determines the size of the output array of the
    on-the-fly reblocking process.

    :param source_data:
    :param min_num_blocks:
    :param max_order:
    :return:
    """
    data_length = source_data.shape[0]
    max_num_blocks = int(floor(log(data_length) / log(2)))
    min_num_blocks = int(ceil(log(min_num_blocks) / log(2)))

    if max_order is None:
        max_order = max_num_blocks
    order = max_num_blocks - min_num_blocks
    return min(order, max_order)


@nb.njit
def init_reblocking_accum_array(max_order: int) -> np.ndarray:
    """Initializes the reblocking array.

    :param max_order:
    :return:
    """
    accum_array = np.zeros(max_order + 1, dtype=accum_array_dtype)
    block_size_array = accum_array[BLOCK_SIZE_FIELD]
    for order in range(max_order + 1):
        block_size = 1 << order
        block_size_array[order] = block_size

    return accum_array


@nb.njit
def on_the_fly_reblocking_accum(source_data: np.ndarray,
                                min_num_blocks: int = 2,
                                max_order: int = None):
    """Realizes a reblocking analysis at increasing levels.

    This function calculates a reblocking analysis at increasing levels.
    Each subsequent level takes twice the number of elements of
    source_data than the previous to construct a block and to calculate
    the averages.

    :param source_data:
    :param min_num_blocks:
    :param max_order:
    :return:
    """
    max_order = get_reblocking_order(source_data, min_num_blocks, max_order)

    # Initialize arrays.
    accum_array = init_reblocking_accum_array(max_order)
    means_array = np.zeros((max_order + 1, 2), dtype=np.float64)

    block_size_array = accum_array[BLOCK_SIZE_FIELD]
    means_sum_array = accum_array[MEANS_SUM_FIELD]
    means_sqr_sum_array = accum_array[MEANS_SQR_SUM_FIELD]
    num_blocks_array = accum_array[NUM_BLOCKS_FIELD]

    # The size of the next block is twice the previous.
    ini_order = order = 0
    ini_block_size = block_size = block_size_array[order]
    ini_next_block_size = next_block_size = block_size << 1

    data_length = source_data.shape[0]
    for index in range(data_length):
        #
        data_mean = source_data[index]
        means_sum_array[order] += data_mean
        means_sqr_sum_array[order] += data_mean ** 2
        num_blocks_array[order] += 1

        block_index = index
        mean_index = block_index % 2
        means_array[order, mean_index] = data_mean

        while True:
            #
            if order >= max_order:
                #
                break

            if not (index + 1) % next_block_size:
                #
                order += 1
                block_size = block_size << 1

                data_mean = means_array[order - 1].mean()
                means_sum_array[order] += data_mean
                means_sqr_sum_array[order] += data_mean ** 2
                num_blocks_array[order] += 1

                block_index = (index + 1) // block_size - 1
                mean_index = block_index % 2
                means_array[order, mean_index] = data_mean

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

    return accum_array


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
class OnTheFlyReblocking(ReblockingBase):
    """Realizes a reblocking analysis at increasing levels."""

    #: The data to analyze.
    source_data: np.ndarray

    #: Variance delta degrees of freedom.
    var_ddof: int = 1

    def __attrs_post_init__(self):
        """"""
        # NOTE: Allow only 1d arrays by now.
        assert len(self.source_data.shape) == 1

        # Only allow reblocking accumulated values.
        assert self.source_data.dtype == accum_array_dtype

    @cached_property
    def source_data_size(self):
        """"""
        return self.num_blocks[0]

    @cached_property
    def source_data_mean(self):
        """The mean of the source data."""
        return self.means[0]

    @cached_property
    def source_data_var(self):
        """The variance of the source data."""
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
        means_sum = self.source_data[MEANS_SUM_FIELD]
        return means_sum / self.num_blocks

    @cached_property
    def vars(self):
        """"""
        num_blocks = self.num_blocks
        means_sqr_sum = self.source_data[MEANS_SQR_SUM_FIELD]
        means_sqr = means_sqr_sum / num_blocks
        ddof_num_blocks = num_blocks - self.var_ddof
        return num_blocks * (means_sqr - self.means ** 2) / ddof_num_blocks
