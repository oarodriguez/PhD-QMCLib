import typing as t
from math import ceil, floor, log2

import attr
import numpy as np
from cached_property import cached_property
from scipy.optimize import curve_fit


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
        self_fit = curve_fit(self, self.times, self.int_corr_times)
        super().__setattr__('results', self_fit)

    def __call__(self, times, int_time, exp_time, const):
        """A function to approximate the correlation times..

        :param times: The size of the block.
        :param int_time:  The integrated correlation time.
        :param exp_time:  The exponential correlation time.
        :param const: The fitting constant.
        :return:
        """
        return int_time - const * np.exp(-times / exp_time)

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


@attr.s(auto_attribs=True, frozen=True)
class Reblocking:
    """Realizes a blocking/binning analysis of serially correlated data."""

    # The data to analyze.
    source_data: np.ndarray

    #: The sizes of the blocks used to estimate the correlation
    #: length and integrated time.
    block_sizes: t.Optional[t.Union[np.ndarray, t.Sequence[int]]] = None

    #: Minimum size
    min_num_blocks: int = 2

    #: Variance delta degrees of freedom.
    var_ddof: int = attr.ib(default=1, init=False)

    #:
    fit_min_block_size: int = 1

    def __attrs_post_init__(self):
        """Post initialization stage."""
        block_sizes = self.block_sizes
        min_num_blocks = self.min_num_blocks
        if block_sizes is not None:
            block_sizes = np.asarray(block_sizes)
        else:
            len_data = len(self.source_data)
            max_num_blocks = int(floor(log2(len_data)))
            min_num_blocks = int(ceil(log2(min_num_blocks)))
            block_sizes = 2 ** np.arange(max_num_blocks - min_num_blocks + 1)

        # Update attribute.
        super().__setattr__('block_sizes', block_sizes.astype(np.int64))

        # NOTE: Allow only 1d arrays by now.
        assert len(block_sizes.shape) == 1
        assert len(self.source_data.shape) == 1

        if self.var_ddof < 0:
            raise ValueError('delta degrees of freedom must be a positive '
                             'integer')

        less_than_min = self.num_blocks < min_num_blocks
        if np.count_nonzero(less_than_min):
            raise ValueError(f'the minimum number of data blocks for the '
                             f'analysis is {min_num_blocks} is not respected '
                             'with the given block_sizes')

    @cached_property
    def num_blocks(self):
        """The number of blocks for each block size."""
        data_len, *_ = self.source_data.shape
        num_blocks = data_len // np.array(self.block_sizes, dtype=np.int64)
        return num_blocks.astype(np.int64)

    @cached_property
    def blocked_data(self) -> t.List[BlockedData]:
        """The source data reshaped in blocks."""
        self_data = self.source_data
        block_sizes = self.block_sizes
        len_data, *rem_shape = self_data.shape

        blocked_data = []
        for size in block_sizes:
            num_blocks = len_data // size
            eff_size = num_blocks * size
            data_shape = (num_blocks, size) + tuple(rem_shape)
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

    @property
    def errors(self):
        """Errors of the mean for each of the block sizes."""
        return np.sqrt(self.vars / self.num_blocks)

    @property
    def int_corr_times(self):
        """Integrated correlation times for each of the block sizes."""
        self_vars = self.vars
        var_ddof = self.var_ddof
        block_sizes = self.block_sizes
        data_var = self.source_data.var(axis=0, ddof=var_ddof)

        # Reshape block sizes to proceed.
        num_sizes, *ext_shape = self_vars.shape
        bdc_shape = (num_sizes,) + (1,) * len(ext_shape)
        block_sizes = block_sizes.reshape(bdc_shape)

        # Correlation times.
        return 0.5 * block_sizes * self_vars / data_var

    @property
    def opt_block_size(self):
        """The optimal block size."""
        block_sizes = self.block_sizes
        len_data = len(self.source_data)
        int_corr_times = self.int_corr_times
        criterion = block_sizes ** 3 > 2 * len_data * int_corr_times ** 2
        opt_block_size = block_sizes[criterion].min()
        return opt_block_size

    @property
    def opt_int_corr_time(self):
        """The optimal integrated correlation time."""
        criterion = self.block_sizes == self.opt_block_size
        opt_corr_time = self.int_corr_times[criterion]
        return opt_corr_time[0]

    @property
    def corr_time_fit(self):
        """Fit of the integrated correlation time."""
        return CorrTimeFit(self.block_sizes, self.int_corr_times)
