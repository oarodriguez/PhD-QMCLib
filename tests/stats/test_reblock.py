import numpy as np

from my_research_libs.stats.reblock import OnTheFlyReblocking, Reblocking


def test_stats():
    """"""
    size_max_order = 16
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    block_analysis = Reblocking(data_sample, min_num_blocks=32)

    print(block_analysis.block_sizes)
    print(block_analysis.num_blocks)
    print(block_analysis.int_corr_times)
    print(block_analysis.corr_time_fit.params)
    print(block_analysis.source_data_eff_size)
    print(block_analysis.source_data_mean_eff_error)


def test_dynamic_reblocking():
    """"""
    size_max_order = 22
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    dyn_reblocking = OnTheFlyReblocking(data_sample)
    dyn_reblocking_vars = dyn_reblocking.vars
    print(dyn_reblocking_vars)

    reblocking = Reblocking(data_sample, min_num_blocks=2)
    reblocking_vars = reblocking.vars
    print(reblocking_vars)

    assert np.allclose(dyn_reblocking_vars, reblocking_vars)
