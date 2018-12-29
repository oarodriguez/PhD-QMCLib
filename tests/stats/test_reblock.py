import numpy as np

import my_research_libs.stats.reblock as reblock


def test_reblocking():
    """"""
    size_max_order = 16
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    block_analysis = reblock.Reblocking(data_sample, min_num_blocks=64)

    print(block_analysis.block_sizes)
    print(block_analysis.num_blocks)
    print(block_analysis.int_corr_times)
    print(block_analysis.corr_time_fit.params)
    print(block_analysis.source_data_eff_size)
    print(block_analysis.source_data_mean_eff_error)


def test_on_the_fly_reblocking():
    """"""
    size_max_order = 22
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    otf_accum = reblock.on_the_fly_reblocking_accum(data_sample,
                                                    min_num_blocks=32)
    dyn_reblocking = reblock.OnTheFlyReblocking(otf_accum)
    dyn_reblocking_vars = dyn_reblocking.vars
    print(dyn_reblocking_vars)

    reblocking = reblock.Reblocking(data_sample, min_num_blocks=32)
    reblocking_vars = reblocking.vars
    print(reblocking_vars)

    assert np.allclose(dyn_reblocking_vars, reblocking_vars)


def test_update_reblocking_accum():
    """"""
    size_max_order = 20
    min_num_blocks = 16
    data_size = 2 ** size_max_order

    data_sample = np.random.random_sample(data_size)
    max_order = reblock.get_reblocking_order(data_sample, min_num_blocks)
    batch_accum = reblock.init_reblocking_accum_array(max_order)

    num_accum = 2 ** 6
    for _ in range(num_accum):
        data_sample = np.random.random_sample(data_size)
        reblocking_accum = \
            reblock.on_the_fly_reblocking_accum(data_sample, min_num_blocks)
        reblock.update_reblocking_accum(batch_accum, reblocking_accum)

    otf_reblocking = reblock.OnTheFlyReblocking(batch_accum)
    print(otf_reblocking.int_corr_times)
    print(otf_reblocking.corr_time_fit.params)
    print(otf_reblocking.opt_int_corr_time)
