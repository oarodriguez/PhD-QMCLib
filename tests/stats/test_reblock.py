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
    print(block_analysis.iac_times)
    print(block_analysis.iac_time_fit.params)
    print(block_analysis.source_data_eff_size)
    print(block_analysis.source_data_mean_eff_error)


def test_on_the_fly_reblocking():
    """"""
    size_max_order = 22
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    otf_data = reblock.on_the_fly_exec(data_sample)
    dyn_reblocking = reblock.OnTheFlyReblocking(otf_data, min_num_blocks=32)
    dyn_reblocking_vars = dyn_reblocking.vars
    print(dyn_reblocking_vars)

    reblocking = reblock.Reblocking(data_sample, min_num_blocks=32)
    reblocking_vars = reblocking.vars
    print(reblocking_vars)

    assert np.allclose(dyn_reblocking_vars, reblocking_vars)


def test_on_the_fly_data_update():
    """"""
    size_max_order = 20
    data_size = 2 ** size_max_order

    data_sample = np.random.random_sample(data_size)
    max_order = reblock.on_the_fly_proc_order(data_sample)
    reblocking_total = reblock.on_the_fly_data_init(max_order)

    num_accum = 2 ** 6
    for idx in range(num_accum):
        data_sample = np.random.random_sample(data_size)
        reblocking_data = \
            reblock.on_the_fly_exec(data_sample)
        reblock.on_the_fly_table_update(reblocking_total, reblocking_data)
        print(f'Completed reblock #{idx}')

    otf_reblocking = reblock.OnTheFlyReblocking(reblocking_total)
    print(otf_reblocking.iac_times)
    print(otf_reblocking.opt_iac_time)
    print(otf_reblocking.iac_time_fit.params)


def test_on_the_fly_extend_table_set():
    """"""
    size_max_order = 10
    data_size = 2 ** size_max_order

    num_accum = 2 ** 5
    reblock_dataset = []
    for _ in range(num_accum):
        data_sample = np.random.random_sample(data_size)
        reblock_data = \
            reblock.on_the_fly_exec(data_sample)
        reblock_dataset.append(reblock_data)

    reblock_total = reblock.on_the_fly_extend_table_set(reblock_dataset)
    print(reblock_total, reblock_total.shape)

    otf_reblocking = reblock.OnTheFlyReblocking(reblock_total)
    print(otf_reblocking.means)
    print(otf_reblocking.iac_times)


def test_on_the_fly_extend_table_set_from_tables():
    """"""
    size_max_order = 10
    data_size = 2 ** size_max_order

    num_accum = 2 ** 5
    reblock_dataset = []
    for _ in range(num_accum):
        data_sample = np.random.random_sample((data_size, 256))
        reblock_data = \
            reblock.on_the_fly_exec(data_sample)
        reblock_dataset.append(reblock_data)

    reblock_set = reblock.on_the_fly_extend_table_set(reblock_dataset)
    print(reblock_set.shape)

    reblock_total = reblock.OnTheFlyTable(reblock_set)
    print(reblock_total.block_sizes)
    print(reblock_total.source_data_size)
    print(reblock_total.source_data_mean)
    print(reblock_total.source_data_var)
    print(reblock_total.means)
    print(reblock_total.vars)
    print(reblock_total.iac_times)
    print(reblock_total.opt_block_size)
