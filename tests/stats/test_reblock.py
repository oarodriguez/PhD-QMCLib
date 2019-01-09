import numpy as np
import pytest

import my_research_libs.stats.reblock as rb


def test_object():
    """"""
    size_max_order = 16
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    block_analysis = rb.Object(data_sample, min_num_blocks=64)

    print(block_analysis.block_sizes)
    print(block_analysis.num_blocks)
    print(block_analysis.iac_times)
    print(block_analysis.iac_time_fit.params)
    print(block_analysis.eff_size)
    print(block_analysis.mean_eff_error)


def test_on_the_fly_object():
    """"""
    size_max_order = 22
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    otf_data = rb.on_the_fly_obj_create(data_sample)
    otf_s_reblock = rb.OTFObject(otf_data, min_num_blocks=32)
    otf_reblock_vars = otf_s_reblock.vars
    print(otf_reblock_vars)
    print(otf_s_reblock.means)
    print(otf_s_reblock.iac_times)
    print(otf_s_reblock.block_sizes)
    print(otf_s_reblock.opt_iac_time)
    print(otf_s_reblock.opt_block_size)

    s_reblock = rb.Object(data_sample, min_num_blocks=32)
    s_reblock_vars = s_reblock.vars
    print(s_reblock_vars)

    assert np.allclose(otf_reblock_vars, s_reblock_vars)


def test_opt_block_size_warning():
    """"""
    # We need a small number of blocks to raise this warning.
    size_max_order = 1
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    with pytest.warns(RuntimeWarning):
        otf_s_reblock = rb.OTFObject.from_non_obj_data(data_sample)
        otf_reblock_vars = otf_s_reblock.vars
        print(otf_s_reblock.opt_iac_time)
        print(otf_s_reblock.opt_block_size)

    assert otf_s_reblock.opt_block_size == otf_s_reblock.block_sizes.max()

    s_reblock = rb.Object(data_sample)
    s_reblock_vars = s_reblock.vars
    print(s_reblock_vars)

    assert np.allclose(otf_reblock_vars, s_reblock_vars)


def test_on_the_fly_fails():
    """"""
    size_max_order = 22
    data_size = 2 ** size_max_order
    data_sample = np.random.random_sample(data_size)

    otf_data_base = rb.on_the_fly_obj_create(data_sample)
    reblock_base = rb.OTFObject(otf_data_base,
                                min_num_blocks=32)
    print(reblock_base.means)

    otf_data = otf_data_base[np.newaxis, :]
    reblock_mod = rb.OTFSet(otf_data, min_num_blocks=32)
    print(reblock_mod.means)

    with pytest.raises(ValueError):
        # 3d array must raise ValueError
        otf_data = otf_data_base[np.newaxis, np.newaxis, :]
        rb.OTFSet(otf_data, min_num_blocks=32)

    reblock_mod = rb.Object(data_sample, min_num_blocks=32)
    reblocking_vars = reblock_mod.vars
    print(reblocking_vars)


def test_on_the_fly_obj_data_update():
    """"""
    size_max_order = 20
    data_size = 2 ** size_max_order

    data_sample = np.random.random_sample(data_size)
    max_order = rb.on_the_fly_obj_data_order(data_sample)
    reblocking_total = rb.on_the_fly_obj_data_init(max_order)

    num_accum = 2 ** 6
    for idx in range(num_accum):
        data_sample = np.random.random_sample(data_size)
        reblocking_data = \
            rb.on_the_fly_obj_create(data_sample)
        rb.on_the_fly_obj_data_update(reblocking_total, reblocking_data)
        print(f'Completed reblock #{idx}')

    otf_reblocking = rb.OTFObject(reblocking_total)
    print(otf_reblocking.iac_times)
    print(otf_reblocking.opt_iac_time)
    print(otf_reblocking.iac_time_fit.params)


def test_on_the_fly_obj_from_obj_data_set():
    """"""
    size_max_order = 10
    data_size = 2 ** size_max_order

    num_accum = 2 ** 5
    reblock_dataset = []
    for _ in range(num_accum):
        data_sample = np.random.random_sample(data_size)
        reblock_data = \
            rb.on_the_fly_obj_create(data_sample)
        reblock_dataset.append(reblock_data)

    otf_reblocking = rb.OTFObject.from_obj_data_set(reblock_dataset)
    print(otf_reblocking, otf_reblocking.source_data.shape)
    print(otf_reblocking.means)
    print(otf_reblocking.iac_times)


def test_on_the_fly_set_from_obj_data_set():
    """"""
    size_max_order = 10
    data_size = 2 ** size_max_order

    num_accum = 2 ** 5
    reblock_dataset = []
    for _ in range(num_accum):
        data_sample = np.random.random_sample((data_size, 256))
        reblock_data = \
            rb.on_the_fly_obj_create(data_sample)
        reblock_dataset.append(reblock_data)

    reblock_set = rb.OTFSet.from_obj_data_set(reblock_dataset)
    print(reblock_set.source_data.shape)

    print(reblock_set.block_sizes)
    print(reblock_set.size)
    print(reblock_set.mean)
    print(reblock_set.var)
    print(reblock_set.means)
    print(reblock_set.vars)
    print(reblock_set.iac_times)
    print(reblock_set.opt_iac_time)
    print(reblock_set.opt_block_size)
