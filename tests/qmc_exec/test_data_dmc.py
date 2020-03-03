import typing as t

import pytest
from numpy.random import random_sample

from my_research_libs.qmc_exec.data.dmc import PropBlocks, UnWeightedPropBlocks

# We will handle objects with a different number of blocks.
NUM_BLOCKS = 1024
NUM_BLOCKS_ = NUM_BLOCKS + 5
MIN_TOT = 100
MAX_TOT_INT = 200
totals = MAX_TOT_INT * random_sample((NUM_BLOCKS,)) + MIN_TOT
totals_ = MAX_TOT_INT * random_sample(NUM_BLOCKS_) + MIN_TOT

MIN_WEIGHT_TOT = 10
WEIGHT_TOT_INT = 20
weight_totals = \
    WEIGHT_TOT_INT * random_sample((NUM_BLOCKS,)) + MIN_WEIGHT_TOT
weight_totals_ = \
    WEIGHT_TOT_INT * random_sample(NUM_BLOCKS_) + MIN_WEIGHT_TOT

T_PropsBlocks = t.Union[PropBlocks, UnWeightedPropBlocks]


def describe_prop_blocks(prop_blocks: T_PropsBlocks):
    """Show info of a ``PropBlocks`` instance."""
    print(f"No. of blocks: {(len(prop_blocks)):d}")
    print(f"Mean: {prop_blocks.mean:.8G}")
    print(f"Mean error: {prop_blocks.mean_error:.8G}")
    print("---")


def test_prop_blocks():
    """Test creation of blocks data."""
    prop_blocks = PropBlocks(totals, weight_totals)
    num_blocks = len(prop_blocks)
    assert num_blocks == NUM_BLOCKS
    describe_prop_blocks(prop_blocks)

    prop_blocks = UnWeightedPropBlocks(totals)
    num_blocks = len(prop_blocks)
    assert num_blocks == NUM_BLOCKS
    describe_prop_blocks(prop_blocks)

    prop_blocks_ = UnWeightedPropBlocks(totals_)
    num_blocks = len(prop_blocks_)
    assert num_blocks == NUM_BLOCKS_
    describe_prop_blocks(prop_blocks_)


def test_prop_blocks_concat():
    """Test concatenation of blocks data."""
    prop_blocks = PropBlocks(totals, weight_totals)
    describe_prop_blocks(prop_blocks)

    # Catch expected error when trying to join with other type of
    # objects.
    with pytest.raises(TypeError):
        prop_blocks + 1
        # noinspection PyTypeChecker
        1 + prop_blocks

    # A simple concatenation of the object with itself.
    prop_blocks = prop_blocks + prop_blocks
    assert len(prop_blocks) == 2 * NUM_BLOCKS
    describe_prop_blocks(prop_blocks)

    # PropsBlocks with no explicit weight.
    uw_prop_blocks = UnWeightedPropBlocks(totals)
    describe_prop_blocks(uw_prop_blocks)

    # Catch expected error when trying to join with other type of
    # objects.
    with pytest.raises(TypeError):
        uw_prop_blocks + 1
        # noinspection PyTypeChecker
        1 + uw_prop_blocks
        uw_prop_blocks + prop_blocks
        prop_blocks + uw_prop_blocks

    # A simple concatenation of the object with itself.
    uw_prop_blocks = uw_prop_blocks + uw_prop_blocks
    assert len(uw_prop_blocks) == 2 * NUM_BLOCKS
    describe_prop_blocks(uw_prop_blocks)

    # PropsBlocks with no explicit weight.
    uw_prop_blocks_ = UnWeightedPropBlocks(totals_)
    describe_prop_blocks(uw_prop_blocks_)

    # Join objects with different number of blocks.
    uw_prop_blocks_ext = uw_prop_blocks + uw_prop_blocks_
    describe_prop_blocks(uw_prop_blocks_ext)
    uw_prop_blocks_ext = uw_prop_blocks_ + uw_prop_blocks
    describe_prop_blocks(uw_prop_blocks_ext)
