from itertools import islice

import numpy as np
from matplotlib import pyplot

from my_research_libs import mrbp_qmc
from my_research_libs.qmc_base.jastrow import SysConfSlot

LATTICE_DEPTH = 100
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 1
BOSON_NUMBER = 16
SUPERCELL_SIZE = 16
TBF_CONTACT_CUTOFF = .25 * SUPERCELL_SIZE

model_spec = mrbp_qmc.Spec(lattice_depth=LATTICE_DEPTH,
                           lattice_ratio=LATTICE_RATIO,
                           interaction_strength=INTERACTION_STRENGTH,
                           boson_number=BOSON_NUMBER,
                           supercell_size=SUPERCELL_SIZE,
                           tbf_contact_cutoff=TBF_CONTACT_CUTOFF)

move_spread = 0.25 * model_spec.well_width
ssf_est_spec = mrbp_qmc.vmc.SSFEstSpec(num_modes=BOSON_NUMBER)
vmc_sampling = mrbp_qmc.vmc.Sampling(model_spec=model_spec,
                                     move_spread=move_spread,
                                     rng_seed=1,
                                     ssf_est_spec=ssf_est_spec)
ini_sys_conf = model_spec.init_get_sys_conf()
ini_state = vmc_sampling.build_state(ini_sys_conf)


def test_sampling():
    """

    :return:
    """
    # TODO: Improve this test.
    num_blocks = 8
    num_steps_block = 128
    blocks = vmc_sampling.blocks(num_steps_block, ini_state)
    block_idx = 0
    energy_data = []
    for block_data in blocks:
        iter_props = block_data.iter_props
        energy_block = iter_props.energy
        energy_data.append(energy_block)
        if block_idx + 1 >= num_blocks:
            break
        block_idx += 1
    energy_data = np.hstack(energy_data)

    print(energy_data.mean(), energy_data.var(ddof=1))


def test_confs_props_blocks():
    """

    :return:
    """
    # TODO: Improve this test.
    ar = 0
    num_steps = 4096 * 1
    sampling_states = vmc_sampling.states(ini_state)
    for cj_, data in enumerate(sampling_states):
        sys_conf, wfv, stat = data
        ar += stat
        if cj_ + 1 >= num_steps:
            break
    ar /= num_steps

    states_data = vmc_sampling.as_chain(num_steps, ini_state)
    sys_props_set = states_data.props
    ar_ = states_data.accept_rate
    assert ar == ar_

    move_stat_field = mrbp_qmc.vmc.StateProp.MOVE_STAT
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    assert (accepted / num_steps) == ar_

    # noinspection PyTypeChecker
    num_slots = len(SysConfSlot)
    sys_confs_set = states_data.confs
    assert sys_confs_set.shape == (num_steps, num_slots, BOSON_NUMBER)

    print(f"Sampling acceptance rate: {ar:.5g}")
    pos_slot = SysConfSlot.pos

    ax = pyplot.gca()
    pos = sys_confs_set[:, pos_slot]
    ax.hist(pos.flatten(), bins=20 * SUPERCELL_SIZE)
    pyplot.show()
    print(sys_confs_set)


def test_blocks():
    """Testing the generator of blocks.

    :return:
    """
    # TODO: Improve this test.
    num_blocks = 2
    num_steps_block = 4096

    # Both samplings (in blocks and as_chain) have a total number
    # of steps of ``num_blocks * num_steps_block``, but the first
    # block will be discarded, so the effective number is
    # ``(num_blocks - 1) * num_steps_block``.
    num_steps = num_blocks * num_steps_block
    eff_num_steps = (num_blocks - 1) * num_steps_block

    sampling_blocks = vmc_sampling.blocks(num_steps_block, ini_state)
    accepted = 0.
    for states_block in islice(sampling_blocks, 1, num_blocks):
        accept_rate = states_block.accept_rate
        accepted += accept_rate * num_steps_block
    blocks_accept_rate = accepted / eff_num_steps

    move_stat_field = mrbp_qmc.vmc.StateProp.MOVE_STAT
    states_data = vmc_sampling.as_chain(num_steps, ini_state)
    sys_props_set = states_data.props[num_steps_block:]
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    chain_accept_rate = accepted / eff_num_steps

    # Both acceptance ratios should be equal.
    assert blocks_accept_rate == chain_accept_rate


if __name__ == '__main__':
    test_sampling()
