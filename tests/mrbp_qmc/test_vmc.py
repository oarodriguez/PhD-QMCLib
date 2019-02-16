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
ini_sys_conf = model_spec.init_get_sys_conf()
vmc_sampling = mrbp_qmc.vmc.Sampling(model_spec=model_spec,
                                     move_spread=move_spread,
                                     rng_seed=1)


def test_sampling():
    """

    :return:
    """
    # TODO: Improve this test.
    ar = 0
    num_steps = 4096 * 32
    sampling_states = vmc_sampling.states(ini_sys_conf)
    for cj_, data in enumerate(sampling_states):
        sys_conf, wfv, stat = data
        ar += stat
        if cj_ + 1 >= num_steps:
            break
    ar /= num_steps

    states_data = vmc_sampling.as_chain(num_steps, ini_sys_conf)
    sys_confs_set = states_data.confs
    sys_props_set = states_data.props
    ar_ = states_data.accept_rate

    move_stat_field = mrbp_qmc.vmc.StateProp.MOVE_STAT
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    assert (accepted / num_steps) == ar_

    # noinspection PyTypeChecker
    num_slots = len(SysConfSlot)
    assert sys_confs_set.shape == (num_steps, num_slots, BOSON_NUMBER)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
    pos_slot = SysConfSlot.pos

    ax = pyplot.gca()
    pos = sys_confs_set[:, pos_slot]
    ax.hist(pos.flatten(), bins=20 * SUPERCELL_SIZE)
    pyplot.show()
    print(sys_confs_set)


def test_batches():
    """Testing the generator of batches.

    :return:
    """
    # TODO: Improve this test.
    num_batches = 128 + 1
    num_steps_batch = 4096

    # Both samplings (in batches and as_chain) have a total number
    # of steps of ``num_batches * num_steps_batch``, but the first
    # batch will be discarded, so the effective number is
    # ``(num_batches - 1) * num_steps_batch``.
    num_steps = num_batches * num_steps_batch
    eff_num_steps = (num_batches - 1) * num_steps_batch

    sampling_batches = vmc_sampling.batches(num_steps_batch, ini_sys_conf)
    accepted = 0.
    for states_batch in islice(sampling_batches, 1, num_batches):
        accept_rate = states_batch.accept_rate
        accepted += accept_rate * num_steps_batch
    batches_accept_rate = accepted / eff_num_steps

    move_stat_field = mrbp_qmc.vmc.StateProp.MOVE_STAT
    states_data = vmc_sampling.as_chain(num_steps, ini_sys_conf)
    sys_props_set = states_data.props[num_steps_batch:]
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    chain_accept_rate = accepted / eff_num_steps

    # Both acceptance ratios should be equal.
    assert batches_accept_rate == chain_accept_rate


if __name__ == '__main__':
    test_sampling()
