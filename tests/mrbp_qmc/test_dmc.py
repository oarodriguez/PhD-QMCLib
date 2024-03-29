from itertools import islice

import attr
import numpy as np
from numba.runtime import rtsys

import phd_qmclib.qmc_base.dmc as dmc_base
from phd_qmclib import mrbp_qmc
from phd_qmclib.qmc_base.jastrow import SysConfSlot
from phd_qmclib.qmc_exec import exec_logger

LATTICE_DEPTH = 0
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 4
BOSON_NUMBER = 16
SUPERCELL_SIZE = 16
TBF_CONTACT_CUTOFF = .25 * SUPERCELL_SIZE
NUM_DEFECTS = 4
DEFECT_MAGNITUDE = 0

# TODO: Improve this test.
model_spec = mrbp_qmc.Spec(lattice_depth=LATTICE_DEPTH,
                           lattice_ratio=LATTICE_RATIO,
                           interaction_strength=INTERACTION_STRENGTH,
                           boson_number=BOSON_NUMBER,
                           supercell_size=SUPERCELL_SIZE,
                           tbf_contact_cutoff=TBF_CONTACT_CUTOFF,
                           num_defects=NUM_DEFECTS,
                           defect_magnitude=DEFECT_MAGNITUDE)

move_spread = 0.25 * model_spec.well_width
num_steps = 4906 * 1
ini_sys_conf = model_spec.init_get_sys_conf()
vmc_sampling = \
    mrbp_qmc.vmc.Sampling(model_spec=model_spec,
                          move_spread=move_spread,
                          rng_seed=1)
vmc_ini_state = vmc_sampling.build_state(ini_sys_conf)

time_step = 1e-3
num_blocks = 4
num_time_steps_block = 512
burn_in_blocks = 2
target_num_walkers = 480
max_num_walkers = 512
ini_ref_energy = None
rng_seed = None
dmc_sampling = \
    mrbp_qmc.dmc.Sampling(model_spec,
                          time_step,
                          max_num_walkers=max_num_walkers,
                          target_num_walkers=target_num_walkers,
                          rng_seed=rng_seed)


def test_build_ini_state():
    """Test the build process of the initial DMC state."""
    ini_sys_conf_set = []
    ini_num_walkers = 128
    for idx in range(ini_num_walkers):
        sys_conf = model_spec.init_get_sys_conf()
        ini_sys_conf_set.append(sys_conf)
    ini_sys_conf_set = np.array(ini_sys_conf_set)

    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set,
                                             ini_ref_energy)
    dmc_ini_sys_confs = dmc_ini_state.confs[:ini_num_walkers]

    # Initial position data was copied correctly?
    assert np.allclose(dmc_ini_sys_confs[:, SysConfSlot.pos],
                       ini_sys_conf_set[:, SysConfSlot.pos])


def test_states():
    """Testing the DMC sampling."""
    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    ini_sys_conf_set = sys_conf_set[-100:]
    num_time_steps = num_blocks * num_time_steps_block
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)

    states = dmc_sampling.states(dmc_ini_state)
    dmc_sampling_slice = islice(states, num_time_steps)
    iter_enum: dmc_base.T_E_SIter = enumerate(dmc_sampling_slice)

    for i_, state in iter_enum:
        state_data = \
            (state.energy, state.weight, state.num_walkers,
             state.ref_energy, state.accum_energy)
        state_props = np.array(state_data, dtype=dmc_base.iter_props_dtype)
        print(i_, state_props)


def test_blocks():
    """Testing the DMC sampling."""
    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    ini_sys_conf_set = sys_conf_set[-100:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)
    sampling_blocks = \
        dmc_sampling.blocks(dmc_ini_state, num_time_steps_block,
                            burn_in_blocks)

    dmc_sampling_blocks: dmc_base.T_SBlocksIter = \
        islice(sampling_blocks, num_blocks)

    for block in dmc_sampling_blocks:
        state_props = block.iter_props
        print(state_props)


def test_state_data_blocks():
    """"""
    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    ini_sys_conf_set = sys_conf_set[-100:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)
    sampling_blocks = \
        dmc_sampling.state_data_blocks(dmc_ini_state, num_time_steps_block)

    dmc_sampling_blocks: dmc_base.T_SDBlocksIter = \
        islice(sampling_blocks, num_blocks)

    for block in dmc_sampling_blocks:
        # state_props = block.iter_props
        states_confs = block.confs
        print(states_confs)


def test_density_est():
    """Testing the calculation of the density."""

    exec_logger.info('Init sampling...')

    # TODO: Improve this test.
    exec_logger.info('Init VMC sampling...')

    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    exec_logger.info('Finished sampling...')

    ini_sys_conf_set = sys_conf_set[-128:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)

    num_bins = SUPERCELL_SIZE * 16

    density_est_spec = mrbp_qmc.dmc.DensityEstSpec(num_bins)
    dmc_density_sampling = attr.evolve(dmc_sampling,
                                       density_est_spec=density_est_spec)
    dmc_es_blocks = dmc_density_sampling.blocks(dmc_ini_state,
                                                num_time_steps_block,
                                                burn_in_blocks)

    es_blocks: dmc_base.T_SBlocksIter = \
        islice(dmc_es_blocks, num_blocks)

    exec_logger.info('Init DMC sampling...')

    for block in es_blocks:
        state_props = block.iter_props
        nw_iter = state_props.num_walkers
        iter_density = block.iter_density
        print(iter_density.shape)
        density_block_data = iter_density / nw_iter[:, np.newaxis, np.newaxis]
        print(density_block_data)
        # print(nw_iter)

    exec_logger.info('Finish DMC sampling.')


def test_dmc_est_sampling():
    """Testing the DMC sampling to evaluate several estimators."""

    exec_logger.info('Init sampling...')

    # TODO: Improve this test.
    exec_logger.info('Init VMC sampling...')

    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    exec_logger.info('Finished sampling...')

    ini_sys_conf_set = sys_conf_set[-128:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)

    num_modes = 2 * BOSON_NUMBER

    ssf_est_spec = mrbp_qmc.dmc.SSFEstSpec(num_modes)
    dmc_ssf_sampling = attr.evolve(dmc_sampling, ssf_est_spec=ssf_est_spec)
    dmc_es_blocks = dmc_ssf_sampling.blocks(dmc_ini_state,
                                            num_time_steps_block,
                                            burn_in_blocks)

    es_blocks: dmc_base.T_SBlocksIter = \
        islice(dmc_es_blocks, num_blocks)

    exec_logger.info('Init DMC sampling...')

    for block in es_blocks:
        state_props = block.iter_props
        nw_iter = state_props.num_walkers
        iter_ssf = block.iter_ssf
        # print(state_props)
        ssf_block_data = iter_ssf / nw_iter[:, np.newaxis, np.newaxis]
        print(ssf_block_data)
        # print(nw_iter)
        # This helps to catch memory leaks in numba compiled functions.
        print(rtsys.get_allocation_stats())
        print('---')

    exec_logger.info('Finish DMC sampling.')


if __name__ == '__main__':
    # test_states()
    test_dmc_est_sampling()
