from itertools import islice

import attr
import numpy as np
from numba.runtime import rtsys

import my_research_libs.qmc_base.dmc as dmc_base
from my_research_libs import mrbp_qmc
from my_research_libs.qmc_base.jastrow import SysConfSlot
from my_research_libs.qmc_exec import exec_logger

LATTICE_DEPTH = 0
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 4
BOSON_NUMBER = 16
SUPERCELL_SIZE = 16
TBF_CONTACT_CUTOFF = .25 * SUPERCELL_SIZE

# TODO: Improve this test.
model_spec = mrbp_qmc.Spec(lattice_depth=LATTICE_DEPTH,
                           lattice_ratio=LATTICE_RATIO,
                           interaction_strength=INTERACTION_STRENGTH,
                           boson_number=BOSON_NUMBER,
                           supercell_size=SUPERCELL_SIZE,
                           tbf_contact_cutoff=TBF_CONTACT_CUTOFF)

move_spread = 0.25 * model_spec.well_width
num_steps = 4906 * 1
ini_sys_conf = model_spec.init_get_sys_conf()
vmc_sampling = \
    mrbp_qmc.vmc.Sampling(model_spec=model_spec,
                          move_spread=move_spread,
                          rng_seed=1)
vmc_ini_state = vmc_sampling.build_state(ini_sys_conf)

time_step = 1e-3
num_batches = 4
num_time_steps_batch = 512
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
    num_time_steps = num_batches * num_time_steps_batch
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


def test_batches():
    """Testing the DMC sampling."""
    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    ini_sys_conf_set = sys_conf_set[-100:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)
    sampling_batches = \
        dmc_sampling.batches(dmc_ini_state, num_time_steps_batch)

    dmc_sampling_batches: dmc_base.T_SBatchesIter = \
        islice(sampling_batches, num_batches)

    for batch in dmc_sampling_batches:
        state_props = batch.iter_props
        print(state_props)


def test_confs_props_batches():
    """"""
    vmc_chain_data = vmc_sampling.as_chain(num_steps, vmc_ini_state)
    sys_conf_set = vmc_chain_data.confs
    ar_ = vmc_chain_data.accept_rate
    print(f"Acceptance ratio: {ar_:.5g}")

    ini_sys_conf_set = sys_conf_set[-100:]
    dmc_ini_state = dmc_sampling.build_state(ini_sys_conf_set, ini_ref_energy)
    sampling_batches = \
        dmc_sampling.confs_props_batches(dmc_ini_state, num_time_steps_batch)

    dmc_sampling_batches: dmc_base.T_SCPBatchesIter = \
        islice(sampling_batches, num_batches)

    for batch in dmc_sampling_batches:
        # state_props = batch.iter_props
        states_confs = batch.states_confs
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
    dmc_es_batches = dmc_density_sampling.batches(dmc_ini_state,
                                                  num_time_steps_batch)

    es_batches: dmc_base.T_SBatchesIter = \
        islice(dmc_es_batches, num_batches)

    exec_logger.info('Init DMC sampling...')

    for batch in es_batches:
        state_props = batch.iter_props
        nw_iter = state_props[dmc_base.IterProp.NUM_WALKERS]
        iter_density = batch.iter_density
        print(iter_density.shape)
        density_batch_data = iter_density / nw_iter[:, np.newaxis, np.newaxis]
        print(density_batch_data)
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
    dmc_es_batches = dmc_ssf_sampling.batches(dmc_ini_state,
                                              num_time_steps_batch)

    es_batches: dmc_base.T_SBatchesIter = \
        islice(dmc_es_batches, num_batches)

    exec_logger.info('Init DMC sampling...')

    for batch in es_batches:
        state_props = batch.iter_props
        nw_iter = state_props[dmc_base.IterProp.NUM_WALKERS]
        iter_ssf = batch.iter_ssf
        # print(state_props)
        ssf_batch_data = iter_ssf / nw_iter[:, np.newaxis, np.newaxis]
        print(ssf_batch_data)
        # print(nw_iter)
        # This helps to catch memory leaks in numba compiled functions.
        print(rtsys.get_allocation_stats())
        print('---')

    exec_logger.info('Finish DMC sampling.')


if __name__ == '__main__':
    # test_states()
    test_dmc_est_sampling()