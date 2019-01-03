from itertools import islice

import attr
import numpy as np
import pytest
from matplotlib import pyplot

import my_research_libs.qmc_base.dmc as dmc_base
from my_research_libs.multirods_qmc import bloch_phonon

LATTICE_DEPTH = 100
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 1
BOSON_NUMBER = 100
SUPERCELL_SIZE = 100
TBF_CONTACT_CUTOFF = .25 * SUPERCELL_SIZE

# Well-formed parameters.
BASE_SPEC_ITEMS = dict(lattice_depth=LATTICE_DEPTH,
                       lattice_ratio=LATTICE_RATIO,
                       interaction_strength=INTERACTION_STRENGTH,
                       boson_number=BOSON_NUMBER,
                       supercell_size=SUPERCELL_SIZE,
                       tbf_contact_cutoff=TBF_CONTACT_CUTOFF)


def test_init():
    """"""
    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    print(repr(model_spec))
    print(attr.asdict(model_spec))


def test_update_params():
    """

    :return:
    """
    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    with pytest.raises(AttributeError):
        # Extra parameter. This will fail.
        new_params = dict(BASE_SPEC_ITEMS, extra_param=True)
        for name, value in new_params.items():
            setattr(model_spec, name, value)


def test_qmc_funcs():
    """"""

    # We have an ideal system...
    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    core_funcs = bloch_phonon.CoreFuncs()

    # Generate a random configuration, pick the model parameters.
    sys_conf = model_spec.init_get_sys_conf()
    cfc_spec = model_spec.cfc_spec_nt

    # Testing a scalar function with own arguments
    energy_func = core_funcs.energy
    energy_v = energy_func(sys_conf, cfc_spec)

    # Testing an array function with no own arguments
    drift = core_funcs.drift
    out_sys_conf = drift(sys_conf, cfc_spec)

    epp = energy_v / BOSON_NUMBER
    print("The energy per particle is: {:.6g}".format(epp))

    drift_values = out_sys_conf[model_spec.sys_conf_slots.drift, :]
    print("The drift is: {}".format(drift_values))

    # Testing that the array function do not modify its inputs
    in_pos_values = sys_conf[model_spec.sys_conf_slots.pos, :]
    out_pos_values = out_sys_conf[model_spec.sys_conf_slots.pos, :]
    assert np.alltrue(out_pos_values == in_pos_values)

    with pytest.raises(AssertionError):
        # Testing that the array function modified the output array
        # where expected.
        in_pos_values = sys_conf[model_spec.sys_conf_slots.drift, :]
        out_pos_values = out_sys_conf[model_spec.sys_conf_slots.drift, :]
        assert np.alltrue(out_pos_values == in_pos_values)


def test_vmc():
    """

    :return:
    """
    boson_number = 10
    supercell_size = 10
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)
    move_spread = 0.25 * model_spec.well_width
    num_steps = 4096 * 128
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)
    ar = 0
    for cj_, data in enumerate(vmc_sampling):
        sys_conf, wfv, stat = data
        ar += stat
        if cj_ + 1 >= num_steps:
            break
    ar /= num_steps

    states_data = vmc_sampling.as_chain(num_steps)
    sys_confs_set, sys_props_set, ar_ = states_data

    move_stat_field = bloch_phonon.vmc.StateProp.MOVE_STAT
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    assert (accepted / num_steps) == ar_

    num_slots = len(model_spec.sys_conf_slots)
    assert sys_confs_set.shape == (num_steps, num_slots, boson_number)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
    pos_slot = model_spec.sys_conf_slots.pos

    ax = pyplot.gca()
    pos = sys_confs_set[:, pos_slot]
    ax.hist(pos.flatten(), bins=20 * supercell_size)
    pyplot.show()
    print(sys_confs_set)


def test_vmc_batches():
    """Testing the generator of batches.

    :return:
    """
    boson_number = 10
    supercell_size = 10
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)
    move_spread = 0.25 * model_spec.well_width
    num_batches = 128 + 1
    num_steps_batch = 4096
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)

    # Both samplings (in batches and as_chain) have a total number
    # of steps of ``num_batches * num_steps_batch``, but the first
    # batch will be discarded, so the effective number is
    # ``(num_batches - 1) * num_steps_batch``.
    num_steps = num_batches * num_steps_batch
    eff_num_steps = (num_batches - 1) * num_steps_batch

    sampling_batches = vmc_sampling.batches(num_steps_batch)
    accepted = 0.
    for states_batch in islice(sampling_batches, 1, num_batches):
        accept_rate = states_batch.accept_rate
        accepted += accept_rate * num_steps_batch
    batches_accept_rate = accepted / eff_num_steps

    move_stat_field = bloch_phonon.vmc.StateProp.MOVE_STAT
    states_data = vmc_sampling.as_chain(num_steps)
    sys_props_set = states_data.props[num_steps_batch:]
    accepted = np.count_nonzero(sys_props_set[move_stat_field])
    chain_accept_rate = accepted / eff_num_steps

    # Both acceptance ratios should be equal.
    assert batches_accept_rate == chain_accept_rate


def test_wf_optimize():
    """Testing of the wave function optimization process."""

    boson_number = 50
    supercell_size = 50
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)
    move_spread = 0.25 * model_spec.well_width
    num_steps = 4096 * 1
    dist_type_regular = model_spec.sys_conf_dist_type.REGULAR
    offset = model_spec.well_width / 2
    ini_sys_conf = model_spec.init_get_sys_conf(dist_type=dist_type_regular,
                                                offset=offset)
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)

    wf_abs_log_field = bloch_phonon.vmc.StateProp.WF_ABS_LOG
    vmc_chain = vmc_sampling.as_chain(num_steps)
    sys_conf_set = vmc_chain.confs[:1000]
    wf_abs_log_set = vmc_chain.props[wf_abs_log_field][:1000]

    cswf_optimizer = bloch_phonon.CSWFOptimizer(model_spec, sys_conf_set,
                                                wf_abs_log_set, num_workers=2,
                                                verbose=True)
    opt_model_spec = cswf_optimizer.exec()

    print("Optimized model spec:")
    print(opt_model_spec)


def test_dmc():
    """Testing the DMC sampling."""
    lattice_depth = 0
    lattice_ratio = 1
    interaction_strength = 40
    boson_number = 50
    supercell_size = 50
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=lattice_depth,
                                   lattice_ratio=lattice_ratio,
                                   interaction_strength=interaction_strength,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)

    move_spread = 0.25 * model_spec.well_width
    num_steps = 4096 * 2
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)

    vmc_chain_data = vmc_sampling.as_chain(num_steps)
    sys_conf_set, sys_props_set, ar_ = vmc_chain_data
    print(f"Acceptance ratio: {ar_:.5g}")

    time_step = 1e-3
    num_batches = 8
    num_time_steps_batch = 128
    ini_sys_conf_set = sys_conf_set[-100:]
    target_num_walkers = 480
    max_num_walkers = 512
    ini_ref_energy = None
    rng_seed = None
    dmc_sampling = \
        bloch_phonon.dmc.Sampling(model_spec,
                                  time_step,
                                  max_num_walkers=max_num_walkers,
                                  target_num_walkers=target_num_walkers,
                                  rng_seed=rng_seed)

    num_time_steps = num_batches * num_time_steps_batch
    states = dmc_sampling.states(ini_sys_conf_set, ini_ref_energy)
    dmc_sampling_slice = islice(states, num_time_steps)
    iter_enum: dmc_base.T_E_SIter = enumerate(dmc_sampling_slice)

    for i_, state in iter_enum:
        state_data = \
            (state.energy, state.weight, state.num_walkers,
             state.ref_energy, state.accum_energy)
        state_props = np.array(state_data, dtype=dmc_base.iter_props_dtype)
        print(i_, state_props)


def test_dmc_batches():
    """Testing the DMC sampling."""
    lattice_depth = 0
    lattice_ratio = 1
    interaction_strength = 40
    boson_number = 50
    supercell_size = 50
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=lattice_depth,
                                   lattice_ratio=lattice_ratio,
                                   interaction_strength=interaction_strength,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)

    move_spread = 0.25 * model_spec.well_width
    num_steps = 4096 * 2
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)

    vmc_chain_data = vmc_sampling.as_chain(num_steps)
    sys_conf_set, sys_props_set, ar_ = vmc_chain_data
    print(f"Acceptance ratio: {ar_:.5g}")

    time_step = 1e-3
    num_batches = 8
    num_time_steps_batch = 128
    ini_sys_conf_set = sys_conf_set[-100:]
    target_num_walkers = 480
    max_num_walkers = 512
    ini_ref_energy = None
    rng_seed = None
    dmc_sampling = \
        bloch_phonon.dmc.Sampling(model_spec,
                                  time_step,
                                  max_num_walkers=max_num_walkers,
                                  target_num_walkers=target_num_walkers,
                                  rng_seed=rng_seed)

    sampling_batches = dmc_sampling.batches(num_time_steps_batch,
                                            ini_sys_conf_set,
                                            ini_ref_energy)
    dmc_sampling_batches: dmc_base.T_SBatchesIter = \
        islice(sampling_batches, num_batches)

    for batch in dmc_sampling_batches:
        state_props = batch.iter_props
        print(state_props)


def test_dmc_energy():
    """Testing the energy calculation during the DMC sampling."""
    boson_number = 50
    supercell_size = 50
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)

    move_spread = 0.25 * model_spec.well_width
    num_steps = 1024 * 1
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)
    vmc_chain_data = vmc_sampling.as_chain(num_steps)
    sys_conf_set, sys_props_set, ar_ = vmc_chain_data

    time_step = 1e-2
    num_batches = 4
    num_time_steps_batch = 128
    ini_sys_conf_set = sys_conf_set[-128:]
    target_num_walkers = 480
    max_num_walkers = 512
    ini_ref_energy = None
    rng_seed = None
    dmc_sampling = \
        bloch_phonon.dmc.Sampling(model_spec,
                                  time_step,
                                  max_num_walkers=max_num_walkers,
                                  target_num_walkers=target_num_walkers,
                                  rng_seed=rng_seed)

    # Alias.
    energy_batch = dmc_sampling.energy_batch
    energy_field = bloch_phonon.dmc.IterProp.ENERGY

    sampling_batches = dmc_sampling.batches(num_time_steps_batch,
                                            ini_sys_conf_set,
                                            ini_ref_energy)
    dmc_sampling_batches: dmc_base.T_SBatchesIter = \
        islice(sampling_batches, num_batches)

    for iter_data in dmc_sampling_batches:
        #
        energy_result = energy_batch(iter_data)
        egy = energy_result.func
        iter_props = iter_data.iter_props
        iter_energy = iter_props[energy_field]
        print(np.stack((egy, iter_energy), axis=-1))
        assert np.allclose(egy, iter_energy)


def test_dmc_batch_func():
    """Testing functions evaluated over DMC sampling data."""
    boson_number = 50
    supercell_size = 50
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)

    move_spread = 0.25 * model_spec.well_width
    num_steps = 1024 * 1
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)
    vmc_chain_data = vmc_sampling.as_chain(num_steps)
    sys_conf_set, sys_props_set, ar_ = vmc_chain_data

    time_step = 1e-2
    num_batches = 8
    num_time_steps_batch = 256
    ini_sys_conf_set = sys_conf_set[-128:]
    target_num_walkers = 480
    max_num_walkers = 512
    ini_ref_energy = None
    rng_seed = None
    dmc_sampling = \
        bloch_phonon.dmc.Sampling(model_spec,
                                  time_step,
                                  max_num_walkers=max_num_walkers,
                                  target_num_walkers=target_num_walkers,
                                  rng_seed=rng_seed)

    # The momentum range for the structure factor.
    nop = model_spec.boson_number
    sc_size = model_spec.supercell_size
    kz = np.arange(1., nop + 1) * 2 * np.pi / sc_size

    # Alias.
    structure_factor_batch = dmc_sampling.structure_factor_batch
    weight_field = bloch_phonon.dmc.IterProp.WEIGHT.value

    sampling_batches = dmc_sampling.batches(num_time_steps_batch,
                                            ini_sys_conf_set,
                                            ini_ref_energy)

    dmc_sampling_batches: dmc_base.T_SBatchesIter = \
        islice(sampling_batches, num_batches)

    for iter_data in dmc_sampling_batches:
        #
        sk_result = structure_factor_batch(kz, iter_data)
        sk = sk_result.func
        iter_props = sk_result.iter_props
        iter_weights = iter_props[weight_field]
        sk_average = sk.sum(axis=0) / iter_weights.sum(axis=0) / nop
        print(np.stack((kz, sk_average), axis=-1))


def test_dmc_est_sampling():
    """Testing the DMC sampling to evaluate several estimators."""

    lattice_depth = 0
    lattice_ratio = 1
    interaction_strength = 4
    boson_number = 30
    supercell_size = 30
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=lattice_depth,
                                   lattice_ratio=lattice_ratio,
                                   interaction_strength=interaction_strength,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)

    move_spread = 0.25 * model_spec.well_width
    num_steps = 1024 * 1
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)

    vmc_chain_data = vmc_sampling.as_chain(num_steps)
    sys_conf_set, sys_props_set, ar_ = vmc_chain_data
    print(f"Acceptance ratio: {ar_:.5g}")

    time_step = 1e-3
    num_batches = 16
    num_time_steps_batch = 32
    ini_sys_conf_set = sys_conf_set[-128:]
    target_num_walkers = 480
    max_num_walkers = 512
    ini_ref_energy = None
    rng_seed = None

    sf_config = bloch_phonon.dmc.StructureFactorEst(num_modes=100)
    dmc_sampling = \
        bloch_phonon.dmc.EstSampling(model_spec,
                                     time_step,
                                     max_num_walkers=max_num_walkers,
                                     target_num_walkers=target_num_walkers,
                                     rng_seed=rng_seed,
                                     structure_factor=sf_config)

    dmc_es_batches = dmc_sampling.batches(num_time_steps_batch,
                                          ini_sys_conf_set,
                                          ini_ref_energy)

    es_batches: dmc_base.T_ESBatchesIter = \
        islice(dmc_es_batches, num_batches)

    for batch in es_batches:
        state_props = batch.iter_props
        nw_iter = state_props[dmc_base.IterProp.NUM_WALKERS]
        sf_iter = batch.iter_structure_factor
        # print(state_props)
        sf_batch_data = sf_iter / nw_iter[:, np.newaxis]
        print(sf_batch_data)
        print(nw_iter)
        print('---')


if __name__ == '__main__':
    test_dmc()
