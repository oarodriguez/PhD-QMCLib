from my_research_libs.multirods_qmc import bloch_phonon

v0, r, gn = 100, 1, 1
nop = 100
sc_size = 100
rm = .25 * sc_size

# Well-formed parameters.
spec_items = dict(lattice_depth=v0,
                  lattice_ratio=r,
                  interaction_strength=gn,
                  boson_number=nop,
                  supercell_size=sc_size,
                  tbf_contact_cutoff=rm)


def test_base_sampling():
    """

    :return:
    """
    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(**spec_items)
    time_step = 0.025 ** 2
    num_steps = 4096 * 8
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             time_step=time_step,
                                             num_steps=num_steps,
                                             ini_sys_conf=ini_sys_conf,
                                             rng_seed=1)
    ar = 0
    for data in vmc_sampling:
        sys_conf, wfv, stat = data
        ar += stat
    ar /= num_steps

    chain_data = vmc_sampling.as_chain()
    sys_conf_chain, wf_abs_log_chain, ar_ = chain_data

    num_slots = len(model_spec.sys_conf_slots)
    assert sys_conf_chain.shape == (num_steps, num_slots, nop)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
