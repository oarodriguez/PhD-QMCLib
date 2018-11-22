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

    ncs, bis = 1000, 0
    time_step = 0.025 ** 2
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             time_step=time_step,
                                             chain_samples=ncs,
                                             ini_sys_conf=ini_sys_conf,
                                             burn_in_samples=bis,
                                             rng_seed=1)
    ar = 0
    for data in vmc_sampling:
        sys_conf, wfv, stat = data
        ar += stat
    ar /= ncs

    chain_data = vmc_sampling.as_chain()
    sys_conf_chain, wf_abs_log_chain, ar_ = chain_data

    assert sys_conf_chain.shape == (ncs, len(model_spec.sys_conf_slots), nop)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
