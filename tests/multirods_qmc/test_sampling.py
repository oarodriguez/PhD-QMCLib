from thesis_lib.multirods_qmc import bloch_phonon

v0, r, gn = 100, 1, 1
nop = 100
sc_size = 100
rm = .25 * sc_size

# Well-formed parameters.
model_params = dict(lattice_depth=v0,
                    lattice_ratio=r,
                    interaction_strength=gn,
                    boson_number=nop,
                    supercell_size=sc_size)
var_params = dict(tbf_contact_cutoff=rm)


def test_base_sampling():
    """

    :return:
    """
    # TODO: Improve this test.
    model = bloch_phonon.Model(model_params, var_params)

    ini_sys_conf = model.init_get_sys_conf()
    sampling_params = dict(move_spread=0.05,
                           ini_sys_conf=ini_sys_conf,
                           chain_samples=10,
                           burn_in_samples=0,
                           rng_seed=1)
    sampling = bloch_phonon.UniformSampling(model, sampling_params)
    for data in sampling:
        sys_conf, wfv, stat = data
        print(sys_conf[model.SysConfSlots.POS_SLOT], stat)

    chain_data = sampling.as_chain()
    print(chain_data)
