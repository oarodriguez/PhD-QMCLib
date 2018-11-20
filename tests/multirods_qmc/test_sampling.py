from numba import jit

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


class WFGUFunc(bloch_phonon.ScalarGUPureFunc):
    """"""
    pass


class EnergyGUFunc(bloch_phonon.ScalarGUFunc):
    """"""

    @property
    def as_func_args(self):
        """"""

        @jit(nopython=True, cache=True)
        def _as_func_args(func_params):
            """"""
            v0_ = func_params[0]
            r_ = func_params[1]
            gn_ = func_params[2]
            func_args_0 = v0_, r_, gn_

            return func_args_0,

        return _as_func_args


def test_base_sampling():
    """

    :return:
    """
    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(**spec_items)

    ncs, bis = 1000, 0
    time_step = 0.025 ** 2
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_spec = bloch_phonon.vmc.VMCSpec(model_spec=model_spec,
                                        time_step=time_step,
                                        chain_samples=ncs,
                                        ini_sys_conf=ini_sys_conf,
                                        burn_in_samples=bis,
                                        rng_seed=1)
    vmc_generator = bloch_phonon.vmc.vmc_core_funcs.generator
    vmc_as_chain = bloch_phonon.vmc.vmc_core_funcs.as_chain
    ar = 0
    for data in vmc_generator(*vmc_spec.cfc_spec_nt):
        sys_conf, wfv, stat = data
        ar += stat
    ar /= ncs

    chain_data = vmc_as_chain(*vmc_spec.cfc_spec_nt)
    sys_conf_chain, wf_abs_log_chain, ar_ = chain_data

    assert sys_conf_chain.shape == (ncs, len(model_spec.sys_conf_slots), nop)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
