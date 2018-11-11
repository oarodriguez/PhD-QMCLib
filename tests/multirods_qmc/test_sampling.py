import numpy as np
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

    ncs, nbs = 1000, 0
    ini_sys_conf = model_spec.init_get_sys_conf()
    move_spread = 0.05
    sampling_params = dict(ini_sys_conf=ini_sys_conf,
                           chain_samples=ncs,
                           burn_in_samples=nbs,
                           rng_seed=1)
    wf_spec = model_spec.cfc_spec_nt
    ppf_spec = bloch_phonon.UTPFSpecNT(model_spec.boson_number,
                                       move_spread,
                                       *model_spec.boundaries)
    sampling = bloch_phonon.uniform_sampling_funcs
    ar = 0
    for data in sampling.generator(wf_spec, ppf_spec, **sampling_params):
        sys_conf, wfv, stat = data
        ar += stat
    ar /= ncs

    chain_data = sampling.as_chain(wf_spec, ppf_spec, **sampling_params)
    sys_conf_chain, wf_abs_log_chain, ar_ = chain_data

    assert sys_conf_chain.shape == (ncs, len(model_spec.sys_conf_slots), nop)
    assert ar == ar_

    wf_abs_log = model_spec.core_funcs.wf_abs_log
    wf_abs_log_guf = WFGUFunc(wf_abs_log)
    wf_abs_log_chain_gu = wf_abs_log_guf(sys_conf_chain,
                                         model_spec.gufunc_args)

    assert wf_abs_log_chain.shape == wf_abs_log_chain_gu.shape
    assert np.allclose(wf_abs_log_chain.shape, wf_abs_log_chain_gu.shape)
