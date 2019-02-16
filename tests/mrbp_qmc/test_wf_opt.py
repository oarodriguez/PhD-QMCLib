from my_research_libs import mrbp_qmc
from my_research_libs.qmc_base.jastrow import SysConfDistType

LATTICE_DEPTH = 100
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 1
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
num_steps = 4096 * 1
dist_type_regular = SysConfDistType.REGULAR
offset = model_spec.well_width / 2
ini_sys_conf = model_spec.init_get_sys_conf(dist_type=dist_type_regular,
                                            offset=offset)
vmc_sampling = mrbp_qmc.vmc.Sampling(model_spec=model_spec,
                                     move_spread=move_spread,
                                     rng_seed=1)


def test_wf_optimize():
    """Testing of the wave function optimization process."""

    wf_abs_log_field = mrbp_qmc.vmc.StateProp.WF_ABS_LOG
    vmc_chain = vmc_sampling.as_chain(num_steps, ini_sys_conf)
    sys_conf_set = vmc_chain.confs[:1000]
    wf_abs_log_set = vmc_chain.props[wf_abs_log_field][:1000]

    cswf_optimizer = mrbp_qmc.CSWFOptimizer(model_spec, sys_conf_set,
                                            wf_abs_log_set, num_workers=2,
                                            verbose=True)
    opt_model_spec = cswf_optimizer.exec()

    print("Optimized model spec:")
    print(opt_model_spec)


if __name__ == '__main__':
    test_wf_optimize()
