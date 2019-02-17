from my_research_libs.mrbp_qmc import Spec, dmc_exec, vmc_exec
from my_research_libs.qmc_base.jastrow import SysConfDistType


def test_proc():
    """Testing the main task to realize a DMC calculation."""

    lattice_depth = 0
    lattice_ratio = 1
    interaction_strength = 4
    boson_number = 8
    supercell_size = 8
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = Spec(lattice_depth=lattice_depth,
                      lattice_ratio=lattice_ratio,
                      interaction_strength=interaction_strength,
                      boson_number=boson_number,
                      supercell_size=supercell_size,
                      tbf_contact_cutoff=tbf_contact_cutoff)
    move_spread = 0.25 * model_spec.well_width
    ini_sys_conf = model_spec.init_get_sys_conf()
    rng_seed = None
    num_batches = 8
    num_steps_batch = 4096
    # num_steps = num_batches * num_steps_batch
    vmc_proc = vmc_exec.Proc(model_spec,
                             move_spread,
                             rng_seed=rng_seed,
                             num_batches=num_batches,
                             num_steps_batch=num_steps_batch)

    time_step = 1e-3
    num_batches = 4
    num_time_steps_batch = 512
    # ini_sys_conf_set = None
    target_num_walkers = 480
    max_num_walkers = 512
    # ini_ref_energy = None
    rng_seed = None

    num_modes = 2 * boson_number
    ssf_spec = dmc_exec.SSFEstSpec(num_modes=num_modes)
    dmc_proc = \
        dmc_exec.Proc(model_spec,
                      time_step,
                      max_num_walkers,
                      target_num_walkers,
                      rng_seed=rng_seed,
                      num_batches=num_batches,
                      num_time_steps_batch=num_time_steps_batch,
                      ssf_spec=ssf_spec)

    vmc_proc_input = vmc_proc.build_input(ini_sys_conf)
    vmc_batch, _ = vmc_proc.exec(vmc_proc_input)

    dmc_proc_input = dmc_proc.build_input_from_model(SysConfDistType.RANDOM)
    dmc_result = dmc_proc.exec(dmc_proc_input)


if __name__ == '__main__':
    test_proc()
