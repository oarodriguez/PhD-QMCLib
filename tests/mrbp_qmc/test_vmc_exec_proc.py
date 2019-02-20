from my_research_libs.mrbp_qmc import Spec, vmc_exec


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

    vmc_proc_input = vmc_proc.build_input(ini_sys_conf)
    vmc_batch, _ = vmc_proc.exec(vmc_proc_input)


if __name__ == '__main__':
    test_proc()
