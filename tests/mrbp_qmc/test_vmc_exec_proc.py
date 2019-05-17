import pathlib

from matplotlib import pyplot

from my_research_libs.constants import ER
from my_research_libs.mrbp_qmc import Spec, vmc_exec

lattice_depth = 5 * ER
lattice_ratio = 1
interaction_strength = 4
boson_number = 24
supercell_size = 24
tbf_contact_cutoff = 0.25 * supercell_size

# TODO: Improve this test.
model_spec = Spec(lattice_depth=lattice_depth,
                  lattice_ratio=lattice_ratio,
                  interaction_strength=interaction_strength,
                  boson_number=boson_number,
                  supercell_size=supercell_size,
                  tbf_contact_cutoff=tbf_contact_cutoff)


def test_proc():
    """Testing the main task to realize a DMC calculation."""
    move_spread = 0.25 * model_spec.well_width
    rng_seed = None
    num_blocks = 64
    num_steps_block = 4096
    vmc_proc = vmc_exec.Proc(model_spec,
                             move_spread,
                             rng_seed=rng_seed,
                             num_blocks=num_blocks,
                             num_steps_block=num_steps_block,
                             keep_iter_data=True)

    sys_conf_spec = vmc_exec.ModelSysConfSpec(dist_type='RANDOM')
    vmc_proc_input = \
        vmc_exec.ProcInput.from_model_sys_conf_spec(sys_conf_spec, vmc_proc)
    result = vmc_proc.exec(vmc_proc_input)
    energy_blocks = result.data.blocks.energy
    mean_energy = energy_blocks.mean
    energy_mean_error = energy_blocks.mean_error
    print(mean_energy, energy_mean_error)

    h5f_path = pathlib.Path('./test-results.h5')
    if h5f_path.exists():
        h5f_path.unlink()
    handler = vmc_exec.HDF5FileHandler(h5f_path, 'test-group')
    handler.dump(result)


def test_load_proc_output():
    """Load VMC data from HDF5 file."""
    handler = vmc_exec.HDF5FileHandler('./test-results.h5', 'test-group')
    result = handler.load()
    energy_blocks = result.data.blocks.energy
    mean_energy = energy_blocks.mean
    energy_mean_error = energy_blocks.mean_error
    print(mean_energy, energy_mean_error)


def test_ssf_proc():
    """Testing the main task to realize a DMC calculation."""
    move_spread = 0.25 * model_spec.well_width
    rng_seed = None
    num_blocks = 128
    num_steps_block = 4096
    num_modes = 2 * boson_number
    ssf_est_spec = vmc_exec.SSFEstSpec(num_modes=num_modes)
    vmc_proc = vmc_exec.Proc(model_spec,
                             move_spread,
                             rng_seed=rng_seed,
                             num_blocks=num_blocks,
                             num_steps_block=num_steps_block,
                             ssf_spec=ssf_est_spec,
                             keep_iter_data=True)

    sys_conf_spec = vmc_exec.ModelSysConfSpec(dist_type='RANDOM')
    vmc_proc_input = \
        vmc_exec.ProcInput.from_model_sys_conf_spec(sys_conf_spec, vmc_proc)
    result = vmc_proc.exec(vmc_proc_input)

    ssf_momenta = vmc_proc.sampling.ssf_momenta
    ss_factor_mean = result.data.blocks.ss_factor.mean

    pyplot.plot(ssf_momenta, ss_factor_mean / boson_number)
    pyplot.xlabel(r'$k / n$')
    pyplot.ylabel(r'$S(k)$')
    pyplot.show()

    h5f_path = pathlib.Path('./test-ssf-results.h5')
    if h5f_path.exists():
        h5f_path.unlink()
    handler = vmc_exec.HDF5FileHandler(h5f_path, 'ssf-data-group')
    handler.dump(result)


def test_load_proc_ssf_output():
    """Load the static structure factor VMC data from HDF5 file."""
    h5f_path = pathlib.Path('./test-ssf-results.h5')
    handler = vmc_exec.HDF5FileHandler(h5f_path, 'ssf-data-group')
    result = handler.load()
    ssf_momenta = result.proc.sampling.ssf_momenta
    ss_factor_mean = result.data.blocks.ss_factor.mean

    pyplot.plot(ssf_momenta, ss_factor_mean / boson_number)
    pyplot.xlabel(r'$k / n$')
    pyplot.ylabel(r'$S(k)$')
    pyplot.show()


if __name__ == '__main__':
    test_proc()
