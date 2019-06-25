import pathlib

from matplotlib import pyplot

from my_research_libs.constants import ER
from my_research_libs.mrbp_qmc import Spec, dmc_exec

lattice_depth = 5 * ER
lattice_ratio = 1
interaction_strength = 2
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


def test_proc():
    """Testing the main task to realize a DMC calculation."""
    time_step = 6.25e-4
    num_blocks = 4
    num_time_steps_block = 4096
    target_num_walkers = 480
    max_num_walkers = 512
    # ini_ref_energy = None
    rng_seed = None

    dmc_proc = \
        dmc_exec.Proc(model_spec,
                      time_step,
                      max_num_walkers,
                      target_num_walkers,
                      rng_seed=rng_seed,
                      num_blocks=num_blocks,
                      num_time_steps_block=num_time_steps_block)

    sys_conf_spec = dmc_exec.ModelSysConfSpec(dist_type='RANDOM')
    dmc_proc_input = \
        dmc_exec.ProcInput.from_model_sys_conf_spec(sys_conf_spec, dmc_proc)
    dmc_result = dmc_proc.exec(dmc_proc_input)

    energy_mean = dmc_result.data.blocks.energy.mean
    print(energy_mean)


def test_density_proc():
    """Testing the calculation of the density."""
    time_step = 6.25e-4
    num_blocks = 2
    num_time_steps_block = 512
    target_num_walkers = 480
    max_num_walkers = 512
    # ini_ref_energy = None
    rng_seed = None

    num_bins = supercell_size * 32
    density_spec = dmc_exec.DensityEstSpec(num_bins=num_bins)
    dmc_proc = \
        dmc_exec.Proc(model_spec,
                      time_step,
                      max_num_walkers,
                      target_num_walkers,
                      rng_seed=rng_seed,
                      num_blocks=num_blocks,
                      num_time_steps_block=num_time_steps_block,
                      density_spec=density_spec)

    sys_conf_spec = dmc_exec.ModelSysConfSpec(dist_type='RANDOM')
    dmc_proc_input = \
        dmc_exec.ProcInput.from_model_sys_conf_spec(sys_conf_spec, dmc_proc)
    result = dmc_proc.exec(dmc_proc_input)

    bin_size = supercell_size / num_bins
    bins_edges = dmc_proc.sampling.density_bins_edges[:-1]
    density_mean = result.data.blocks.density.mean / bin_size

    pyplot.plot(bins_edges, density_mean)
    pyplot.xlabel(r'$z / (a + b)$')
    pyplot.ylabel(r'$n(z)$')
    pyplot.show()

    h5f_path = pathlib.Path('./test-dmc-density-results.h5')
    if h5f_path.exists():
        h5f_path.unlink()
    handler = dmc_exec.HDF5FileHandler(h5f_path, 'density-data-group')
    handler.dump(result)


def test_load_proc_density_output():
    """Load the static structure factor VMC data from HDF5 file."""
    h5f_path = pathlib.Path('./test-dmc-density-results.h5')
    handler = dmc_exec.HDF5FileHandler(h5f_path, 'density-data-group')
    result = handler.load()
    density_bin_edges = result.proc.sampling.density_bins_edges
    density_mean = result.data.blocks.density.mean

    pyplot.plot(density_bin_edges[:-1], density_mean / boson_number)
    pyplot.xlabel(r'$z / l$')
    pyplot.ylabel(r'$n_{1}(z)$')
    pyplot.show()


def test_ssf_proc():
    """Testing the calculation of the static structure factor."""
    time_step = 6.25e-4
    num_blocks = 2
    num_time_steps_block = 1024
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
                      num_blocks=num_blocks,
                      num_time_steps_block=num_time_steps_block,
                      ssf_spec=ssf_spec)

    sys_conf_spec = dmc_exec.ModelSysConfSpec(dist_type='RANDOM')
    dmc_proc_input = \
        dmc_exec.ProcInput.from_model_sys_conf_spec(sys_conf_spec, dmc_proc)
    result = dmc_proc.exec(dmc_proc_input)

    ssf_momenta = dmc_proc.sampling.ssf_momenta
    ssf_mean = result.data.blocks.ss_factor.mean

    pyplot.plot(ssf_momenta, ssf_mean)
    pyplot.xlabel(r'$k / n$')
    pyplot.ylabel(r'$S(k)$')
    pyplot.show()

    h5f_path = pathlib.Path('./test-dmc-ssf-results.h5')
    if h5f_path.exists():
        h5f_path.unlink()
    handler = dmc_exec.HDF5FileHandler(h5f_path, 'ssf-data-group')
    handler.dump(result)


def test_load_ssf_proc_output():
    """Load the static structure factor VMC data from HDF5 file."""
    h5f_path = pathlib.Path('./test-dmc-ssf-results.h5')
    handler = dmc_exec.HDF5FileHandler(h5f_path, 'ssf-data-group')
    result = handler.load()
    ssf_momenta = result.proc.sampling.ssf_momenta
    ssf_mean = result.data.blocks.ss_factor.mean

    pyplot.plot(ssf_momenta, ssf_mean / boson_number)
    pyplot.xlabel(r'$k / n$')
    pyplot.ylabel(r'$S(k)$')
    pyplot.show()


if __name__ == '__main__':
    test_proc()
