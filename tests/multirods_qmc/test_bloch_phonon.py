import numpy as np
import pytest
from matplotlib import pyplot

from my_research_libs.multirods_qmc import bloch_phonon

LATTICE_DEPTH = 100
LATTICE_RATIO = 1
INTERACTION_STRENGTH = 1
BOSON_NUMBER = 100
SUPERCELL_SIZE = 100
TBF_CONTACT_CUTOFF = .25 * SUPERCELL_SIZE

# Well-formed parameters.
BASE_SPEC_ITEMS = dict(lattice_depth=LATTICE_DEPTH,
                       lattice_ratio=LATTICE_RATIO,
                       interaction_strength=INTERACTION_STRENGTH,
                       boson_number=BOSON_NUMBER,
                       supercell_size=SUPERCELL_SIZE,
                       tbf_contact_cutoff=TBF_CONTACT_CUTOFF)


def test_init():
    """"""

    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    print(repr(model_spec))


def test_update_params():
    """

    :return:
    """
    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    with pytest.raises(AttributeError):
        # Extra parameter. This will fail.
        new_params = dict(BASE_SPEC_ITEMS, extra_param=True)
        for name, value in new_params.items():
            setattr(model_spec, name, value)


def test_qmc_funcs():
    """"""

    # We have an ideal system...
    model_spec = bloch_phonon.Spec(**BASE_SPEC_ITEMS)
    core_funcs = bloch_phonon.CoreFuncs()

    # Generate a random configuration, pick the model parameters.
    sys_conf = model_spec.init_get_sys_conf()
    cfc_spec = model_spec.cfc_spec_nt

    # Testing a scalar function with own arguments
    energy_func = core_funcs.energy
    energy_v = energy_func(sys_conf, cfc_spec)

    # Testing an array function with no own arguments
    drift = core_funcs.drift
    out_sys_conf = drift(sys_conf, cfc_spec)

    epp = energy_v / BOSON_NUMBER
    print("The energy per particle is: {:.6g}".format(epp))

    drift_values = out_sys_conf[model_spec.sys_conf_slots.drift, :]
    print("The drift is: {}".format(drift_values))

    # Testing that the array function do not modify its inputs
    in_pos_values = sys_conf[model_spec.sys_conf_slots.pos, :]
    out_pos_values = out_sys_conf[model_spec.sys_conf_slots.pos, :]
    assert np.alltrue(out_pos_values == in_pos_values)

    with pytest.raises(AssertionError):
        # Testing that the array function modified the output array
        # where expected.
        in_pos_values = sys_conf[model_spec.sys_conf_slots.drift, :]
        out_pos_values = out_sys_conf[model_spec.sys_conf_slots.drift, :]
        assert np.alltrue(out_pos_values == in_pos_values)


def test_vmc():
    """

    :return:
    """
    boson_number = 10
    supercell_size = 10
    tbf_contact_cutoff = 0.25 * supercell_size

    # TODO: Improve this test.
    model_spec = bloch_phonon.Spec(lattice_depth=LATTICE_DEPTH,
                                   lattice_ratio=LATTICE_RATIO,
                                   interaction_strength=INTERACTION_STRENGTH,
                                   boson_number=boson_number,
                                   supercell_size=supercell_size,
                                   tbf_contact_cutoff=tbf_contact_cutoff)
    move_spread = 0.25 * model_spec.well_width
    num_steps = 4096 * 128
    ini_sys_conf = model_spec.init_get_sys_conf()
    vmc_sampling = bloch_phonon.vmc.Sampling(model_spec=model_spec,
                                             move_spread=move_spread,
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
    assert sys_conf_chain.shape == (num_steps, num_slots, boson_number)
    assert ar == ar_

    print(f"Sampling acceptance rate: {ar:.5g}")
    pos_slot = model_spec.sys_conf_slots.pos

    ax = pyplot.gca()
    pos = sys_conf_chain[:, pos_slot]
    ax.hist(pos.flatten(), bins=20 * supercell_size)
    pyplot.show()
    print(sys_conf_chain)
