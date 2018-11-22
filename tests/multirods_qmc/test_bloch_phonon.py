import numpy as np
import pytest

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


def test_init():
    """"""

    model_spec = bloch_phonon.Spec(**spec_items)
    print(repr(model_spec))


def test_update_params():
    """

    :return:
    """
    model_spec = bloch_phonon.Spec(**spec_items)
    with pytest.raises(AttributeError):
        # Extra parameter. This will fail.
        new_params = dict(spec_items, extra_param=True)
        for name, value in new_params.items():
            setattr(model_spec, name, value)


def test_qmc_funcs():
    """"""

    # We have an ideal system...
    model_spec = bloch_phonon.Spec(**spec_items)
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

    epp = energy_v / nop
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


