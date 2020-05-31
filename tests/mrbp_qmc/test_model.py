import attr
import numpy as np
import pytest

from phd_qmclib import mrbp_qmc
from phd_qmclib.mrbp_qmc.model import DIST_REGULAR
from phd_qmclib.qmc_base.jastrow import SysConfSlot

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
    model_spec = mrbp_qmc.Spec(**BASE_SPEC_ITEMS)
    print(repr(model_spec))
    print(attr.asdict(model_spec))


def test_update_params():
    """

    :return:
    """
    model_spec = mrbp_qmc.Spec(**BASE_SPEC_ITEMS)
    with pytest.raises(AttributeError):
        # Extra parameter. This will fail.
        new_params = dict(BASE_SPEC_ITEMS, extra_param=True)
        for name, value in new_params.items():
            setattr(model_spec, name, value)


def test_wf_abs_log():
    """Test the execution of the wave function."""
    model_spec = mrbp_qmc.Spec(**BASE_SPEC_ITEMS)
    core_funcs = mrbp_qmc.core_funcs

    wf_abs_log = core_funcs.wf_abs_log
    sys_conf = model_spec.init_get_sys_conf(DIST_REGULAR)
    wf_abs_log_v = wf_abs_log(sys_conf, *model_spec.cfc_spec)
    print(wf_abs_log_v)


def test_qmc_funcs():
    """"""

    # We have an ideal system...
    model_spec = mrbp_qmc.Spec(**BASE_SPEC_ITEMS)
    core_funcs = mrbp_qmc.CoreFuncs()

    # Generate a random configuration, pick the model parameters.
    sys_conf = model_spec.init_get_sys_conf()
    cfc_spec = model_spec.cfc_spec

    # Testing a scalar function with own arguments
    energy_func = core_funcs.energy
    energy_v = energy_func(sys_conf, *cfc_spec)

    # Testing an array function with no own arguments
    drift = core_funcs.drift
    out_sys_conf = drift(sys_conf, *cfc_spec)

    epp = energy_v / BOSON_NUMBER
    print("The energy per particle is: {:.6g}".format(epp))

    drift_values = out_sys_conf[SysConfSlot.drift, :]
    print("The drift is: {}".format(drift_values))

    # Testing that the array function do not modify its inputs
    in_pos_values = sys_conf[SysConfSlot.pos, :]
    out_pos_values = out_sys_conf[SysConfSlot.pos, :]
    assert np.alltrue(out_pos_values == in_pos_values)

    with pytest.raises(AssertionError):
        # Testing that the array function modified the output array
        # where expected.
        in_pos_values = sys_conf[SysConfSlot.drift, :]
        out_pos_values = out_sys_conf[SysConfSlot.drift, :]
        assert np.alltrue(out_pos_values == in_pos_values)


if __name__ == '__main__':
    test_qmc_funcs()
