import numpy as np
import pytest
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


def test_init():
    """"""

    model_spec = bloch_phonon.Model(**spec_items)
    print(repr(model_spec))


def test_update_params():
    """

    :return:
    """
    model_spec = bloch_phonon.Model(**spec_items)
    with pytest.raises(AttributeError):
        # Extra parameter. This will fail.
        new_params = dict(spec_items, extra_param=True)
        for name, value in new_params.items():
            setattr(model_spec, name, value)


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


def test_qmc_funcs():
    """"""

    # We have an ideal system...
    model = bloch_phonon.Model(**spec_items)
    core_funcs = bloch_phonon.ModelCoreFuncs()

    # Generate a random configuration, pick the model parameters.
    sys_conf = model.init_get_sys_conf()
    func_args = model.core_func_args
    energy_args = model.energy_args

    # Testing a scalar function with own arguments
    energy_func = core_funcs.energy
    energy_v = energy_func(sys_conf, energy_args, *func_args)

    # Testing an array function with no own arguments
    drift = core_funcs.drift
    out_sys_conf = sys_conf.copy()
    drift(sys_conf, *func_args, out_sys_conf)

    epp = energy_v / nop
    print("The energy per particle is: {:.6g}".format(epp))

    drift_values = out_sys_conf[model.sys_conf_slots.DRIFT_SLOT, :]
    print("The drift is: {}".format(drift_values))

    # Testing that the array function do not modify its inputs
    in_pos_values = sys_conf[model.sys_conf_slots.POS_SLOT, :]
    out_pos_values = out_sys_conf[model.sys_conf_slots.POS_SLOT, :]
    assert np.alltrue(out_pos_values == in_pos_values)

    with pytest.raises(AssertionError):
        # Testing that the array function modified the output array
        # where expected.
        in_pos_values = sys_conf[model.sys_conf_slots.DRIFT_SLOT, :]
        out_pos_values = out_sys_conf[model.sys_conf_slots.DRIFT_SLOT, :]
        assert np.alltrue(out_pos_values == in_pos_values)


def test_gufunc():
    """Testing the behavior of the generalized universal functions."""

    model = bloch_phonon.Model(**spec_items)
    core_funcs = bloch_phonon.ModelCoreFuncs()

    # Generate a random configuration, pick the model parameters.
    core_func_args = model.core_func_args
    energy_args = model.energy_args
    gufunc_args = model.gufunc_args
    dist_type_regular = model.sys_conf_dist_type.REGULAR
    sys_conf = model.init_get_sys_conf(dist_type=dist_type_regular)

    # Instantiate a universal function
    wf_abs_log = core_funcs.wf_abs_log
    energy = core_funcs.energy
    wf_abs_log_gufunc = WFGUFunc(wf_abs_log)
    energy_gufunc = EnergyGUFunc(energy)

    energy_v = energy(sys_conf, energy_args, *core_func_args)
    wf_abs_log_v = wf_abs_log(sys_conf, *core_func_args)

    energy_gv = energy_gufunc(sys_conf, energy_args, gufunc_args)
    wf_abs_log_gv = wf_abs_log_gufunc(sys_conf, gufunc_args)

    assert energy_gv == energy_v
    assert wf_abs_log_gv == wf_abs_log_v

    # Now let's see how broadcasting works...
    # Create an array with ``sys_conf`` repeated 1000 times
    sys_conf_copies = 1000
    sys_conf_set = np.repeat(sys_conf[np.newaxis, ...],
                             sys_conf_copies, axis=0)
    energy_args = np.asarray(energy_args)

    # GUFuncs must evaluate many times over the loop dimensions
    # In this case the only loop dimension will be ``sys_conf_copies``.
    energy_gv = energy_gufunc(sys_conf_set, energy_args, gufunc_args)
    wf_abs_log_gv = wf_abs_log_gufunc(sys_conf_set, gufunc_args)

    # Verify the equivalences.
    assert energy_gv.shape == (sys_conf_copies,)
    assert np.alltrue(energy_gv == energy_v)

    assert wf_abs_log_gv.shape == (sys_conf_copies,)
    assert np.alltrue(wf_abs_log_gv == wf_abs_log_v)
