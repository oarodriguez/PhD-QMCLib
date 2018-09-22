import pytest

from thesis_lib.multirods_qmc import bp_jastrow

v0, r, gn = 100, 1, 1
nop = 100
sc_size = 100
rm = .25 * sc_size

# Well-formed parameters.
correct_params = dict(lattice_depth=v0,
                      lattice_ratio=r,
                      interaction_strength=gn,
                      boson_number=nop,
                      supercell_size=sc_size)


def test_init():
    """"""

    with pytest.raises(KeyError):
        # Extra parameter. This will fail.
        params = dict(correct_params, extra_param=True)
        bp_jastrow.Model(params)

    with pytest.raises(KeyError):
        # Missing parameter. This will fail too.
        params = dict(correct_params)
        params.pop('boson_number')
        bp_jastrow.Model(params)

    # With and without initial variational parameters
    bp_jastrow.Model(correct_params)
    bp_jastrow.Model(correct_params, var_params={})


def test_update_params():
    """

    :return:
    """
    model = bp_jastrow.Model(correct_params)
    with pytest.raises(KeyError):
        # Extra parameter. This will fail.
        new_params = dict(correct_params, extra_param=True)
        model.update_params(new_params)

    # Missing parameter. This should not fail.
    new_params = dict(correct_params)
    new_params.pop('boson_number')
    model.update_params(new_params)

    # Current parameter should remain intact
    assert model.params[model.ParamsSlots.BOSON_NUMBER] == nop

    # This will pass...
    model = bp_jastrow.Model(correct_params)
    var_params = dict(tbf_contact_cutoff=rm)
    model.update_var_params(var_params)
    pdf_params = model.wf_params
    print(correct_params, pdf_params)


def test_strict_model():
    """

    :return:
    """
    with pytest.raises(ValueError):
        bp_jastrow.StrictModel(correct_params, None)

    var_params = dict(tbf_contact_cutoff=rm)
    model = bp_jastrow.StrictModel(correct_params, var_params)

    with pytest.raises(AttributeError):
        model.update_params(correct_params)
        model.update_params(var_params)
