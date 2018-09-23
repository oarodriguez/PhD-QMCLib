import operator
from collections import OrderedDict
from functools import reduce
from math import sqrt

from thesis_lib.ideal import eigen_energy
from thesis_lib.utils import cached_property
from . import trial_funcs as tf
from .jastrow import ModelBase, QMCFuncsBase

__all__ = [
    'Model',
    'QMCFuncs'
]

# Some alias..
_two_body_func_params = tf.two_body_func_match_params


class Model(ModelBase):
    """Concrete implementation of a QMC model with a trial wave function
    of the Bijl-Jastrow type and type :class:`jastrow.Model`.
    """

    @property
    def obf_params(self):
        """

        :return:
        """
        v0 = self.lattice_depth
        r = self.lattice_ratio

        e0 = float(eigen_energy(v0, r))
        k1, kp1 = sqrt(e0), sqrt(v0 - e0)
        return v0, r, e0, k1, kp1

    @property
    def tbf_params(self):
        """

        :return:
        """
        gn = self.interaction_strength
        nop = self.boson_number
        sc_size = self.supercell_size
        var_params = self.var_params

        # Convert to float, as numba jit-functions will not accept
        # other type.
        rm = float(var_params['tbf_contact_cutoff'])
        return (rm, sc_size) + _two_body_func_params(gn, nop, rm, sc_size)

    @property
    def wf_params(self):
        """"""
        obf_params = self.obf_params
        tbf_params = self.tbf_params
        return obf_params, tbf_params

    @property
    def energy_params(self):
        """"""
        v0 = self.lattice_depth
        r = self.lattice_ratio
        gn = self.interaction_strength
        return v0, r, gn

    @property
    def all_params(self):
        """Concatenate the :attr:`Model.wf_params` tuples and returns
        a single numpy array.
        """
        params = self.params
        obf_params = self.obf_params
        tbf_params = self.tbf_params
        return reduce(operator.add, (params, obf_params, tbf_params))

    @property
    def var_params_bounds(self):
        """

        :return:
        """
        sc_size = self.supercell_size
        bounds = [
            ('tbf_contact_cutoff', (5e-3, (0.5 - 5e-3) * sc_size))
        ]
        return OrderedDict(bounds)


class QMCFuncs(QMCFuncsBase):
    """"""

    def __init__(self):
        """"""
        super().__init__()

    @cached_property
    def one_body_func(self):
        """

        :return:
        """
        return tf.gs_one_body_func

    @cached_property
    def two_body_func(self):
        """

        :return: float
        """
        return tf.phonon_two_body_func

    @cached_property
    def one_body_func_log_dz(self):
        """

        :return:
        """
        return tf.gs_one_body_func_log_dz

    @cached_property
    def two_body_func_log_dz(self):
        """

        :return:
        """
        return tf.phonon_two_body_func_log_dz

    @cached_property
    def one_body_func_log_dz2(self):
        """

        :return:
        """
        return tf.gs_one_body_func_log_dz2

    @cached_property
    def two_body_func_log_dz2(self):
        """

        :return:
        """
        return tf.phonon_two_body_func_log_dz2
