import operator
from abc import ABCMeta
from collections import OrderedDict
from functools import reduce
from math import sqrt

from numba import jit

from thesis_lib.ideal import eigen_energy
from thesis_lib.qmc_lib import jastrow
from thesis_lib.utils import cached_property
from .. import trial_funcs as tf
from ..jastrow import ModelBase, QMCFuncsBase

__all__ = [
    'ArrayGUFuncBase',
    'GUFuncBase',
    'Model',
    'NOAArrayGUFuncBase',
    'NOAScalarGUFuncBase',
    'QMCFuncs',
    'ScalarGUFuncBase'
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
    def full_params(self):
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


class GUFuncBase(jastrow.BaseGUFuncBase, metaclass=ABCMeta):
    """A generalized universal function interface for compatible functions
    with the Bijl-Jastrow QMC model.
    """

    @property
    def as_model_args(self):
        """Takes the model parameters from an array."""

        @jit(nopython=True)
        def _as_model_args(model_all_params):
            """Takes the model parameters from the array, and group them
            in tuples. The constructed tuples are returned to the caller so
            they an be user as arguments for a ``QMCFuncs`` function.
            """
            # TODO: Is there a better way to do this?
            v0 = model_all_params[0]
            r = model_all_params[1]
            gn = model_all_params[2]
            nop = model_all_params[3]
            sc_size = model_all_params[4]
            model_params = v0, r, gn, nop, sc_size

            #
            v0 = model_all_params[5]
            r = model_all_params[6]
            e0 = model_all_params[7]
            k1 = model_all_params[8]
            kp1 = model_all_params[9]
            obf_params = v0, r, e0, k1, kp1

            #
            rm = model_all_params[10]
            sc_size = model_all_params[11]
            k2 = model_all_params[12]
            beta = model_all_params[13]
            r_off = model_all_params[14]
            am = model_all_params[15]
            tbf_params = rm, sc_size, k2, beta, r_off, am

            # NOTICE: This way to access data may generate corrupted results.
            # v0, r, gn, nop, sc_size = model_all_params[0:5]
            # v0, r, e0, k1, kp1 = model_all_params[5:10]
            # rm, scs, k2, beta, r_off, am = model_all_params[10:16]
            #
            return model_params, obf_params, tbf_params

        return _as_model_args


class ScalarGUFuncBase(GUFuncBase,
                       jastrow.ScalarGUFuncBase,
                       metaclass=ABCMeta):
    """"""

    def __init__(self, base_func, target=None):
        """"""
        signatures = [
            'void(f8[:,:],f8[:],f8[:],f8[:])'
        ]
        layout = '(ns,nop),(p1),(p2)->()'
        super().__init__(base_func, signatures, layout, target)


class ArrayGUFuncBase(GUFuncBase,
                      jastrow.ArrayGUFuncBase,
                      metaclass=ABCMeta):
    """"""

    def __init__(self, base_func, target=None):
        """"""
        signatures = [
            'void(f8[:,:],f8[:],f8[:],f8[:,:])'
        ]
        layout = '(ns,nop),(p1),(p2)->(ns,nop)'
        super().__init__(base_func, signatures, layout, target)


class NOAScalarGUFuncBase(GUFuncBase,
                          jastrow.NOAScalarGUFuncBase,
                          metaclass=ABCMeta):
    """"""

    def __init__(self, base_func, target=None):
        """"""
        signatures = [
            'void(f8[:,:],f8[:],f8[:])'
        ]
        layout = '(ns,nop),(p1)->()'
        super().__init__(base_func, signatures, layout, target)


class NOAArrayGUFuncBase(GUFuncBase,
                         jastrow.ArrayGUFuncBase,
                         metaclass=ABCMeta):
    """"""

    def __init__(self, base_func, target=None):
        """"""
        signatures = [
            'void(f8[:,:],f8[:],f8[:,:])'
        ]
        layout = '(ns,nop),(p1)->(ns,nop)'
        super().__init__(base_func, signatures, layout, target)
