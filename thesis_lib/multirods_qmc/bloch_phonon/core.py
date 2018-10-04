import operator
from abc import ABCMeta
from collections import OrderedDict
from enum import unique
from functools import reduce
from math import sqrt

from numba import jit

from thesis_lib import ideal, qmc_lib
from thesis_lib.utils import cached_property
from .. import jastrow, trial_funcs as tf

__all__ = [
    'ArrayGUFunc',
    'ArrayGUPureFunc',
    'Model',
    'ModelFuncs',
    'ModelParams',
    'ModelVarParams',
    'ScalarGUFunc',
    'ScalarGUPureFunc'
]

# Some alias..
_two_body_func_params = tf.two_body_func_match_params


@unique
class ParamName(qmc_lib.core.ParamNameEnum):
    """Enumerates the parameters of the model (Bijl-Jastrow type) of a
    quantum system in a Multi-Rods periodic structure with repulsive,
    contact interactions.
    """
    LATTICE_DEPTH = 'lattice_depth'
    LATTICE_RATIO = 'lattice_ratio'
    INTERACTION_STRENGTH = 'interaction_strength'
    BOSON_NUMBER = 'boson_number'
    SUPERCELL_SIZE = 'supercell_size'


class ModelParams(qmc_lib.ParamsSet):
    """Represents the parameters of the model."""
    names = ParamName


@unique
class VarParamName(qmc_lib.ParamNameEnum):
    """Enumerates the variational parameters of the wave function of the
    model (Bijl-Jastrow type) of a quantum system in a Multi-Rods periodic
    structure with repulsive, contact interactions.
    """
    TBF_CONTACT_CUTOFF = 'tbf_contact_cutoff'


class ModelVarParams(qmc_lib.ParamsSet):
    """Represents the variational parameters of the model"""
    names = VarParamName


class Model(jastrow.Model):
    """QMC model for a system with a trial wave function of the
    Bijl-Jastrow type.
    """
    #
    params_cls = ModelParams
    var_params_cls = ModelVarParams

    @property
    def obf_args(self):
        """

        :return:
        """
        v0 = self.lattice_depth
        r = self.lattice_ratio

        e0 = float(ideal.eigen_energy(v0, r))
        k1, kp1 = sqrt(e0), sqrt(v0 - e0)
        return v0, r, e0, k1, kp1

    @property
    def tbf_args(self):
        """

        :return:
        """
        gn = self.interaction_strength
        nop = self.boson_number
        sc_size = self.supercell_size
        var_params = self.var_params

        # Convert to float, as numba jit-functions will not accept
        # other type.
        rm = float(var_params[self.var_params_cls.names.TBF_CONTACT_CUTOFF])
        return (rm, sc_size) + _two_body_func_params(gn, nop, rm, sc_size)

    @property
    def wf_args(self):
        """"""
        obf_args = self.obf_args
        tbf_args = self.tbf_args
        return obf_args, tbf_args

    @property
    def func_args(self):
        """"""
        self_args = self.args
        obf_args = self.obf_args
        tbf_args = self.tbf_args
        return self_args, obf_args, tbf_args

    @property
    def energy_args(self):
        """"""
        v0 = self.lattice_depth
        r = self.lattice_ratio
        gn = self.interaction_strength
        return v0, r, gn

    @property
    def full_args(self):
        """Concatenate the :attr:`Model.func_args` tuples and returns
        a single tuple.
        """
        return reduce(operator.add, self.func_args)

    @property
    def var_params_bounds(self):
        """

        :return:
        """
        sc_size = self.supercell_size
        names = self.var_params_cls.names
        bounds = [
            (names.TBF_CONTACT_CUTOFF.value, (5e-3, (0.5 - 5e-3) * sc_size))
        ]
        return OrderedDict(bounds)

    @cached_property
    def funcs(self):
        """"""
        return ModelFuncs()


class ModelFuncs(jastrow.ModelFuncs):
    """Functions of a QMC model for a system with a trial wave function
    of the Bijl-Jastrow type.
    """
    #
    params_cls = ModelParams
    var_params_cls = ModelVarParams

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


@jit(nopython=True, cache=True)
def _as_model_args(model_full_params):
    """Takes the model parameters from the array, and group them
    in tuples. The constructed tuples are returned to the caller so
    they an be user as arguments for a ``ModelFuncs`` function.
    """
    # TODO: Is there a better way to do this?
    v0 = model_full_params[0]
    r = model_full_params[1]
    gn = model_full_params[2]
    nop = model_full_params[3]
    sc_size = model_full_params[4]
    model_params = v0, r, gn, nop, sc_size

    #
    v0 = model_full_params[5]
    r = model_full_params[6]
    e0 = model_full_params[7]
    k1 = model_full_params[8]
    kp1 = model_full_params[9]
    obf_params = v0, r, e0, k1, kp1

    #
    rm = model_full_params[10]
    sc_size = model_full_params[11]
    k2 = model_full_params[12]
    beta = model_full_params[13]
    r_off = model_full_params[14]
    am = model_full_params[15]
    tbf_params = rm, sc_size, k2, beta, r_off, am

    # NOTICE: This way to access data may generate corrupted results.
    # v0, r, gn, nop, sc_size = model_full_params[0:5]
    # v0, r, e0, k1, kp1 = model_full_params[5:10]
    # rm, scs, k2, beta, r_off, am = model_full_params[10:16]
    #
    return model_params, obf_params, tbf_params


class ArrayGUFunc(qmc_lib.jastrow.ArrayGUFunc, metaclass=ABCMeta):
    """"""

    signatures = ['void(f8[:,:],f8[:],f8[:],f8[:,:])']
    layout = '(ns,nop),(p1),(p2)->(ns,nop)'

    @cached_property
    def as_model_args(self):
        """Takes the model parameters from an array."""
        return _as_model_args

    def __init__(self, base_func, target=None):
        """

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)


class ScalarGUFunc(ArrayGUFunc, qmc_lib.jastrow.ScalarGUFunc,
                   metaclass=ABCMeta):
    """"""
    signatures = ['void(f8[:,:],f8[:],f8[:],f8[:])']
    layout = '(ns,nop),(p1),(p2)->()'

    def __init__(self, base_func, target=None):
        """"""
        super().__init__(base_func, target)


class ArrayGUPureFunc(qmc_lib.jastrow.ArrayGUPureFunc, metaclass=ABCMeta):
    """"""

    signatures = ['void(f8[:,:],f8[:],f8[:,:])']
    layout = '(ns,nop),(p1)->(ns,nop)'

    @cached_property
    def as_model_args(self):
        """"""
        return _as_model_args

    def __init__(self, base_func, target=None):
        """

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)


class ScalarGUPureFunc(ArrayGUPureFunc, qmc_lib.jastrow.ScalarGUPureFunc,
                       metaclass=ABCMeta):
    """"""
    signatures = ['void(f8[:,:],f8[:],f8[:])']
    layout = '(ns,nop),(p1)->()'

    def __init__(self, base_func, target=None):
        """"""
        super().__init__(base_func, target)
