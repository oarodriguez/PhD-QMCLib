from math import pi
from typing import Sequence

import numba
import numpy as np

from thesis_lib.qmc_lib import gufunc


@numba.njit
def scalar_base_func(sys_conf: np.ndarray,
                     func_params: Sequence[float]):
    """A jit-compiled function that returns a scalar."""
    sigma, = func_params
    return sigma * sys_conf.mean()


@numba.njit
def field_base_func(sys_conf: np.ndarray,
                    func_params: Sequence[float],
                    result: np.ndarray):
    """A jit-compiled function that receives an array buffer to store
    the results. The ``result`` buffer has the same dimensions as
    ``sys_conf``.
    """
    sigma, = func_params
    scs = sys_conf.shape[0]
    for j_ in range(scs):
        result[j_] = sigma * sys_conf[j_].mean()


class ArrayGUFunc(gufunc.ArrayGUFunc):
    """Concrete implementation of ``gufunc.ArrayGUFunc``."""

    signatures = ['void(f8[:,:],f8[:],f8[:,:])']
    layout = '(ss,ns),(nf)->(ss,ns)'

    def __init__(self, base_func, target=None):
        """

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)

    @property
    def as_elem_func_args(self):
        """"""

        @numba.njit
        def _as_elem_func_args(func_params):
            """Picks the parameters of the base function and returns them
            in the suitable shape.
            """
            sigma = func_params[0]
            func_arg_0 = sigma,
            return func_arg_0,

        return _as_elem_func_args


class ScalarGUFunc(ArrayGUFunc, gufunc.ScalarGUFunc):
    """Concrete implementation of ``gufunc.ScalarGUFunc``."""

    signatures = ['void(f8[:,:],f8[:],f8[:])']
    layout = '(ss,ns),(nf)->()'

    def __init__(self, base_func, target=None):
        """Initializer.

        :param base_func:
        :param target:
        """
        super().__init__(base_func, target)


def test_base_func_exec():
    """"""

    func = ScalarGUFunc(scalar_base_func, target='cpu')
    scs_shape = (1000, 100, 2)
    sigma = pi ** 2

    sys_conf_set = np.ones(scs_shape)
    func_params = sigma,
    result = func(sys_conf_set, func_params)

    assert result.shape == (scs_shape[0],)
    assert np.allclose(result, sigma)


def test_field_base_func_exec():
    """"""

    func = ArrayGUFunc(field_base_func, target='parallel')
    scs_shape = (1000, 100, 5)
    sigma = pi ** 2

    sys_conf_set = np.ones(scs_shape)
    func_params = sigma,
    result = func.__call__(sys_conf_set, func_params)

    assert result.shape == scs_shape
    assert np.allclose(result, sigma)
