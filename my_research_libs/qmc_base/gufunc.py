from abc import ABCMeta
from typing import Any

import numpy as np
from cached_property import cached_property
from numba import guvectorize, jit

from . import model

__all__ = [
    'ArrayGUFunc',
    'ArrayGUPureFunc',
    'ScalarGUFunc',
    'ScalarGUPureFunc'
]


class ArrayGUFunc(model.GUFunc, metaclass=ABCMeta):
    """Generalized universal function interface for functions that
    evaluate an "array" (non-scalar) property over a system configuration.
    """

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf, func_params, result):
            """"""
            func_args = as_elem_func_args(func_params)
            func_args_spec = func_args + (result,)
            _base_func(sys_conf, *func_args_spec)

        return _elem_func

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       func_params: np.ndarray,
                       result: np.ndarray):
            """"""
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, func_params, result)

        return _core_func

    def __call__(self, sys_conf: np.ndarray,
                 func_params: Any,
                 result: np.ndarray = None) -> np.ndarray:
        """

        :param sys_conf:
        :param func_params:
        :param result:
        :return:
        """
        sys_conf = np.asarray(sys_conf)
        func_params = np.asarray(func_params)
        return self.core_func(sys_conf, func_params, result)


class ScalarGUFunc(ArrayGUFunc, metaclass=ABCMeta):
    """Generalized universal function interface for functions that
    evaluate a scalar property over a system configuration.
    """

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf, func_params, result):
            """"""
            func_args = as_elem_func_args(func_params)
            result[0] = _base_func(sys_conf, *func_args)

        return _elem_func


class ArrayGUPureFunc(model.GUFunc, metaclass=ABCMeta):
    """Generalized universal function interface for functions that
    evaluate an "array" (non-scalar) property over a system configuration.
    """

    @property
    def as_elem_func_args(self):
        """"""
        # Do nothing
        return None

    @cached_property
    def elem_func(self):
        """"""
        return self._base_func

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       result: np.ndarray):
            """"""
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, result)

        return _core_func

    def __call__(self, sys_conf: np.ndarray,
                 result: np.ndarray = None) -> np.ndarray:
        """

        :param sys_conf:
        :param result:
        :return:
        """
        sys_conf = np.asarray(sys_conf)
        return self.core_func(sys_conf, result)


class ScalarGUPureFunc(ArrayGUPureFunc, metaclass=ABCMeta):
    """Generalized universal function interface for functions that
    evaluate an "array" (non-scalar) property over a system configuration.
    """

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf, result):
            """"""
            result[0] = _base_func(sys_conf)

        return _elem_func
