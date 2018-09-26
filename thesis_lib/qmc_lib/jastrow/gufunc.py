from abc import ABCMeta, abstractmethod

import numpy as np
from numba import guvectorize, jit

from thesis_lib.utils import cached_property
from .. import abc as qmc_lib_abc

__all__ = [
    'ArrayGUFunc',
    'ArrayGUPureFunc',
    'ScalarGUFunc',
    'ScalarGUPureFunc'
]


class ArrayGUFunc(qmc_lib_abc.GUFunc, metaclass=ABCMeta):
    """"""

    @property
    @abstractmethod
    def as_func_args(self):
        """Takes the function parameters from an array."""
        pass

    @property
    @abstractmethod
    def as_model_args(self):
        """Takes the model parameters from an array."""
        pass

    @cached_property
    def as_elem_func_args(self):
        """"""
        as_func_args = self.as_func_args
        as_model_args = self.as_model_args

        @jit(nopython=True, cache=True)
        def _as_elem_func_args(func_params, model_params):
            """"""
            func_args = as_func_args(func_params)
            model_args = as_model_args(model_params)
            return func_args + model_args

        return _as_elem_func_args

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf: np.ndarray,
                       func_params: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(func_params, model_params)
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
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, func_params, model_params, result)

        return _core_func

    def __call__(self, sys_conf: np.ndarray,
                 func_params: np.ndarray,
                 model_params: np.ndarray, *,
                 result: np.ndarray = None):
        """

        :param sys_conf:
        :param func_params:
        :param model_params:
        :param result:
        :return:
        """
        sys_conf = np.asarray(sys_conf)
        func_params = np.asarray(func_params)
        model_params = np.asarray(model_params)
        return self.core_func(sys_conf, func_params, model_params, result)


class ScalarGUFunc(ArrayGUFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf: np.ndarray,
                       func_params: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(func_params, model_params)
            result[0] = _base_func(sys_conf, *func_args)

        return _elem_func


class ArrayGUPureFunc(qmc_lib_abc.GUFunc, metaclass=ABCMeta):
    """"""

    @property
    def as_func_args(self):
        """Takes the function parameters from an array."""
        return None

    @property
    @abstractmethod
    def as_model_args(self):
        """Takes the model parameters from an array."""
        pass

    @cached_property
    def as_elem_func_args(self):
        """"""
        as_model_args = self.as_model_args

        @jit(nopython=True, cache=True)
        def _as_elem_func_args(model_params):
            """"""
            return as_model_args(model_params)

        return _as_elem_func_args

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(model_params)
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
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, model_params, result)

        return _core_func

    def __call__(self, sys_conf: np.ndarray,
                 model_params: np.ndarray, *,
                 result: np.ndarray = None):
        """

        :param sys_conf:
        :param model_params:
        :param result:
        :return:
        """
        sys_conf = np.asarray(sys_conf)
        model_params = np.asarray(model_params)
        return self.core_func(sys_conf, model_params, result)


class ScalarGUPureFunc(ArrayGUPureFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def elem_func(self):
        """"""
        _base_func = self._base_func
        as_elem_func_args = self.as_elem_func_args

        @jit(nopython=True, cache=True)
        def _elem_func(sys_conf: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(model_params)
            result[0] = _base_func(sys_conf, *func_args)

        return _elem_func
