from abc import ABCMeta, abstractmethod

import numpy as np
from numba import guvectorize, jit

from thesis_lib.utils import cached_property
from .. import abc

__all__ = [
    'ArrayGUFunc',
    'BaseGUFunc',
    'GUFunc',
    'NOAArrayGUFunc',
    'NOAGUFunc',
    'NOAScalarGUFunc',
    'ScalarGUFunc'
]


class BaseGUFunc(abc.GUFunc):
    """A generalized universal function interface for compatible functions
    with the Bijl-Jastrow QMC model.
    """

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

    @property
    @abstractmethod
    def as_elem_func_args(self):
        """"""
        pass

    @property
    def elem_func(self):
        """"""
        return self._base_func


class GUFunc(BaseGUFunc, metaclass=ABCMeta):
    """"""

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


class ScalarGUFunc(GUFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout
        as_elem_func_args = self.as_elem_func_args

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       func_params: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(func_params, model_params)
            result[0] = elem_func(sys_conf, *func_args)

        return _core_func


class ArrayGUFunc(GUFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout
        as_elem_func_args = self.as_elem_func_args

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       func_params: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(func_params, model_params)
            func_args_spec = func_args + (result,)
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, *func_args_spec)

        return _core_func


class NOAGUFunc(BaseGUFunc, metaclass=ABCMeta):
    """A generalized universal function interface for compatible functions
    with the Bijl-Jastrow QMC model.
    """

    @property
    def as_func_args(self):
        """Takes the function parameters from an array."""
        return None

    @property
    def as_elem_func_args(self):
        """"""
        as_model_args = self.as_model_args

        @jit(nopython=True, cache=True)
        def _as_elem_func_args(model_params):
            """"""
            return as_model_args(model_params)

        return _as_elem_func_args

    @property
    def elem_func(self):
        """"""
        return self._base_func

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


class NOAScalarGUFunc(NOAGUFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout
        as_elem_func_args = self.as_elem_func_args

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(model_params)
            result[0] = elem_func(sys_conf, *func_args)

        return _core_func


class NOAArrayGUFunc(NOAGUFunc, metaclass=ABCMeta):
    """"""

    @cached_property
    def core_func(self):
        """"""
        target = self.target
        elem_func = self.elem_func
        signatures = self.signatures
        layout = self.layout
        as_elem_func_args = self.as_elem_func_args

        @guvectorize(signatures, layout, nopython=True, target=target)
        def _core_func(sys_conf: np.ndarray,
                       model_params: np.ndarray,
                       result: np.ndarray):
            """"""
            func_args = as_elem_func_args(model_params)
            func_args_spec = func_args + (result,)
            # NOTE: Any loop must be done inside the elem_func
            elem_func(sys_conf, *func_args_spec)

        return _core_func
